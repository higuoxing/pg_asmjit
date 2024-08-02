#include "asmjit/x86/x86compiler.h"
#include "asmjit_common.h"

namespace jit = asmjit;
namespace x86 = asmjit::x86;

extern "C" {

static bool JitSessionInitialized = false;
static jit::JitRuntime Runtime;

static void ResOwnerReleaseJitContext(Datum res) {
  AsmJitContext *Context = (AsmJitContext *)DatumGetPointer(res);

  Context->resowner = NULL;
  jit_release_context((JitContext *)Context);
}

static const ResourceOwnerDesc jit_resowner_desc = {
    .name = "AsmJit context",
    .release_phase = RESOURCE_RELEASE_BEFORE_LOCKS,
    .release_priority = RELEASE_PRIO_JIT_CONTEXTS,
    .ReleaseResource = ResOwnerReleaseJitContext,
    .DebugPrint = NULL /* the default message is fine */
};

/* Convenience wrappers over ResourceOwnerRemember/Forget */
static inline void ResourceOwnerRememberJIT(ResourceOwner owner,
                                            AsmJitContext *handle) {
  ResourceOwnerRemember(owner, PointerGetDatum(handle), &jit_resowner_desc);
}
static inline void ResourceOwnerForgetJIT(ResourceOwner owner,
                                          AsmJitContext *handle) {
  ResourceOwnerForget(owner, PointerGetDatum(handle), &jit_resowner_desc);
}

void AsmJitReleaseContext(JitContext *Ctx) {
  AsmJitContext *Context = (AsmJitContext *)Ctx;
  ListCell *FnCell;

  /*
   * Copy&pasted from llvmjit.c
   * When this backend is exiting, don't clean up LLVM. As an error might
   * have occurred from within LLVM, we do not want to risk reentering. All
   * resource cleanup is going to happen through process exit.
   */
  if (proc_exit_inprogress)
    return;

  foreach (FnCell, Context->funcs) {
    ExprStateEvalFunc EvalFunc = (ExprStateEvalFunc)lfirst(FnCell);
    Runtime.release(EvalFunc);
  }

  list_free(Context->funcs);
  Context->funcs = NIL;

  if (Context->resowner)
    ResourceOwnerForgetJIT(Context->resowner, Context);
}

void AsmJitResetAfterError(void) { /* TODO */
}

static void JitInitializeSession(void) {
  if (JitSessionInitialized)
    return;

  JitSessionInitialized = true;
}

static AsmJitContext *JitCreateContext(int JitFlags) {
  JitInitializeSession();

  ResourceOwnerEnlarge(CurrentResourceOwner);

  AsmJitContext *Context = (AsmJitContext *)MemoryContextAllocZero(
      TopMemoryContext, sizeof(AsmJitContext));
  Context->base.flags = JitFlags;
  Context->funcs = NIL;

  /* ensure cleanup */
  Context->resowner = CurrentResourceOwner;
  ResourceOwnerRememberJIT(CurrentResourceOwner, Context);

  return Context;
}

static Datum ExecCompiledExpr(ExprState *State, ExprContext *EContext,
                              bool *IsNull) {
  ExprStateEvalFunc Func = (ExprStateEvalFunc)State->evalfunc_private;
  State->evalfunc = Func;
  State->evalfunc_private = nullptr;
  /*
   * Before executing the generated expression, we should make sure the
   * expression is still valid.
   */
  CheckExprStillValid(State, EContext);
  return Func(State, EContext, IsNull);
}

#define TYPES_INFO(struct_type, member_type, member_name, reg_type)            \
  static inline x86::Gp emit_load_##member_name##_from_##struct_type(          \
      x86::Compiler &cc, x86::Gp &object_addr) {                               \
    x86::Mem member_ptr = x86::ptr(                                            \
        object_addr, offsetof(struct_type, member_name), sizeof(member_type)); \
    x86::Gp member = cc.new##reg_type(#struct_type "_" #member_name);          \
    cc.mov(member, member_ptr);                                                \
    return member;                                                             \
  }
#include "jit_types_info.inc"
#undef TYPES_INFO

static inline void EmitLoadFromArray(x86::Compiler &cc, x86::Gp &Array,
                                     size_t Index, x86::Gp &Elem,
                                     size_t ElemSize) {
  x86::Mem ElemPtr = x86::ptr(Array, Index * ElemSize, sizeof(ElemSize));
  cc.mov(Elem, ElemPtr);
}

static inline void EmitStoreToArray(x86::Compiler &cc, x86::Gp &Array,
                                    size_t Index, x86::Gp &Elem,
                                    size_t ElemSize) {
  x86::Mem ElemPtr = x86::ptr(Array, Index * ElemSize, sizeof(ElemSize));
  cc.mov(ElemPtr, Elem);
}

bool AsmJitCompileExpr(ExprState *State) {
  PlanState *Parent = State->parent;
  AsmJitContext *Context = nullptr;
  instr_time CodeGenStartTime, CodeGenEndTime, DeformStartTime, DeformEndTime;

  /*
   * Right now we don't support compiling expressions without a parent, as
   * we need access to the EState.
   */
  Assert(Parent);

  /* get or create JIT context */
  if (Parent->state->es_jit) {
    Context = (AsmJitContext *)Parent->state->es_jit;
  } else {
    Context = JitCreateContext(Parent->state->es_jit_flags);
    Parent->state->es_jit = &Context->base;
  }

  INSTR_TIME_SET_CURRENT(CodeGenStartTime);

  jit::CodeHolder Code;
  Code.init(Runtime.environment(), Runtime.cpuFeatures());
  x86::Compiler Jitcc(&Code);

  /*
   * Datum ExprStateEvalFunc(struct ExprState *expression,
   *                         struct ExprContext *econtext,
   *                         bool *isNull);
   */
  jit::FuncNode *JittedFunc = Jitcc.addFunc(
      jit::FuncSignature::build<Datum, ExprState *, ExprContext *, bool *>());

  x86::Gp Expression = Jitcc.newUIntPtr("expression"),
          EContext = Jitcc.newUIntPtr("econtext"),
          IsNull = Jitcc.newUIntPtr("isnull");

  JittedFunc->setArg(0, Expression);
  JittedFunc->setArg(1, EContext);
  JittedFunc->setArg(2, IsNull);

  jit::Label *L_Opblocks =
      (jit::Label *)palloc(State->steps_len * sizeof(jit::Label));
  for (int OpIndex = 0; OpIndex < State->steps_len; ++OpIndex)
    L_Opblocks[OpIndex] = Jitcc.newLabel();

  for (int OpIndex = 0; OpIndex < State->steps_len; ++OpIndex) {
    ExprEvalStep *Op = &State->steps[OpIndex];
    ExprEvalOp Opcode = ExecEvalStepOp(State, Op);

    Jitcc.bind(L_Opblocks[OpIndex]);

#define BuildEvalXFunc2(Func)                                                  \
  do {                                                                         \
    jit::InvokeNode *JitFunc;                                                  \
    Jitcc.invoke(                                                              \
        &JitFunc, jit::imm(Func),                                              \
        jit::FuncSignature::build<void, ExprState *, ExprEvalStep *>());       \
    JitFunc->setArg(0, Expression);                                            \
    JitFunc->setArg(1, jit::imm(Op));                                          \
  } while (0);

#define BuildEvalXFunc3(Func)                                                  \
  do {                                                                         \
    jit::InvokeNode *JitFunc;                                                  \
    Jitcc.invoke(&JitFunc, jit::imm(Func),                                     \
                 jit::FuncSignature::build<void, ExprState *, ExprEvalStep *,  \
                                           ExprContext *>());                  \
    JitFunc->setArg(0, Expression);                                            \
    JitFunc->setArg(1, jit::imm(Op));                                          \
    JitFunc->setArg(2, EContext);                                              \
  } while (0);

    switch (Opcode) {
    case EEOP_DONE: {
      /* Load expression->resvalue and expression->resnull */
      x86::Gp Resvalue = emit_load_resvalue_from_ExprState(Jitcc, Expression),
              Resnull = emit_load_resnull_from_ExprState(Jitcc, Expression);

      /* *isnull = expression->resnull */
      x86::Mem IsNullPtr = x86::ptr(IsNull, 0, sizeof(bool));
      Jitcc.mov(IsNullPtr, Resnull);

      /* return expression->resvalue */
      Jitcc.ret(Resvalue);
      Jitcc.endFunc();
      break;
    }
    case EEOP_INNER_FETCHSOME:
    case EEOP_OUTER_FETCHSOME:
    case EEOP_SCAN_FETCHSOME: {
      const TupleTableSlotOps *TtsOps =
          Op->d.fetch.fixed ? Op->d.fetch.kind : nullptr;
      TupleDesc Desc = Op->d.fetch.known_desc;
      TupleDeformingFunc CompiledTupleDeformingFunc = nullptr;

      /* Step should not have been generated. */
      Assert(TtsOps != &TTSOpsVirtual);

      /* Compute the address of Slot->tts_nvalid */
      x86::Gp Slot =
          Opcode == EEOP_INNER_FETCHSOME
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, EContext)
              : (Opcode == EEOP_OUTER_FETCHSOME
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  EContext)
                     : emit_load_ecxt_scantuple_from_ExprContext(Jitcc,
                                                                 EContext));

      x86::Gp TtsNvalid = emit_load_tts_nvalid_from_TupleTableSlot(Jitcc, Slot);

      /*
       * Check if all required attributes are available, or whether deforming is
       * required.
       */
      Jitcc.cmp(TtsNvalid, jit::imm(Op->d.fetch.last_var));
      Jitcc.jge(L_Opblocks[OpIndex + 1]);

      if (TtsOps && Desc && (Context->base.flags & PGJIT_DEFORM)) {
        INSTR_TIME_SET_CURRENT(DeformStartTime);

        CompiledTupleDeformingFunc = CompileTupleDeformingFunc(
            Context, Runtime, Desc, TtsOps, Op->d.fetch.last_var);

        INSTR_TIME_SET_CURRENT(DeformEndTime);
        INSTR_TIME_ACCUM_DIFF(Context->base.instr.deform_counter, DeformEndTime,
                              DeformStartTime);
      }

      jit::InvokeNode *SlotGetSomeAttrsInt = nullptr;
      if (CompiledTupleDeformingFunc) {
        Jitcc.invoke(&SlotGetSomeAttrsInt, jit::imm(CompiledTupleDeformingFunc),
                     jit::FuncSignature::build<void, TupleTableSlot *>());
        SlotGetSomeAttrsInt->setArg(0, Slot);
      } else {
        Jitcc.invoke(&SlotGetSomeAttrsInt, jit::imm(slot_getsomeattrs_int),
                     jit::FuncSignature::build<void, TupleTableSlot *, int>());
        SlotGetSomeAttrsInt->setArg(0, Slot);
        SlotGetSomeAttrsInt->setArg(1, jit::imm(Op->d.fetch.last_var));
      }

      break;
    }

    case EEOP_INNER_VAR:
    case EEOP_OUTER_VAR:
    case EEOP_SCAN_VAR: {
      x86::Gp Slot =
          Opcode == EEOP_INNER_VAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, EContext)
              : (Opcode == EEOP_OUTER_VAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  EContext)
                     : emit_load_ecxt_scantuple_from_ExprContext(Jitcc,
                                                                 EContext));

      x86::Gp SlotValues =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, Slot),
              SlotIsNulls =
                  emit_load_tts_isnull_from_TupleTableSlot(Jitcc, Slot);

      int Attrnum = Op->d.var.attnum;

      x86::Gp SlotValue = Jitcc.newUIntPtr(), SlotIsNull = Jitcc.newInt8();
      EmitLoadFromArray(Jitcc, SlotValues, Attrnum, SlotValue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, SlotIsNulls, Attrnum, SlotIsNull, sizeof(bool));

      x86::Gp OpResvalue = Jitcc.newUIntPtr("op.resvalue"),
              OpResnull = Jitcc.newUIntPtr("op.resnull");
      Jitcc.mov(OpResvalue, jit::imm(Op->resvalue));
      Jitcc.mov(OpResnull, jit::imm(Op->resnull));
      x86::Mem OpResValuePtr = x86::ptr(OpResvalue, 0, sizeof(Datum)),
               OpResNullPtr = x86::ptr(OpResnull, 0, sizeof(bool));

      Jitcc.mov(OpResValuePtr, SlotValue);
      Jitcc.mov(OpResNullPtr, SlotIsNull);

      break;
    }
    case EEOP_INNER_SYSVAR:
    case EEOP_OUTER_SYSVAR:
    case EEOP_SCAN_SYSVAR: {
      x86::Gp Slot =
          Opcode == EEOP_INNER_VAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, EContext)
              : (Opcode == EEOP_OUTER_SYSVAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  EContext)
                     : emit_load_ecxt_scantuple_from_ExprContext(Jitcc,
                                                                 EContext));

      jit::InvokeNode *ExecEvalSysVarFunc;
      Jitcc.invoke(
          &ExecEvalSysVarFunc, jit::imm(ExecEvalSysVar),
          jit::FuncSignature::build<void, ExprState *, ExprEvalStep *,
                                    ExprContext *, TupleTableSlot *>());
      ExecEvalSysVarFunc->setArg(0, Expression);
      ExecEvalSysVarFunc->setArg(1, jit::imm(Op));
      ExecEvalSysVarFunc->setArg(2, EContext);
      ExecEvalSysVarFunc->setArg(3, Slot);
      break;
    }

    case EEOP_WHOLEROW: {
      BuildEvalXFunc3(ExecEvalWholeRowVar);
      break;
    }

    case EEOP_ASSIGN_INNER_VAR:
    case EEOP_ASSIGN_OUTER_VAR:
    case EEOP_ASSIGN_SCAN_VAR: {
      x86::Gp Slot =
          Opcode == EEOP_ASSIGN_INNER_VAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, EContext)
              : (Opcode == EEOP_ASSIGN_OUTER_VAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  EContext)
                     : emit_load_ecxt_scantuple_from_ExprContext(Jitcc,
                                                                 EContext));

      x86::Gp SlotValues =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, Slot),
              SlotIsNulls =
                  emit_load_tts_isnull_from_TupleTableSlot(Jitcc, Slot);

      int Attrnum = Op->d.assign_var.attnum;

      /* Load data. */
      x86::Gp SlotValue = Jitcc.newUIntPtr("slotvalue"),
              SlotIsNull = Jitcc.newInt8("slotisnull");
      EmitLoadFromArray(Jitcc, SlotValues, Attrnum, SlotValue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, SlotIsNulls, Attrnum, SlotIsNull, sizeof(bool));

      /* Save the result. */
      int Resultnum = Op->d.assign_var.resultnum;
      x86::Gp ResultSlot =
          emit_load_resultslot_from_ExprState(Jitcc, Expression);
      x86::Gp Resvalues =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, ResultSlot),
              Resisnull =
                  emit_load_tts_isnull_from_TupleTableSlot(Jitcc, ResultSlot);

      EmitStoreToArray(Jitcc, Resvalues, Resultnum, SlotValue, sizeof(Datum));
      EmitStoreToArray(Jitcc, Resisnull, Resultnum, SlotIsNull, sizeof(bool));

      break;
    }

    case EEOP_ASSIGN_TMP:
    case EEOP_ASSIGN_TMP_MAKE_RO: {
      size_t ResultNum = Op->d.assign_tmp.resultnum;

      /* Load expression->resvalue and expression->resnull */
      x86::Gp Resvalue = emit_load_resvalue_from_ExprState(Jitcc, Expression),
              Resnull = emit_load_resnull_from_ExprState(Jitcc, Expression);

      /*
       * Compute the addresses of
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      x86::Gp ResultSlot =
          emit_load_resultslot_from_ExprState(Jitcc, Expression);
      x86::Gp Resvalues =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, ResultSlot),
              Resisnull =
                  emit_load_tts_isnull_from_TupleTableSlot(Jitcc, ResultSlot);
      /*
       * Store nullness.
       */
      EmitStoreToArray(Jitcc, Resisnull, ResultNum, Resnull, sizeof(bool));

      if (Opcode == EEOP_ASSIGN_TMP_MAKE_RO) {
        Jitcc.cmp(Resnull, jit::imm(1));
        Jitcc.je(L_Opblocks[OpIndex + 1]);

        jit::InvokeNode *MakeExpandedObjectReadOnlyInternalFunc;
        Jitcc.invoke(&MakeExpandedObjectReadOnlyInternalFunc,
                     jit::imm(MakeExpandedObjectReadOnlyInternal),
                     jit::FuncSignature::build<Datum, Datum>());
        MakeExpandedObjectReadOnlyInternalFunc->setArg(0, Resvalue);
        MakeExpandedObjectReadOnlyInternalFunc->setRet(0, Resvalue);
      }

      /* Finally, store the result. */
      EmitStoreToArray(Jitcc, Resvalues, ResultNum, Resvalue, sizeof(Datum));
      break;
    }
    case EEOP_CONST: {
      x86::Gp ConstVal = Jitcc.newUIntPtr("constval.value"),
              ConstNull = Jitcc.newInt8("constval.isnull");

      Jitcc.mov(ConstVal, jit::imm(Op->d.constval.value));
      Jitcc.mov(ConstNull, jit::imm(Op->d.constval.isnull));

      /*
       * Store Op->d.constval.value to Op->resvalue.
       * Store Op->d.constval.isnull to Op->resnull.
       */
      x86::Gp OpResValue = Jitcc.newUIntPtr("op.resvalue"),
              OpResNull = Jitcc.newUIntPtr("op.resnull");
      Jitcc.mov(OpResValue, jit::imm(Op->resvalue));
      Jitcc.mov(OpResNull, jit::imm(Op->resnull));
      x86::Mem OpResValuePtr = x86::ptr(OpResValue, 0, sizeof(Datum)),
               OpResNullPtr = x86::ptr(OpResNull, 0, sizeof(bool));
      Jitcc.mov(OpResValuePtr, ConstVal);
      Jitcc.mov(OpResNullPtr, ConstNull);

      break;
    }
    case EEOP_FUNCEXPR:
    case EEOP_FUNCEXPR_STRICT: {
      FunctionCallInfo FuncCallInfo = Op->d.func.fcinfo_data;

      x86::Gp FuncCallInfoAddr = Jitcc.newUIntPtr();
      Jitcc.mov(FuncCallInfoAddr, FuncCallInfo);

      jit::Label L_InvokePGFunc = Jitcc.newLabel();

      if (Opcode == EEOP_FUNCEXPR_STRICT) {
        jit::Label L_StrictFail = Jitcc.newLabel();
        /* Should make sure that they're optimized beforehand. */
        int ArgsNum = Op->d.func.nargs;
        if (ArgsNum == 0) {
          ereport(ERROR,
                  (errmsg("Argumentless strict functions are pointless")));
        }

        /* Check for NULL args for strict function. */
        for (int ArgIndex = 0; ArgIndex < ArgsNum; ++ArgIndex) {
          x86::Mem FuncCallInfoArgNIsNullPtr =
              x86::ptr(FuncCallInfoAddr,
                       offsetof(FunctionCallInfoBaseData, args) +
                           ArgIndex * sizeof(NullableDatum) +
                           offsetof(NullableDatum, isnull),
                       sizeof(bool));
          x86::Gp FuncCallInfoArgNIsNull = Jitcc.newInt8();
          Jitcc.mov(FuncCallInfoArgNIsNull, FuncCallInfoArgNIsNullPtr);
          Jitcc.cmp(FuncCallInfoArgNIsNull, jit::imm(1));
          Jitcc.je(L_StrictFail);
        }

        Jitcc.jmp(L_InvokePGFunc);

        Jitcc.bind(L_StrictFail);
        /* Op->resnull = true */
        x86::Gp OpResNull = Jitcc.newUIntPtr("op.resnull");
        Jitcc.mov(OpResNull, jit::imm(Op->resnull));
        x86::Mem OpResNullPtr = x86::ptr(OpResNull, 0, sizeof(bool));
        Jitcc.mov(OpResNullPtr, jit::imm(1));
        Jitcc.jmp(L_Opblocks[OpIndex + 1]);
      }

      /*
       * Before invoking PGFuncs, we should set FuncCallInfo->isnull to false.
       */
      Jitcc.bind(L_InvokePGFunc);
      x86::Mem FuncCallInfoIsNullPtr =
          x86::ptr(FuncCallInfoAddr, offsetof(FunctionCallInfoBaseData, isnull),
                   sizeof(bool));
      Jitcc.mov(FuncCallInfoIsNullPtr, jit::imm(0));

      jit::InvokeNode *PGFunc;
      x86::Gp RetValue = Jitcc.newUIntPtr();
      Jitcc.invoke(&PGFunc, jit::imm(FuncCallInfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->setArg(0, FuncCallInfo);
      PGFunc->setRet(0, RetValue);

      /* Write result values. */
      x86::Gp OpResValue = Jitcc.newUIntPtr("op.resvalue"),
              OpResNull = Jitcc.newUIntPtr("op.resnull");
      Jitcc.mov(OpResValue, jit::imm(Op->resvalue));
      Jitcc.mov(OpResNull, jit::imm(Op->resnull));
      x86::Mem OpResValuePtr = x86::ptr(OpResValue, 0, sizeof(Datum)),
               OpResNullPtr = x86::ptr(OpResNull, 0, sizeof(bool));
      Jitcc.mov(OpResValuePtr, RetValue);
      x86::Gp FuncCallInfoIsNull = Jitcc.newInt8();
      Jitcc.mov(FuncCallInfoIsNull, FuncCallInfoIsNullPtr);
      Jitcc.mov(OpResNullPtr, FuncCallInfoIsNull);

      break;
    }

    case EEOP_FUNCEXPR_FUSAGE: {
      BuildEvalXFunc3(ExecEvalFuncExprFusage);
      break;
    }

    case EEOP_FUNCEXPR_STRICT_FUSAGE: {
      BuildEvalXFunc3(ExecEvalFuncExprStrictFusage);
      break;
    }

    case EEOP_QUAL: {
      jit::Label L_HandleNullOrFalse = Jitcc.newLabel();

      x86::Gp Resvalue = Jitcc.newUIntPtr(), Resnull = Jitcc.newInt8();
      x86::Gp OpResValue = Jitcc.newUIntPtr("op.resvalue"),
              OpResNull = Jitcc.newUIntPtr("op.resnull");
      Jitcc.mov(OpResValue, jit::imm(Op->resvalue));
      Jitcc.mov(OpResNull, jit::imm(Op->resnull));
      x86::Mem OpResValuePtr = x86::ptr(OpResValue, 0, sizeof(Datum)),
               OpResNullPtr = x86::ptr(OpResNull, 0, sizeof(bool));
      Jitcc.mov(Resvalue, OpResValuePtr);
      Jitcc.mov(Resnull, OpResNullPtr);

      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.je(L_HandleNullOrFalse);

      Jitcc.cmp(Resvalue, jit::imm(0));
      Jitcc.je(L_HandleNullOrFalse);

      Jitcc.jmp(L_Opblocks[OpIndex + 1]);

      /* Handling null or false. */
      Jitcc.bind(L_HandleNullOrFalse);

      /* Set resnull and resvalue to false. */
      Jitcc.mov(OpResValuePtr, jit::imm(0));
      Jitcc.mov(OpResNullPtr, jit::imm(0));

      Jitcc.jmp(L_Opblocks[Op->d.qualexpr.jumpdone]);

      break;
    }

    case EEOP_SQLVALUEFUNCTION: {
      BuildEvalXFunc2(ExecEvalSQLValueFunction);
      break;
    }

    case EEOP_CURRENTOFEXPR: {
      BuildEvalXFunc2(ExecEvalCurrentOfExpr);
      break;
    }

    case EEOP_NEXTVALUEEXPR: {
      BuildEvalXFunc2(ExecEvalNextValueExpr);
      break;
    }

    case EEOP_ARRAYEXPR: {
      BuildEvalXFunc2(ExecEvalArrayExpr);
      break;
    }

    case EEOP_ARRAYCOERCE: {
      BuildEvalXFunc3(ExecEvalArrayCoerce);
      break;
    }

    case EEOP_ROW: {
      BuildEvalXFunc2(ExecEvalRow);
      break;
    }

    case EEOP_MINMAX: {
      BuildEvalXFunc2(ExecEvalMinMax);
      break;
    }

    case EEOP_FIELDSELECT: {
      BuildEvalXFunc3(ExecEvalFieldSelect);
      break;
    }

    case EEOP_FIELDSTORE_DEFORM: {
      BuildEvalXFunc3(ExecEvalFieldStoreDeForm);
      break;
    }

    case EEOP_FIELDSTORE_FORM: {
      BuildEvalXFunc3(ExecEvalFieldStoreForm);
      break;
    }

    case EEOP_DOMAIN_NOTNULL: {
      BuildEvalXFunc2(ExecEvalConstraintNotNull);
      break;
    }

    case EEOP_DOMAIN_CHECK: {
      BuildEvalXFunc2(ExecEvalConstraintCheck);
      break;
    }

    case EEOP_CONVERT_ROWTYPE: {
      BuildEvalXFunc3(ExecEvalConvertRowtype);
      break;
    }

    case EEOP_SCALARARRAYOP: {
      BuildEvalXFunc2(ExecEvalScalarArrayOp);
      break;
    }

    case EEOP_HASHED_SCALARARRAYOP: {
      BuildEvalXFunc3(ExecEvalHashedScalarArrayOp);
      break;
    }

    case EEOP_XMLEXPR: {
      BuildEvalXFunc2(ExecEvalXmlExpr);
      break;
    }

    case EEOP_GROUPING_FUNC: {
      BuildEvalXFunc2(ExecEvalGroupingFunc);
      break;
    }

    case EEOP_SUBPLAN: {
      BuildEvalXFunc3(ExecEvalSubPlan);
      break;
    }

    case EEOP_AGG_ORDERED_TRANS_DATUM: {
      BuildEvalXFunc3(ExecEvalAggOrderedTransDatum);
      break;
    }

    case EEOP_AGG_ORDERED_TRANS_TUPLE: {
      BuildEvalXFunc3(ExecEvalAggOrderedTransTuple);
      break;
    }

    case EEOP_LAST: {
      Assert(false);
      break;
    }

    default:
      ereport(LOG, (errmsg("Jit for operator (%d) is not supported", Opcode)));
      return false;
    }
  }

  Jitcc.finalize();

  ExprStateEvalFunc EvalFunc =
      (ExprStateEvalFunc)EmitJittedFunction(Context, Code);
  if (!EvalFunc)
    return false;

  {
    State->evalfunc = ExecCompiledExpr;
    State->evalfunc_private = (void *)EvalFunc;
  }

  INSTR_TIME_SET_CURRENT(CodeGenEndTime);
  INSTR_TIME_ACCUM_DIFF(Context->base.instr.generation_counter, CodeGenEndTime,
                        CodeGenStartTime);

  return true;
}
}

void *EmitJittedFunction(AsmJitContext *Context, jit::CodeHolder &Code) {
  instr_time CodeEmissionStartTime, CodeEmissionEndTime;
  void *EmittedFunc;
  INSTR_TIME_SET_CURRENT(CodeEmissionStartTime);
  jit::Error err = Runtime.add(&EmittedFunc, &Code);
  if (err) {
    ereport(LOG,
            (errmsg("Jit failed: %s", jit::DebugUtils::errorAsString(err))));
    return nullptr;
  }
  INSTR_TIME_SET_CURRENT(CodeEmissionEndTime);
  INSTR_TIME_ACCUM_DIFF(Context->base.instr.emission_counter,
                        CodeEmissionEndTime, CodeEmissionStartTime);

  {
    MemoryContext OldContext = MemoryContextSwitchTo(TopMemoryContext);
    Context->funcs = lappend(Context->funcs, EmittedFunc);
    Context->base.instr.created_functions++;
    MemoryContextSwitchTo(OldContext);
  }

  return EmittedFunc;
}
