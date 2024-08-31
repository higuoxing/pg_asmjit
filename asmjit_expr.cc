#include "asmjit/x86/x86compiler.h"
#include "asmjit_common.h"
#include "executor/execExpr.h"

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

void AsmJitResetAfterError(void) { /* TODO */ }

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

  x86::Gp Expression = Jitcc.newUIntPtr("expression.uintptr"),
          EContext = Jitcc.newUIntPtr("econtext.uintptr"),
          IsNull = Jitcc.newUIntPtr("isnull.uintptr");

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

#define todo()                                                                 \
  do {                                                                         \
    elog(LOG, "TODO: Opcode (%d)", Opcode);                                    \
    return false;                                                              \
  } while (0);

    switch (Opcode) {
    case EEOP_DONE: {
      /* Load expression->resvalue and expression->resnull */
      x86::Gp Resvalue = emit_load_resvalue_from_ExprState(Jitcc, Expression),
              Resnull = emit_load_resnull_from_ExprState(Jitcc, Expression);

      /* *isnull = expression->resnull */
      EmitStoreToArray(Jitcc, IsNull, 0, Resnull, sizeof(bool));

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
        /* Invoke the JIT-ed deforming function. */
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

      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);

      EmitStoreToArray(Jitcc, OpResvalue, 0, SlotValue, sizeof(Datum));
      EmitStoreToArray(Jitcc, OpResnull, 0, SlotIsNull, sizeof(bool));

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
      x86::Gp SlotValue = Jitcc.newUIntPtr("slotvalue.uintptr"),
              SlotIsNull = Jitcc.newInt8("slotisnull.i8");
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
      x86::Gp ConstVal = EmitLoadConstUInt64(Jitcc, "constval.value.u64",
                                             Op->d.constval.value),
              ConstNull = EmitLoadConstUInt8(Jitcc, "constval.isnull.u8",
                                             Op->d.constval.isnull);

      /*
       * Store Op->d.constval.value to Op->resvalue.
       * Store Op->d.constval.isnull to Op->resnull.
       */
      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      EmitStoreToArray(Jitcc, OpResvalue, 0, ConstVal, sizeof(Datum));
      EmitStoreToArray(Jitcc, OpResnull, 0, ConstNull, sizeof(bool));

      break;
    }
    case EEOP_FUNCEXPR:
    case EEOP_FUNCEXPR_STRICT: {
      FunctionCallInfo FuncCallInfo = Op->d.func.fcinfo_data;
      x86::Gp FuncCallInfoAddr =
          EmitLoadConstUIntPtr(Jitcc, "func.fcinfo_data", FuncCallInfo);

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
        x86::Gp OpResNull =
            EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr", Op->resnull);
        EmitStoreToArray(Jitcc, OpResNull, 0, jit::imm(1), sizeof(bool));
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
      x86::Gp OpResValue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResNull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);

      EmitStoreToArray(Jitcc, OpResValue, 0, RetValue, sizeof(Datum));
      x86::Gp FuncCallInfoIsNull = Jitcc.newInt8();
      Jitcc.mov(FuncCallInfoIsNull, FuncCallInfoIsNullPtr);
      EmitStoreToArray(Jitcc, OpResNull, 0, FuncCallInfoIsNull, sizeof(bool));

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
      /*
       * Treat them the same for now, optimizer can remove
       * redundancy. Could be worthwhile to optimize during emission
       * though.
       */
    case EEOP_BOOL_AND_STEP_FIRST:
    case EEOP_BOOL_AND_STEP:
    case EEOP_BOOL_AND_STEP_LAST: {
      x86::Gp Anynull = EmitLoadConstUIntPtr(Jitcc, "op.d.boolexpr.anynull",
                                             Op->d.boolexpr.anynull);
      jit::Label L_BoolCheckFalse = Jitcc.newLabel(),
                 L_BoolCont = Jitcc.newLabel();

      if (Opcode == EEOP_BOOL_AND_STEP_FIRST)
        EmitStoreToArray(Jitcc, Anynull, 0, jit::imm(0), sizeof(bool));

      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      x86::Gp Boolvalue = Jitcc.newUIntPtr("resvalue.uintptr"),
              Boolnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResvalue, 0, Boolvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, OpResnull, 0, Boolnull, sizeof(bool));

      /* check if current input is NULL */
      Jitcc.cmp(Boolnull, jit::imm(1));
      Jitcc.jne(L_BoolCheckFalse);
      {
        /* b_boolisnull */
        /* set boolanynull to true */
        EmitStoreToArray(Jitcc, Anynull, 0, jit::imm(1), sizeof(bool));
        Jitcc.jmp(L_BoolCont);
      }

      Jitcc.bind(L_BoolCheckFalse);
      {
        Jitcc.cmp(Boolvalue, jit::imm(0));
        Jitcc.jne(L_BoolCont);

        /* b_boolisfalse */
        /* result is already set to FALSE, need not change it */
        /* and jump to the end of the AND expression */
        Jitcc.jmp(L_Opblocks[Op->d.boolexpr.jumpdone]);
      }

      Jitcc.bind(L_BoolCont);
      {
        x86::Gp BoolAnyNull = Jitcc.newInt8("boolanynull.i8");
        EmitLoadFromArray(Jitcc, Anynull, 0, BoolAnyNull, sizeof(bool));
        Jitcc.cmp(BoolAnyNull, jit::imm(0));
        Jitcc.je(L_Opblocks[OpIndex + 1]);
      }

      /* set resnull to true */
      EmitStoreToArray(Jitcc, OpResnull, 0, jit::imm(1), sizeof(bool));
      /* reset resvalue */
      EmitStoreToArray(Jitcc, OpResvalue, 0, jit::imm(0), sizeof(Datum));

      break;
    }
      /*
       * Treat them the same for now, optimizer can remove
       * redundancy. Could be worthwhile to optimize during emission
       * though.
       */
    case EEOP_BOOL_OR_STEP_FIRST:
    case EEOP_BOOL_OR_STEP:
    case EEOP_BOOL_OR_STEP_LAST: {
      x86::Gp Anynull = EmitLoadConstUIntPtr(Jitcc, "op.d.boolexpr.anynull",
                                             Op->d.boolexpr.anynull);
      jit::Label L_BoolCheckTrue = Jitcc.newLabel(),
                 L_BoolCont = Jitcc.newLabel();

      if (Opcode == EEOP_BOOL_OR_STEP_FIRST)
        EmitStoreToArray(Jitcc, Anynull, 0, jit::imm(0), sizeof(bool));

      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      x86::Gp Boolvalue = Jitcc.newUIntPtr("resvalue.uintptr"),
              Boolnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResvalue, 0, Boolvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, OpResnull, 0, Boolnull, sizeof(bool));

      /* check if current input is NULL */
      Jitcc.cmp(Boolnull, jit::imm(1));
      Jitcc.jne(L_BoolCheckTrue);
      {
        /* b_boolisnull */
        /* set boolanynull to true */
        EmitStoreToArray(Jitcc, Anynull, 0, jit::imm(1), sizeof(bool));
        Jitcc.jmp(L_BoolCont);
      }

      Jitcc.bind(L_BoolCheckTrue);
      {
        Jitcc.cmp(Boolvalue, jit::imm(1));
        Jitcc.jne(L_BoolCont);

        /* b_boolistrue */
        /* result is already set to FALSE, need not change it */
        /* and jump to the end of the AND expression */
        Jitcc.jmp(L_Opblocks[Op->d.boolexpr.jumpdone]);
      }

      Jitcc.bind(L_BoolCont);
      {
        x86::Gp BoolAnyNull = Jitcc.newInt8("boolanynull.i8");
        EmitLoadFromArray(Jitcc, Anynull, 0, BoolAnyNull, sizeof(bool));
        Jitcc.cmp(BoolAnyNull, jit::imm(0));
        Jitcc.je(L_Opblocks[OpIndex + 1]);
      }

      /* set resnull to true */
      EmitStoreToArray(Jitcc, OpResnull, 0, jit::imm(1), sizeof(bool));
      /* reset resvalue */
      EmitStoreToArray(Jitcc, OpResvalue, 0, jit::imm(0), sizeof(Datum));
      break;
    }

    case EEOP_BOOL_NOT_STEP: {
      x86::Gp OpResvalue =
          EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr", Op->resvalue);
      x86::Gp Boolvalue = Jitcc.newUIntPtr("resvalue.uintptr"),
              NegBool = Jitcc.newUIntPtr("negbool.uintptr");

      EmitLoadFromArray(Jitcc, OpResvalue, 0, Boolvalue, sizeof(Datum));
      Jitcc.xor_(NegBool, NegBool);
      Jitcc.cmp(Boolvalue, jit::imm(0));
      Jitcc.sete(NegBool);

      EmitStoreToArray(Jitcc, OpResvalue, 0, NegBool, sizeof(Datum));
      break;
    }

    case EEOP_QUAL: {
      jit::Label L_HandleNullOrFalse = Jitcc.newLabel();

      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      x86::Gp Resvalue = Jitcc.newUIntPtr("resvalue.uintptr"),
              Resnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResvalue, 0, Resvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, OpResnull, 0, Resnull, sizeof(bool));

      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.je(L_HandleNullOrFalse);

      Jitcc.cmp(Resvalue, jit::imm(0));
      Jitcc.je(L_HandleNullOrFalse);

      Jitcc.jmp(L_Opblocks[OpIndex + 1]);

      /* Handling null or false. */
      Jitcc.bind(L_HandleNullOrFalse);

      /* Set resnull and resvalue to false. */
      EmitStoreToArray(Jitcc, OpResvalue, 0, jit::imm(0), sizeof(Datum));
      EmitStoreToArray(Jitcc, OpResnull, 0, jit::imm(0), sizeof(bool));

      Jitcc.jmp(L_Opblocks[Op->d.qualexpr.jumpdone]);

      break;
    }

    case EEOP_JUMP: {
      Jitcc.jmp(L_Opblocks[Op->d.jump.jumpdone]);
      break;
    }

    case EEOP_JUMP_IF_NULL: {
      /* Transfer control if current result is null */
      x86::Gp OpResnull =
          EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr", Op->resnull);
      x86::Gp Resnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResnull, 0, Resnull, sizeof(bool));

      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.je(L_Opblocks[Op->d.jump.jumpdone]);

      break;
    }

    case EEOP_JUMP_IF_NOT_NULL: {
      /* Transfer control if current result is non-null */
      x86::Gp OpResnull =
          EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr", Op->resnull);
      x86::Gp Resnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResnull, 0, Resnull, sizeof(bool));

      Jitcc.cmp(Resnull, jit::imm(0));
      Jitcc.je(L_Opblocks[Op->d.jump.jumpdone]);

      break;
    }

    case EEOP_JUMP_IF_NOT_TRUE: {
      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      x86::Gp Resvalue = Jitcc.newUIntPtr("resvalue.uintptr"),
              Resnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResvalue, 0, Resvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, OpResnull, 0, Resnull, sizeof(bool));

      /* Transfer control if current result is null or false */
      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.je(L_Opblocks[Op->d.jump.jumpdone]);
      Jitcc.cmp(Resvalue, jit::imm(0));
      Jitcc.je(L_Opblocks[Op->d.jump.jumpdone]);

      break;
    }

    case EEOP_NULLTEST_ISNULL: {
      todo();
    }

    case EEOP_NULLTEST_ISNOTNULL: {
      todo();
    }

    case EEOP_NULLTEST_ROWISNULL: {
      todo();
    }

    case EEOP_NULLTEST_ROWISNOTNULL: {
      todo();
    }

    case EEOP_BOOLTEST_IS_TRUE:
    case EEOP_BOOLTEST_IS_NOT_FALSE:
    case EEOP_BOOLTEST_IS_FALSE:
    case EEOP_BOOLTEST_IS_NOT_TRUE: {
      todo();
    }

    case EEOP_PARAM_EXEC: {
      todo();
    }

    case EEOP_PARAM_EXTERN: {
      todo();
    }

    case EEOP_PARAM_CALLBACK: {
      todo();
    }

    case EEOP_PARAM_SET: {
      todo();
    }

    case EEOP_SBSREF_SUBSCRIPTS: {
      todo();
    }

    case EEOP_SBSREF_OLD:
    case EEOP_SBSREF_ASSIGN:
    case EEOP_SBSREF_FETCH: {
      todo();
    }

    case EEOP_CASE_TESTVAL: {
      todo();
    }
    case EEOP_MAKE_READONLY: {
      todo();
    }

    case EEOP_IOCOERCE: {
      todo();
    }

    case EEOP_IOCOERCE_SAFE: {
      todo();
    }

    case EEOP_DISTINCT:
    case EEOP_NOT_DISTINCT: {
      todo();
    }

    case EEOP_NULLIF: {
      todo();
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

    case EEOP_ROWCOMPARE_STEP: {
      todo();
    }

    case EEOP_ROWCOMPARE_FINAL: {
      todo();
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

    case EEOP_DOMAIN_TESTVAL: {
      todo();
    }

    case EEOP_DOMAIN_NOTNULL: {
      BuildEvalXFunc2(ExecEvalConstraintNotNull);
      break;
    }

    case EEOP_DOMAIN_CHECK: {
      BuildEvalXFunc2(ExecEvalConstraintCheck);
      break;
    }

    case EEOP_HASHDATUM_SET_INITVAL: {
      todo();
    }
    case EEOP_HASHDATUM_FIRST:
    case EEOP_HASHDATUM_FIRST_STRICT:
    case EEOP_HASHDATUM_NEXT32:
    case EEOP_HASHDATUM_NEXT32_STRICT: {
      todo();
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

    case EEOP_JSON_CONSTRUCTOR: {
      todo();
    }

    case EEOP_IS_JSON: {
      todo();
    }

    case EEOP_JSONEXPR_PATH: {
      todo();
    }

    case EEOP_JSONEXPR_COERCION: {
      todo();
    }
    case EEOP_JSONEXPR_COERCION_FINISH: {
      todo();
    }

    case EEOP_AGGREF: {
      todo();
    }

    case EEOP_GROUPING_FUNC: {
      BuildEvalXFunc2(ExecEvalGroupingFunc);
      break;
    }

    case EEOP_WINDOW_FUNC: {
      todo();
    }

    case EEOP_MERGE_SUPPORT_FUNC: {
      todo();
    }

    case EEOP_SUBPLAN: {
      BuildEvalXFunc3(ExecEvalSubPlan);
      break;
    }

    case EEOP_AGG_STRICT_DESERIALIZE:
    case EEOP_AGG_DESERIALIZE: {
      todo();
    }

    case EEOP_AGG_STRICT_INPUT_CHECK_ARGS:
    case EEOP_AGG_STRICT_INPUT_CHECK_NULLS: {
      todo();
    }
    case EEOP_AGG_PLAIN_PERGROUP_NULLCHECK: {
      todo();
    }

    case EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYVAL:
    case EEOP_AGG_PLAIN_TRANS_STRICT_BYVAL:
    case EEOP_AGG_PLAIN_TRANS_BYVAL:
    case EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYREF:
    case EEOP_AGG_PLAIN_TRANS_STRICT_BYREF:
    case EEOP_AGG_PLAIN_TRANS_BYREF: {
      todo();
    }
    case EEOP_AGG_PRESORTED_DISTINCT_SINGLE: {
      todo();
    }
    case EEOP_AGG_PRESORTED_DISTINCT_MULTI: {
      todo();
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

      /* Don't need a default case, since we want to know if any case is
       * missing. */
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
