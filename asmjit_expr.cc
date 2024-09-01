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
      x86::Gp v_fcinfo =
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
          x86::Gp FuncCallInfoArgNIsNull =
              LoadFuncArgNull(Jitcc, v_fcinfo, ArgIndex);
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
      emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo,
                                                    jit::imm(0));

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
      x86::Gp FuncCallInfoIsNull =
          emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);
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
      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      x86::Gp Resvalue = Jitcc.newUIntPtr("resvalue.uintptr"),
              Resnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResnull, 0, Resnull, sizeof(bool));
      Jitcc.xor_(Resvalue, Resvalue);
      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.sete(Resvalue);
      EmitStoreToArray(Jitcc, OpResvalue, 0, Resvalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, OpResnull, 0, jit::imm(0), sizeof(bool));

      break;
    }

    case EEOP_NULLTEST_ISNOTNULL: {
      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      x86::Gp Resvalue = Jitcc.newUIntPtr("resvalue.uintptr"),
              Resnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResnull, 0, Resnull, sizeof(bool));
      Jitcc.xor_(Resvalue, Resvalue);
      Jitcc.cmp(Resnull, jit::imm(0));
      Jitcc.sete(Resvalue);

      EmitStoreToArray(Jitcc, OpResvalue, 0, Resvalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, OpResnull, 0, jit::imm(0), sizeof(bool));

      break;
    }

    case EEOP_NULLTEST_ROWISNULL: {
      BuildEvalXFunc3(ExecEvalRowNull);
      break;
    }

    case EEOP_NULLTEST_ROWISNOTNULL: {
      BuildEvalXFunc3(ExecEvalRowNotNull);
      break;
    }

    case EEOP_BOOLTEST_IS_TRUE:
    case EEOP_BOOLTEST_IS_NOT_FALSE:
    case EEOP_BOOLTEST_IS_FALSE:
    case EEOP_BOOLTEST_IS_NOT_TRUE: {
      jit::Label L_NotNull = Jitcc.newLabel();
      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      x86::Gp Resnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, OpResnull, 0, Resnull, sizeof(bool));
      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.jne(L_NotNull);
      /* result is not null. */
      EmitStoreToArray(Jitcc, OpResnull, 0, jit::imm(0), sizeof(bool));
      EmitStoreToArray(
          Jitcc, OpResvalue, 0,
          (Opcode == EEOP_BOOLTEST_IS_TRUE || Opcode == EEOP_BOOLTEST_IS_FALSE)
              ? jit::imm(0)
              : jit::imm(1),
          sizeof(Datum));
      Jitcc.jmp(L_Opblocks[OpIndex + 1]);

      Jitcc.bind(L_NotNull);
      if (Opcode == EEOP_BOOLTEST_IS_TRUE ||
          Opcode == EEOP_BOOLTEST_IS_NOT_FALSE) {
        /*
         * if value is not null NULL, return value (already
         * set)
         */
      } else {
        x86::Gp Resvalue = Jitcc.newUIntPtr("resvalue.uintptr");
        x86::Gp ResvalueIsFalse = Jitcc.newUIntPtr("resvalueisfalse.uintptr");
        EmitLoadFromArray(Jitcc, OpResvalue, 0, Resvalue, sizeof(Datum));
        Jitcc.xor_(ResvalueIsFalse, ResvalueIsFalse);
        Jitcc.cmp(Resvalue, jit::imm(0));
        Jitcc.sete(ResvalueIsFalse);
        EmitStoreToArray(Jitcc, OpResvalue, 0, ResvalueIsFalse, sizeof(Datum));
      }

      break;
    }

    case EEOP_PARAM_EXEC: {
      BuildEvalXFunc3(ExecEvalParamExec);
      break;
    }

    case EEOP_PARAM_EXTERN: {
      BuildEvalXFunc3(ExecEvalParamExtern);
      break;
    }

    case EEOP_PARAM_CALLBACK: {
      BuildEvalXFunc3(Op->d.cparam.paramfunc);
      break;
    }

    case EEOP_PARAM_SET: {
      BuildEvalXFunc3(ExecEvalParamSet);
      break;
    }

    case EEOP_SBSREF_SUBSCRIPTS: {
      jit::InvokeNode *InvokeSubscriptFunc;
      x86::Gp RetVal = Jitcc.newInt8("ret.i8");
      Jitcc.invoke(
          &InvokeSubscriptFunc, jit::imm(Op->d.sbsref_subscript.subscriptfunc),
          jit::FuncSignature::build<bool, ExprState *, struct ExprEvalStep *,
                                    ExprContext *>());
      InvokeSubscriptFunc->setArg(0, Expression);
      InvokeSubscriptFunc->setArg(1, jit::imm(Op));
      InvokeSubscriptFunc->setArg(2, EContext);
      InvokeSubscriptFunc->setRet(0, RetVal);

      Jitcc.cmp(RetVal, jit::imm(0));
      Jitcc.je(L_Opblocks[Op->d.sbsref_subscript.jumpdone]);
      break;
    }

    case EEOP_SBSREF_OLD:
    case EEOP_SBSREF_ASSIGN:
    case EEOP_SBSREF_FETCH: {
      BuildEvalXFunc3(Op->d.sbsref.subscriptfunc);
      break;
    }

    case EEOP_CASE_TESTVAL: {
      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      if (Op->d.casetest.value) {
        x86::Gp CaseValuep =
                    EmitLoadConstUIntPtr(Jitcc, "op.d.casetest.valuep.uintptr",
                                         Op->d.casetest.value),
                CaseNullp =
                    EmitLoadConstUIntPtr(Jitcc, "op.d.casetest.isnullp.uintptr",
                                         Op->d.casetest.isnull);
        x86::Gp CaseValue = Jitcc.newUIntPtr("op.d.casetest.value.uintptr"),
                CaseNull = Jitcc.newInt8("op.d.casetest.isnull.i8");
        EmitLoadFromArray(Jitcc, CaseValuep, 0, CaseValue, sizeof(Datum));
        EmitLoadFromArray(Jitcc, CaseNullp, 0, CaseNull, sizeof(bool));

        EmitStoreToArray(Jitcc, OpResvalue, 0, CaseValue, sizeof(Datum));
        EmitStoreToArray(Jitcc, OpResnull, 0, CaseNull, sizeof(bool));
      } else {
        x86::Gp CaseValue =
            emit_load_caseValue_datum_from_ExprContext(Jitcc, EContext);
        x86::Gp CaseNull =
            emit_load_caseValue_isNull_from_ExprContext(Jitcc, EContext);

        EmitStoreToArray(Jitcc, OpResvalue, 0, CaseValue, sizeof(Datum));
        EmitStoreToArray(Jitcc, OpResnull, 0, CaseNull, sizeof(bool));
      }
      break;
    }
    case EEOP_MAKE_READONLY: {
      x86::Gp Nullp =
          EmitLoadConstUIntPtr(Jitcc, "op.d.make_readonly.isnullp.uintptr",
                               Op->d.make_readonly.isnull);
      x86::Gp Null = Jitcc.newInt8("op.d.make_readonly.isnull.i8");

      EmitLoadFromArray(Jitcc, Nullp, 0, Null, sizeof(bool));

      /* store null isnull value in result */
      x86::Gp OpResnull =
          EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr", Op->resnull);
      EmitStoreToArray(Jitcc, OpResnull, 0, Null, sizeof(bool));

      Jitcc.cmp(Null, jit::imm(1));
      Jitcc.je(L_Opblocks[OpIndex + 1]);

      /* if value is not null, convert to RO datum */
      x86::Gp Valuep =
          EmitLoadConstUIntPtr(Jitcc, "op.d.make_readonly.valuep.uintptr",
                               Op->d.make_readonly.value);
      x86::Gp Value = Jitcc.newUIntPtr("op.d.make_readonly.value.uintptr");
      EmitLoadFromArray(Jitcc, Valuep, 0, Value, sizeof(Datum));
      jit::InvokeNode *InvokeMakeExpandedObjectReadOnly;
      Jitcc.invoke(&InvokeMakeExpandedObjectReadOnly,
                   jit::imm(MakeExpandedObjectReadOnlyInternal),
                   jit::FuncSignature::build<Datum, Datum>());
      InvokeMakeExpandedObjectReadOnly->setArg(0, Value);
      InvokeMakeExpandedObjectReadOnly->setRet(0, Value);

      x86::Gp OpResvalue =
          EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr", Op->resvalue);
      EmitStoreToArray(Jitcc, OpResvalue, 0, Value, sizeof(Datum));
      break;
    }

    case EEOP_IOCOERCE: {
      FunctionCallInfo fcinfo_out = Op->d.iocoerce.fcinfo_data_out,
                       fcinfo_in = Op->d.iocoerce.fcinfo_data_in;
      jit::Label L_SkipOutputCall = Jitcc.newLabel(),
                 L_InputCall = Jitcc.newLabel();

      x86::Gp v_fcinfo_out = EmitLoadConstUIntPtr(Jitcc, "v_fcinfo_out.uintptr",
                                                  fcinfo_out),
              v_fcinfo_in =
                  EmitLoadConstUIntPtr(Jitcc, "v_fcinfo_in.uintptr", fcinfo_in);
      x86::Gp v_output = Jitcc.newUInt64("v_output.u64");

      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", Op->resnull);
      x86::Gp v_resnull = Jitcc.newInt8("op.resnull.i8");
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      Jitcc.cmp(v_resnull, jit::imm(1));
      Jitcc.je(L_SkipOutputCall);
      {
        /* Not null, call output. */
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr", Op->resvalue);
        x86::Gp v_resvalue = Jitcc.newUIntPtr("v_resvalue");
        EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
        StoreFuncArgValue(Jitcc, v_fcinfo_out, 0, v_resvalue);
        StoreFuncArgNull(Jitcc, v_fcinfo_out, 0, jit::imm(0));

        jit::InvokeNode *PGFunc;
        Jitcc.invoke(&PGFunc, jit::imm(fcinfo_out->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->setArg(0, v_fcinfo_out);
        PGFunc->setRet(0, v_output);
        Jitcc.jmp(L_InputCall);
      }

      Jitcc.bind(L_SkipOutputCall);
      Jitcc.mov(v_output, jit::imm(0));

      Jitcc.bind(L_InputCall);
      {
        if (Op->d.iocoerce.finfo_in->fn_strict) {
          Jitcc.cmp(v_output, jit::imm(0));
          Jitcc.je(L_Opblocks[OpIndex + 1]);
        }
        EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
        /* Call input function. */
        StoreFuncArgValue(Jitcc, v_fcinfo_in, 0, v_output);
        StoreFuncArgNull(Jitcc, v_fcinfo_in, 0, v_resnull);
        emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo_in,
                                                      jit::imm(0));
        jit::InvokeNode *PGFunc;
        Jitcc.invoke(&PGFunc, jit::imm(fcinfo_in->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->setArg(0, v_fcinfo_in);
        PGFunc->setRet(0, v_output);

        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr", Op->resvalue);
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_output, sizeof(Datum));
      }
      break;
    }

    case EEOP_IOCOERCE_SAFE: {
      BuildEvalXFunc2(ExecEvalCoerceViaIOSafe);
      break;
    }

    case EEOP_DISTINCT:
    case EEOP_NOT_DISTINCT: {
      FunctionCallInfo fcinfo = Op->d.func.fcinfo_data;
      jit::Label L_NoArgIsNull = Jitcc.newLabel(),
                 L_AnyArgIsNull = Jitcc.newLabel();
      x86::Gp v_fcinfo =
          EmitLoadConstUIntPtr(Jitcc, "v_fcinfo.uintptr", fcinfo);
      /* load args[0|1].isnull for both arguments */
      x86::Gp v_argnull0 = LoadFuncArgNull(Jitcc, v_fcinfo, 0),
              v_argnull1 = LoadFuncArgNull(Jitcc, v_fcinfo, 1);
      x86::Gp v_anyargisnull = Jitcc.newInt8("v_anyargisnull.i8");
      Jitcc.mov(v_anyargisnull, v_argnull0);
      Jitcc.or_(v_anyargisnull, v_argnull1);

      Jitcc.cmp(v_anyargisnull, jit::imm(0));
      Jitcc.je(L_NoArgIsNull);
      {
        /* check both arguments */
        x86::Gp v_bothargisnull = Jitcc.newInt8("v_bothargisnull.i8");
        Jitcc.mov(v_bothargisnull, v_argnull0);
        Jitcc.and_(v_bothargisnull, v_argnull1);
        Jitcc.cmp(v_bothargisnull, jit::imm(1));
        Jitcc.jne(L_AnyArgIsNull);
        {
          x86::Gp v_resnullp =
              EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", Op->resnull);
          x86::Gp v_resvaluep =
              EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", Op->resvalue);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
          if (Opcode == EEOP_NOT_DISTINCT)
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(1), sizeof(Datum));
          else
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          Jitcc.jmp(L_Opblocks[OpIndex + 1]);
        }

        Jitcc.bind(L_AnyArgIsNull);
        {
          x86::Gp v_resnullp =
              EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", Op->resnull);
          x86::Gp v_resvaluep =
              EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", Op->resvalue);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
          if (Opcode == EEOP_NOT_DISTINCT)
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          else
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(1), sizeof(Datum));
          Jitcc.jmp(L_Opblocks[OpIndex + 1]);
        }
      }

      Jitcc.bind(L_NoArgIsNull);
      {
        x86::Gp v_retval = Jitcc.newUInt64("v_retval.u64");
        jit::InvokeNode *PGFunc;
        Jitcc.invoke(&PGFunc, jit::imm(fcinfo->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->setArg(0, v_fcinfo);
        PGFunc->setRet(0, v_retval);
        x86::Gp v_fcinfo_isnull =
            emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);
        if (Opcode == EEOP_DISTINCT) {
          /* Must invert the result of "=" */
          x86::Gp v_tmpretval = Jitcc.newUInt64("v_tmpretval.u64");
          Jitcc.mov(v_tmpretval, v_retval);
          Jitcc.xor_(v_retval, v_retval);
          Jitcc.cmp(v_tmpretval, jit::imm(0));
          Jitcc.sete(v_retval);
        }
        x86::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", Op->resnull);
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", Op->resvalue);

        EmitStoreToArray(Jitcc, v_resnullp, 0, v_fcinfo_isnull, sizeof(bool));
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
      }
      break;
    }

    case EEOP_NULLIF: {
      FunctionCallInfo fcinfo = Op->d.func.fcinfo_data;
      jit::Label L_NonNull = Jitcc.newLabel(), L_HasNull = Jitcc.newLabel();
      x86::Gp v_fcinfo =
          EmitLoadConstUIntPtr(Jitcc, "v_fcinfo.uintptr", fcinfo);

      /* if either argument is NULL they can't be equal */
      x86::Gp v_argnull0 = LoadFuncArgNull(Jitcc, v_fcinfo, 0);
      x86::Gp v_argnull1 = LoadFuncArgNull(Jitcc, v_fcinfo, 1);
      x86::Gp v_anyargisnull = Jitcc.newInt8("v_anyargisnull.i8");
      Jitcc.mov(v_anyargisnull, v_argnull0);
      Jitcc.or_(v_anyargisnull, v_argnull1);

      Jitcc.cmp(v_anyargisnull, jit::imm(1));
      Jitcc.jne(L_NonNull);
      Jitcc.bind(L_HasNull);
      {
        x86::Gp v_arg0 = LoadFuncArgValue(Jitcc, v_fcinfo, 0);
        x86::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", Op->resnull);
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", Op->resvalue);
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_argnull0, sizeof(bool));
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_arg0, sizeof(Datum));
        Jitcc.jmp(L_Opblocks[OpIndex + 1]);
      }

      Jitcc.bind(L_NonNull);
      {
        x86::Gp v_retval = Jitcc.newUInt64("v_retval.u64");
        jit::InvokeNode *PGFunc;
        Jitcc.invoke(&PGFunc, jit::imm(fcinfo->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->setArg(0, v_fcinfo);
        PGFunc->setRet(0, v_retval);
        x86::Gp v_fcinfo_isnull =
            emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);

        /*
         * If result not null, and arguments are equal return null
         * (same result as if there'd been NULLs, hence reuse
         * b_hasnull).
         */
        x86::Gp v_argsequal = Jitcc.newUInt64("v_argsequal.u64");
        Jitcc.xor_(v_argsequal, v_argsequal);
        Jitcc.cmp(v_fcinfo_isnull, jit::imm(0));
        Jitcc.sete(v_argsequal);
        Jitcc.and_(v_argsequal, v_retval);

        Jitcc.cmp(v_argsequal, jit::imm(1));
        Jitcc.jne(L_HasNull);

        /* build block setting result to NULL, if args are equal */
        x86::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", Op->resnull);
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", Op->resvalue);
        EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
        EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
      }

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

    case EEOP_ROWCOMPARE_STEP: {
      FunctionCallInfo fcinfo = Op->d.rowcompare_step.fcinfo_data;
      x86::Gp v_fcinfo =
          EmitLoadConstUIntPtr(Jitcc, "v_fcinfo.uintptr", fcinfo);
      jit::Label L_Null = Jitcc.newLabel();
      /*
       * If function is strict, and either arg is null, we're
       * done.
       */
      if (Op->d.rowcompare_step.finfo->fn_strict) {
        x86::Gp v_argnull0 = LoadFuncArgNull(Jitcc, v_fcinfo, 0);
        x86::Gp v_argnull1 = LoadFuncArgNull(Jitcc, v_fcinfo, 1);
        x86::Gp v_anyargisnull = Jitcc.newInt8("v_anyargisnull.i8");
        Jitcc.mov(v_anyargisnull, v_argnull0);
        Jitcc.or_(v_anyargisnull, v_argnull1);
        Jitcc.cmp(v_anyargisnull, jit::imm(1));
        Jitcc.je(L_Null);
      }

      x86::Gp v_retval = Jitcc.newUInt64("v_retval.u64");
      jit::InvokeNode *PGFunc;
      Jitcc.invoke(&PGFunc, jit::imm(fcinfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->setArg(0, v_fcinfo);
      PGFunc->setRet(0, v_retval);
      x86::Gp v_fcinfo_isnull =
          emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);

      x86::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", Op->resvalue);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
      /* if result of function is NULL, force NULL result */
      Jitcc.cmp(v_fcinfo_isnull, jit::imm(0));
      Jitcc.jne(L_Null);
      /* if results equal, compare next, otherwise done */
      Jitcc.cmp(v_retval, jit::imm(0));
      Jitcc.je(L_Opblocks[OpIndex + 1]);
      Jitcc.jmp(L_Opblocks[Op->d.rowcompare_step.jumpdone]);

      Jitcc.bind(L_Null);
      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", Op->resnull);
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
      Jitcc.jmp(L_Opblocks[Op->d.rowcompare_step.jumpnull]);
      break;
    }

    case EEOP_ROWCOMPARE_FINAL: {
      RowCompareType rctype = Op->d.rowcompare_final.rctype;

      /*
       * Btree comparators return 32 bit results, need to be
       * careful about sign (used as a 64 bit value it's
       * otherwise wrong).
       */
      x86::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", Op->resvalue);
      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", Op->resnull);
      x86::Gp v_cmpop = Jitcc.newInt32("v_cmpop.i32");
      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_cmpop, sizeof(int32));
      x86::Gp v_cmpresult = Jitcc.newUInt64("v_cmpresult.u64");
      Jitcc.xor_(v_cmpresult, v_cmpresult);
      Jitcc.cmp(v_cmpop, jit::imm(0));

      switch (rctype) {
      case ROWCOMPARE_LT:
        Jitcc.setl(v_cmpresult);
        break;
      case ROWCOMPARE_LE:
        Jitcc.setle(v_cmpresult);
        break;
      case ROWCOMPARE_GT:
        Jitcc.setg(v_cmpresult);
        break;
      case ROWCOMPARE_GE:
        Jitcc.setge(v_cmpresult);
        break;
      default:
        /* EQ and NE cases aren't allowed here */
        Assert(false);
        break;
      }

      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_cmpresult, sizeof(Datum));

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

    case EEOP_DOMAIN_TESTVAL: {
      x86::Gp OpResvalue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResnull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      if (Op->d.casetest.value) {
        x86::Gp CaseValuep =
                    EmitLoadConstUIntPtr(Jitcc, "op.d.casetest.valuep.uintptr",
                                         Op->d.casetest.value),
                CaseNullp =
                    EmitLoadConstUIntPtr(Jitcc, "op.d.casetest.isnullp.uintptr",
                                         Op->d.casetest.isnull);
        x86::Gp CaseValue = Jitcc.newUIntPtr("op.d.casetest.value.uintptr"),
                CaseNull = Jitcc.newInt8("op.d.casetest.isnull.i8");
        EmitLoadFromArray(Jitcc, CaseValuep, 0, CaseValue, sizeof(Datum));
        EmitLoadFromArray(Jitcc, CaseNullp, 0, CaseNull, sizeof(bool));

        EmitStoreToArray(Jitcc, OpResvalue, 0, CaseValue, sizeof(Datum));
        EmitStoreToArray(Jitcc, OpResnull, 0, CaseNull, sizeof(bool));
      } else {
        x86::Gp CaseValue =
            emit_load_domainValue_datum_from_ExprContext(Jitcc, EContext);
        x86::Gp CaseNull =
            emit_load_domainValue_isNull_from_ExprContext(Jitcc, EContext);

        EmitStoreToArray(Jitcc, OpResvalue, 0, CaseValue, sizeof(Datum));
        EmitStoreToArray(Jitcc, OpResnull, 0, CaseNull, sizeof(bool));
      }
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

    case EEOP_HASHDATUM_SET_INITVAL: {
      x86::Gp OpResValue = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                Op->resvalue),
              OpResNull = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                               Op->resnull);
      EmitStoreToArray(Jitcc, OpResValue, 0,
                       jit::imm(Op->d.hashdatum_initvalue.init_value),
                       sizeof(Datum));
      EmitStoreToArray(Jitcc, OpResNull, 0, jit::imm(0), sizeof(bool));

      break;
    }
    case EEOP_HASHDATUM_FIRST:
    case EEOP_HASHDATUM_FIRST_STRICT:
    case EEOP_HASHDATUM_NEXT32:
    case EEOP_HASHDATUM_NEXT32_STRICT: {
      jit::Label L_IfNull = Jitcc.newLabel();
      FunctionCallInfo fcinfo = Op->d.hashdatum.fcinfo_data;
      x86::Gp v_prevhash = Jitcc.newUIntPtr("prevhash.uintptr");
      /*
       * When performing the next hash and not in strict mode we
       * perform a rotation of the previously stored hash value
       * before doing the NULL check.  We want to do this even
       * when we receive a NULL Datum to hash.  In strict mode,
       * we do this after the NULL check so as not to waste the
       * effort of rotating the bits when we're going to throw
       * away the hash value and return NULL.
       */
      if (Opcode == EEOP_HASHDATUM_NEXT32) {
        /*
         * Fetch the previously hashed value from where the
         * EEOP_HASHDATUM_FIRST operation stored it.
         */
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr", Op->resvalue);
        EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_prevhash, sizeof(Datum));

        /*
         * Rotate bits left by 1 bit.  Be careful not to
         * overflow uint32 when working with size_t.
         */
        x86::Gp v_tmp = Jitcc.newUInt64("tmp.u64");
        Jitcc.mov(v_tmp, v_prevhash);
        Jitcc.shl(v_tmp, jit::imm(1));
        Jitcc.and_(v_tmp, jit::imm(0xffffffff));
        Jitcc.shr(v_prevhash, jit::imm(31));
        Jitcc.or_(v_prevhash, v_tmp);
      }

      /* We expect the hash function to have 1 argument */
      if (fcinfo->nargs != 1)
        ereport(ERROR, (errmsg("incorrect number of function arguments")));

      x86::Gp v_fcinfo = EmitLoadConstUIntPtr(Jitcc, "fcinfo.uintptr", fcinfo);
      /* emit code to check if the input parameter is NULL */
      x86::Gp v_argisnull = LoadFuncArgNull(Jitcc, v_fcinfo, 0);
      Jitcc.cmp(v_argisnull, jit::imm(1));
      Jitcc.je(L_IfNull);
      {
        /* If not null. */
        /*
         * Rotate the previously stored hash value when performing
         * NEXT32 in strict mode.  In non-strict mode we already
         * did this before checking for NULLs.
         */
        if (Opcode == EEOP_HASHDATUM_NEXT32_STRICT) {
          /*
           * Fetch the previously hashed value from where the
           * EEOP_HASHDATUM_FIRST_STRICT operation stored it.
           */
          x86::Gp v_resvaluep =
              EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr", Op->resvalue);
          EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_prevhash, sizeof(Datum));

          /*
           * Rotate bits left by 1 bit.  Be careful not to
           * overflow uint32 when working with size_t.
           */
          x86::Gp v_tmp = Jitcc.newUInt64("v_tmp.u64");
          Jitcc.mov(v_tmp, v_prevhash);
          Jitcc.shl(v_tmp, jit::imm(1));
          Jitcc.and_(v_tmp, jit::imm(0xffffffff));
          Jitcc.shr(v_prevhash, jit::imm(31));
          Jitcc.or_(v_prevhash, v_tmp);
        }

        /* call the hash function */
        x86::Gp v_retval = Jitcc.newUInt64("v_retval.u64");
        jit::InvokeNode *PGFunc;
        Jitcc.invoke(&PGFunc, jit::imm(fcinfo->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->setArg(0, v_fcinfo);
        PGFunc->setRet(0, v_retval);
        /*
         * For NEXT32 ops, XOR (^) the returned hash value with
         * the existing hash value.
         */
        if (Opcode == EEOP_HASHDATUM_NEXT32 ||
            Opcode == EEOP_HASHDATUM_NEXT32_STRICT)
          Jitcc.xor_(v_retval, v_prevhash);

        x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                   Op->resvalue),
                v_resnullp = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                                  Op->resnull);
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

        Jitcc.jmp(L_Opblocks[OpIndex + 1]);
      }

      Jitcc.bind(L_IfNull);
      {
        if (Opcode == EEOP_HASHDATUM_FIRST_STRICT ||
            Opcode == EEOP_HASHDATUM_NEXT32_STRICT) {
          /*
           * In strict node, NULL inputs result in NULL.  Save
           * the NULL result and goto jumpdone.
           */
          x86::Gp v_resvaluep = EmitLoadConstUIntPtr(
                      Jitcc, "op.resvalue.uintptr", Op->resvalue),
                  v_resnullp = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                                    Op->resnull);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
          EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));

          Jitcc.jmp(L_Opblocks[Op->d.hashdatum.jumpdone]);
        } else {
          x86::Gp v_resvaluep = EmitLoadConstUIntPtr(
                      Jitcc, "op.resvalue.uintptr", Op->resvalue),
                  v_resnullp = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                                    Op->resnull);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

          if (Opcode == EEOP_HASHDATUM_NEXT32) {
            /* Assert(v_prevhash != NULL) */
            EmitStoreToArray(Jitcc, v_resvaluep, 0, v_prevhash, sizeof(Datum));
          } else {
            Assert(Opcode == EEOP_HASHDATUM_FIRST);
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          }

          Jitcc.jmp(L_Opblocks[OpIndex + 1]);
        }
      }

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

    case EEOP_JSON_CONSTRUCTOR: {
      BuildEvalXFunc3(ExecEvalJsonConstructor);
      break;
    }

    case EEOP_IS_JSON: {
      BuildEvalXFunc2(ExecEvalJsonIsPredicate);
      break;
    }

    case EEOP_JSONEXPR_PATH: {
      todo();
    }

    case EEOP_JSONEXPR_COERCION: {
      BuildEvalXFunc3(ExecEvalJsonCoercion);
      break;
    }
    case EEOP_JSONEXPR_COERCION_FINISH: {
      BuildEvalXFunc2(ExecEvalJsonCoercionFinish);
      break;
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
      BuildEvalXFunc3(ExecEvalMergeSupportFunc);
      break;
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
