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

  x86::Gp v_state = Jitcc.newUIntPtr("v_state.uintptr"),
          v_econtext = Jitcc.newUIntPtr("v_econtext.uintptr"),
          v_isnullp = Jitcc.newUIntPtr("v_isnullp.uintptr");

  JittedFunc->setArg(0, v_state);
  JittedFunc->setArg(1, v_econtext);
  JittedFunc->setArg(2, v_isnullp);

  jit::Label *L_opblocks =
      (jit::Label *)palloc(State->steps_len * sizeof(jit::Label));
  for (int opno = 0; opno < State->steps_len; ++opno)
    L_opblocks[opno] = Jitcc.newLabel();

  for (int opno = 0; opno < State->steps_len; ++opno) {
    ExprEvalStep *op = &State->steps[opno];
    ExprEvalOp opcode = ExecEvalStepOp(State, op);

    Jitcc.bind(L_opblocks[opno]);

#define BuildEvalXFunc2(Func)                                                  \
  do {                                                                         \
    jit::InvokeNode *JitFunc;                                                  \
    Jitcc.invoke(                                                              \
        &JitFunc, jit::imm(Func),                                              \
        jit::FuncSignature::build<void, ExprState *, ExprEvalStep *>());       \
    JitFunc->setArg(0, v_state);                                               \
    JitFunc->setArg(1, jit::imm(op));                                          \
  } while (0);

#define BuildEvalXFunc3(Func)                                                  \
  do {                                                                         \
    jit::InvokeNode *JitFunc;                                                  \
    Jitcc.invoke(&JitFunc, jit::imm(Func),                                     \
                 jit::FuncSignature::build<void, ExprState *, ExprEvalStep *,  \
                                           ExprContext *>());                  \
    JitFunc->setArg(0, v_state);                                               \
    JitFunc->setArg(1, jit::imm(op));                                          \
    JitFunc->setArg(2, v_econtext);                                            \
  } while (0);

    switch (opcode) {
    case EEOP_DONE: {
      /* Load expression->resvalue and expression->resnull */
      x86::Gp v_resvalue = emit_load_resvalue_from_ExprState(Jitcc, v_state),
              v_resnull = emit_load_resnull_from_ExprState(Jitcc, v_state);

      /* *isnull = expression->resnull */
      EmitStoreToArray(Jitcc, v_isnullp, 0, v_resnull, sizeof(bool));

      /* return expression->resvalue */
      Jitcc.ret(v_resvalue);
      Jitcc.endFunc();
      break;
    }
    case EEOP_INNER_FETCHSOME:
    case EEOP_OUTER_FETCHSOME:
    case EEOP_SCAN_FETCHSOME: {
      const TupleTableSlotOps *tts_ops =
          op->d.fetch.fixed ? op->d.fetch.kind : nullptr;
      TupleDesc desc = op->d.fetch.known_desc;
      TupleDeformingFunc jit_deform = nullptr;

      /* Step should not have been generated. */
      Assert(tts_ops != &TTSOpsVirtual);

      /* Compute the address of Slot->tts_nvalid */
      x86::Gp v_slot =
          opcode == EEOP_INNER_FETCHSOME
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, v_econtext)
              : (opcode == EEOP_OUTER_FETCHSOME
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  v_econtext)
                     : emit_load_ecxt_scantuple_from_ExprContext(Jitcc,
                                                                 v_econtext));

      x86::Gp v_nvalid =
          emit_load_tts_nvalid_from_TupleTableSlot(Jitcc, v_slot);

      /*
       * Check if all required attributes are available, or whether deforming is
       * required.
       */
      Jitcc.cmp(v_nvalid, jit::imm(op->d.fetch.last_var));
      Jitcc.jge(L_opblocks[opno + 1]);

      if (tts_ops && desc && (Context->base.flags & PGJIT_DEFORM)) {
        INSTR_TIME_SET_CURRENT(DeformStartTime);

        jit_deform = CompileTupleDeformingFunc(Context, Runtime, desc, tts_ops,
                                               op->d.fetch.last_var);

        INSTR_TIME_SET_CURRENT(DeformEndTime);
        INSTR_TIME_ACCUM_DIFF(Context->base.instr.deform_counter, DeformEndTime,
                              DeformStartTime);
      }

      jit::InvokeNode *SlotGetSomeAttrsInt = nullptr;
      if (jit_deform) {
        /* Invoke the JIT-ed deforming function. */
        Jitcc.invoke(&SlotGetSomeAttrsInt, jit::imm(jit_deform),
                     jit::FuncSignature::build<void, TupleTableSlot *>());
        SlotGetSomeAttrsInt->setArg(0, v_slot);
      } else {
        Jitcc.invoke(&SlotGetSomeAttrsInt, jit::imm(slot_getsomeattrs_int),
                     jit::FuncSignature::build<void, TupleTableSlot *, int>());
        SlotGetSomeAttrsInt->setArg(0, v_slot);
        SlotGetSomeAttrsInt->setArg(1, jit::imm(op->d.fetch.last_var));
      }

      break;
    }

    case EEOP_INNER_VAR:
    case EEOP_OUTER_VAR:
    case EEOP_SCAN_VAR: {
      x86::Gp v_slot =
          opcode == EEOP_INNER_VAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, v_econtext)
              : (opcode == EEOP_OUTER_VAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  v_econtext)
                     : emit_load_ecxt_scantuple_from_ExprContext(Jitcc,
                                                                 v_econtext));

      x86::Gp v_values =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot),
              v_nulls = emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_slot);

      int attnum = op->d.var.attnum;

      x86::Gp v_value = Jitcc.newUIntPtr(), v_isnull = Jitcc.newInt8();
      EmitLoadFromArray(Jitcc, v_values, attnum, v_value, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_nulls, attnum, v_isnull, sizeof(bool));

      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);

      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_value, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_isnull, sizeof(bool));

      break;
    }
    case EEOP_INNER_SYSVAR:
    case EEOP_OUTER_SYSVAR:
    case EEOP_SCAN_SYSVAR: {
      x86::Gp v_slot =
          opcode == EEOP_INNER_VAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, v_econtext)
              : (opcode == EEOP_OUTER_SYSVAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  v_econtext)
                     : emit_load_ecxt_scantuple_from_ExprContext(Jitcc,
                                                                 v_econtext));

      jit::InvokeNode *ExecEvalSysVarFunc;
      Jitcc.invoke(
          &ExecEvalSysVarFunc, jit::imm(ExecEvalSysVar),
          jit::FuncSignature::build<void, ExprState *, ExprEvalStep *,
                                    ExprContext *, TupleTableSlot *>());
      ExecEvalSysVarFunc->setArg(0, v_state);
      ExecEvalSysVarFunc->setArg(1, jit::imm(op));
      ExecEvalSysVarFunc->setArg(2, v_econtext);
      ExecEvalSysVarFunc->setArg(3, v_slot);
      break;
    }

    case EEOP_WHOLEROW: {
      BuildEvalXFunc3(ExecEvalWholeRowVar);
      break;
    }

    case EEOP_ASSIGN_INNER_VAR:
    case EEOP_ASSIGN_OUTER_VAR:
    case EEOP_ASSIGN_SCAN_VAR: {
      x86::Gp v_slot =
          opcode == EEOP_ASSIGN_INNER_VAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, v_econtext)
              : (opcode == EEOP_ASSIGN_OUTER_VAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  v_econtext)
                     : emit_load_ecxt_scantuple_from_ExprContext(Jitcc,
                                                                 v_econtext));

      x86::Gp v_values =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot),
              v_nulls = emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_slot);

      int attnum = op->d.assign_var.attnum;

      /* Load data. */
      x86::Gp v_value = Jitcc.newUIntPtr("v_value.uintptr"),
              v_null = Jitcc.newInt8("v_null.i8");
      EmitLoadFromArray(Jitcc, v_values, attnum, v_value, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_nulls, attnum, v_null, sizeof(bool));

      /* Save the result. */
      int resultnum = op->d.assign_var.resultnum;
      x86::Gp v_resultslot =
          emit_load_resultslot_from_ExprState(Jitcc, v_state);
      x86::Gp v_rvaluep =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, v_resultslot),
              v_risnullp =
                  emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_resultslot);

      EmitStoreToArray(Jitcc, v_rvaluep, resultnum, v_value, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_risnullp, resultnum, v_null, sizeof(bool));

      break;
    }

    case EEOP_ASSIGN_TMP:
    case EEOP_ASSIGN_TMP_MAKE_RO: {
      size_t resultnum = op->d.assign_tmp.resultnum;

      /* Load expression->resvalue and expression->resnull */
      x86::Gp v_rvalue = emit_load_resvalue_from_ExprState(Jitcc, v_state),
              v_risnull = emit_load_resnull_from_ExprState(Jitcc, v_state);

      /*
       * Compute the addresses of
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      x86::Gp v_resultslot =
          emit_load_resultslot_from_ExprState(Jitcc, v_state);
      x86::Gp v_tmpvaluep =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, v_resultslot),
              v_tmpisnullp =
                  emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_resultslot);
      /*
       * Store nullness.
       */
      EmitStoreToArray(Jitcc, v_tmpisnullp, resultnum, v_risnull, sizeof(bool));

      if (opcode == EEOP_ASSIGN_TMP_MAKE_RO) {
        Jitcc.cmp(v_risnull, jit::imm(1));
        Jitcc.je(L_opblocks[opno + 1]);

        jit::InvokeNode *MakeExpandedObjectReadOnlyInternalFunc;
        Jitcc.invoke(&MakeExpandedObjectReadOnlyInternalFunc,
                     jit::imm(MakeExpandedObjectReadOnlyInternal),
                     jit::FuncSignature::build<Datum, Datum>());
        MakeExpandedObjectReadOnlyInternalFunc->setArg(0, v_rvalue);
        MakeExpandedObjectReadOnlyInternalFunc->setRet(0, v_rvalue);
      }

      /* Finally, store the result. */
      EmitStoreToArray(Jitcc, v_tmpvaluep, resultnum, v_rvalue, sizeof(Datum));
      break;
    }
    case EEOP_CONST: {
      x86::Gp v_constvalue = EmitLoadConstUInt64(Jitcc, "constval.value.u64",
                                                 op->d.constval.value),
              v_constnull = EmitLoadConstUInt8(Jitcc, "constval.isnull.u8",
                                               op->d.constval.isnull);

      /*
       * Store Op->d.constval.value to Op->resvalue.
       * Store Op->d.constval.isnull to Op->resnull.
       */
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "op.resnull.uintptr",
                                                op->resnull);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_constvalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_constnull, sizeof(bool));

      break;
    }
    case EEOP_FUNCEXPR:
    case EEOP_FUNCEXPR_STRICT: {
      FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
      x86::Gp v_fcinfo =
          EmitLoadConstUIntPtr(Jitcc, "v_fcinfo.uintptr", fcinfo);

      jit::Label L_InvokePGFunc = Jitcc.newLabel();

      if (opcode == EEOP_FUNCEXPR_STRICT) {
        jit::Label L_StrictFail = Jitcc.newLabel();
        /* Should make sure that they're optimized beforehand. */
        int argnum = op->d.func.nargs;
        if (argnum == 0) {
          ereport(ERROR,
                  (errmsg("Argumentless strict functions are pointless")));
        }

        /* Check for NULL args for strict function. */
        for (int argno = 0; argno < argnum; ++argno) {
          x86::Gp v_argisnull = LoadFuncArgNull(Jitcc, v_fcinfo, argno);
          Jitcc.cmp(v_argisnull, jit::imm(1));
          Jitcc.je(L_StrictFail);
        }

        Jitcc.jmp(L_InvokePGFunc);

        Jitcc.bind(L_StrictFail);
        /* Op->resnull = true */
        x86::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr", op->resnull);
        EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
        Jitcc.jmp(L_opblocks[opno + 1]);
      }

      /*
       * Before invoking PGFuncs, we should set FuncCallInfo->isnull to false.
       */
      Jitcc.bind(L_InvokePGFunc);
      emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo,
                                                    jit::imm(0));

      jit::InvokeNode *PGFunc;
      x86::Gp v_retval = Jitcc.newUIntPtr("v_retval.uintptr");
      Jitcc.invoke(&PGFunc, jit::imm(fcinfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->setArg(0, v_fcinfo);
      PGFunc->setRet(0, v_retval);

      /* Write result values. */
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);

      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
      x86::Gp v_fcinfo_isnull =
          emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_fcinfo_isnull, sizeof(bool));

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
      x86::Gp v_boolanynullp = EmitLoadConstUIntPtr(
          Jitcc, "op.d.boolexpr.anynull", op->d.boolexpr.anynull);
      jit::Label L_BoolCheckFalse = Jitcc.newLabel(),
                 L_BoolCont = Jitcc.newLabel();

      if (opcode == EEOP_BOOL_AND_STEP_FIRST)
        EmitStoreToArray(Jitcc, v_boolanynullp, 0, jit::imm(0), sizeof(bool));

      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      x86::Gp v_boolvalue = Jitcc.newUIntPtr("v_boolvalue.uintptr"),
              v_boolnull = Jitcc.newInt8("v_boolnull.i8");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_boolvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_boolnull, sizeof(bool));

      /* check if current input is NULL */
      Jitcc.cmp(v_boolnull, jit::imm(1));
      Jitcc.jne(L_BoolCheckFalse);
      {
        /* b_boolisnull */
        /* set boolanynull to true */
        EmitStoreToArray(Jitcc, v_boolanynullp, 0, jit::imm(1), sizeof(bool));
        Jitcc.jmp(L_BoolCont);
      }

      Jitcc.bind(L_BoolCheckFalse);
      {
        Jitcc.cmp(v_boolvalue, jit::imm(0));
        Jitcc.jne(L_BoolCont);

        /* b_boolisfalse */
        /* result is already set to FALSE, need not change it */
        /* and jump to the end of the AND expression */
        Jitcc.jmp(L_opblocks[op->d.boolexpr.jumpdone]);
      }

      Jitcc.bind(L_BoolCont);
      {
        x86::Gp v_boolanynull = Jitcc.newInt8("v_boolanynull.i8");
        EmitLoadFromArray(Jitcc, v_boolanynullp, 0, v_boolanynull,
                          sizeof(bool));
        Jitcc.cmp(v_boolanynull, jit::imm(0));
        Jitcc.je(L_opblocks[opno + 1]);
      }

      /* set resnull to true */
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
      /* reset resvalue */
      EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));

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
      x86::Gp v_boolanynullp = EmitLoadConstUIntPtr(
          Jitcc, "v_boolanynullp.uintptr", op->d.boolexpr.anynull);
      jit::Label L_BoolCheckTrue = Jitcc.newLabel(),
                 L_BoolCont = Jitcc.newLabel();

      if (opcode == EEOP_BOOL_OR_STEP_FIRST)
        EmitStoreToArray(Jitcc, v_boolanynullp, 0, jit::imm(0), sizeof(bool));

      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      x86::Gp v_boolvalue = Jitcc.newUIntPtr("v_boolvalue.uintptr"),
              v_boolnull = Jitcc.newInt8("v_boolnull.i8");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_boolvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_boolnull, sizeof(bool));

      /* check if current input is NULL */
      Jitcc.cmp(v_boolnull, jit::imm(1));
      Jitcc.jne(L_BoolCheckTrue);
      {
        /* b_boolisnull */
        /* set boolanynull to true */
        EmitStoreToArray(Jitcc, v_boolanynullp, 0, jit::imm(1), sizeof(bool));
        Jitcc.jmp(L_BoolCont);
      }

      Jitcc.bind(L_BoolCheckTrue);
      {
        Jitcc.cmp(v_boolvalue, jit::imm(1));
        Jitcc.jne(L_BoolCont);

        /* b_boolistrue */
        /* result is already set to FALSE, need not change it */
        /* and jump to the end of the AND expression */
        Jitcc.jmp(L_opblocks[op->d.boolexpr.jumpdone]);
      }

      Jitcc.bind(L_BoolCont);
      {
        x86::Gp v_boolanynull = Jitcc.newInt8("v_boolanynull.i8");
        EmitLoadFromArray(Jitcc, v_boolanynullp, 0, v_boolanynull,
                          sizeof(bool));
        Jitcc.cmp(v_boolanynull, jit::imm(0));
        Jitcc.je(L_opblocks[opno + 1]);
      }

      /* set resnull to true */
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
      /* reset resvalue */
      EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
      break;
    }

    case EEOP_BOOL_NOT_STEP: {
      x86::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr", op->resvalue);
      x86::Gp v_boolvalue = Jitcc.newUIntPtr("v_boolvalue.uintptr"),
              v_negbool = Jitcc.newUIntPtr("v_negbool.uintptr");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_boolvalue, sizeof(Datum));
      Jitcc.xor_(v_negbool, v_negbool);
      Jitcc.cmp(v_boolvalue, jit::imm(0));
      Jitcc.sete(v_negbool);

      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_negbool, sizeof(Datum));
      break;
    }

    case EEOP_QUAL: {
      jit::Label L_HandleNullOrFalse = Jitcc.newLabel();

      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      x86::Gp v_resvalue = Jitcc.newUIntPtr("v_resvalue.uintptr"),
              v_resnull = Jitcc.newInt8("v_resnull.i8");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      Jitcc.cmp(v_resnull, jit::imm(1));
      Jitcc.je(L_HandleNullOrFalse);

      Jitcc.cmp(v_resvalue, jit::imm(0));
      Jitcc.je(L_HandleNullOrFalse);

      Jitcc.jmp(L_opblocks[opno + 1]);

      /* Handling null or false. */
      Jitcc.bind(L_HandleNullOrFalse);

      /* Set resnull and resvalue to false. */
      EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

      Jitcc.jmp(L_opblocks[op->d.qualexpr.jumpdone]);

      break;
    }

    case EEOP_JUMP: {
      Jitcc.jmp(L_opblocks[op->d.jump.jumpdone]);
      break;
    }

    case EEOP_JUMP_IF_NULL: {
      /* Transfer control if current result is null */
      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr", op->resnull);
      x86::Gp v_resnull = Jitcc.newInt8("v_resnull.i8");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      Jitcc.cmp(v_resnull, jit::imm(1));
      Jitcc.je(L_opblocks[op->d.jump.jumpdone]);

      break;
    }

    case EEOP_JUMP_IF_NOT_NULL: {
      /* Transfer control if current result is non-null */
      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr", op->resnull);
      x86::Gp v_resnull = Jitcc.newInt8("v_resnull.i8");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      Jitcc.cmp(v_resnull, jit::imm(0));
      Jitcc.je(L_opblocks[op->d.jump.jumpdone]);

      break;
    }

    case EEOP_JUMP_IF_NOT_TRUE: {
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      x86::Gp v_resvalue = Jitcc.newUIntPtr("v_resvalue.uintptr"),
              v_resnull = Jitcc.newInt8("v_resnull.i8");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      /* Transfer control if current result is null or false */
      Jitcc.cmp(v_resnull, jit::imm(1));
      Jitcc.je(L_opblocks[op->d.jump.jumpdone]);
      Jitcc.cmp(v_resvalue, jit::imm(0));
      Jitcc.je(L_opblocks[op->d.jump.jumpdone]);

      break;
    }

    case EEOP_NULLTEST_ISNULL: {
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      x86::Gp v_resvalue = Jitcc.newUIntPtr("v_resvalue.uintptr"),
              v_resnull = Jitcc.newInt8("v_resnull.i8");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
      Jitcc.xor_(v_resvalue, v_resvalue);
      Jitcc.cmp(v_resnull, jit::imm(1));
      Jitcc.sete(v_resvalue);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

      break;
    }

    case EEOP_NULLTEST_ISNOTNULL: {
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      x86::Gp v_resvalue = Jitcc.newUIntPtr("v_resvalue.uintptr"),
              v_resnull = Jitcc.newInt8("v_resnull.i8");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
      Jitcc.xor_(v_resvalue, v_resvalue);
      Jitcc.cmp(v_resnull, jit::imm(0));
      Jitcc.sete(v_resvalue);

      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

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
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      x86::Gp v_resnull = Jitcc.newInt8("resnull.i8");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
      Jitcc.cmp(v_resnull, jit::imm(1));
      Jitcc.jne(L_NotNull);
      /* result is not null. */
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
      EmitStoreToArray(
          Jitcc, v_resvaluep, 0,
          (opcode == EEOP_BOOLTEST_IS_TRUE || opcode == EEOP_BOOLTEST_IS_FALSE)
              ? jit::imm(0)
              : jit::imm(1),
          sizeof(Datum));
      Jitcc.jmp(L_opblocks[opno + 1]);

      Jitcc.bind(L_NotNull);
      if (opcode == EEOP_BOOLTEST_IS_TRUE ||
          opcode == EEOP_BOOLTEST_IS_NOT_FALSE) {
        /*
         * if value is not null NULL, return value (already
         * set)
         */
      } else {
        x86::Gp v_resvalue = Jitcc.newUIntPtr("v_resvalue.uintptr");
        x86::Gp v_resvalue_is_false =
            Jitcc.newUIntPtr("v_resvalue_is_false.uintptr");
        EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
        Jitcc.xor_(v_resvalue_is_false, v_resvalue_is_false);
        Jitcc.cmp(v_resvalue, jit::imm(0));
        Jitcc.sete(v_resvalue_is_false);
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_resvalue_is_false,
                         sizeof(Datum));
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
      BuildEvalXFunc3(op->d.cparam.paramfunc);
      break;
    }

    case EEOP_PARAM_SET: {
      BuildEvalXFunc3(ExecEvalParamSet);
      break;
    }

    case EEOP_SBSREF_SUBSCRIPTS: {
      jit::InvokeNode *InvokeSubscriptFunc;
      x86::Gp v_retval = Jitcc.newInt8("ret.i8");
      Jitcc.invoke(
          &InvokeSubscriptFunc, jit::imm(op->d.sbsref_subscript.subscriptfunc),
          jit::FuncSignature::build<bool, ExprState *, struct ExprEvalStep *,
                                    ExprContext *>());
      InvokeSubscriptFunc->setArg(0, v_state);
      InvokeSubscriptFunc->setArg(1, jit::imm(op));
      InvokeSubscriptFunc->setArg(2, v_econtext);
      InvokeSubscriptFunc->setRet(0, v_retval);

      Jitcc.cmp(v_retval, jit::imm(0));
      Jitcc.je(L_opblocks[op->d.sbsref_subscript.jumpdone]);
      break;
    }

    case EEOP_SBSREF_OLD:
    case EEOP_SBSREF_ASSIGN:
    case EEOP_SBSREF_FETCH: {
      BuildEvalXFunc3(op->d.sbsref.subscriptfunc);
      break;
    }

    case EEOP_CASE_TESTVAL: {
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      if (op->d.casetest.value) {
        x86::Gp v_casevaluep = EmitLoadConstUIntPtr(
                    Jitcc, "v_casevaluep.uintptr", op->d.casetest.value),
                v_casenullp = EmitLoadConstUIntPtr(Jitcc, "v_casenullp.uintptr",
                                                   op->d.casetest.isnull);
        x86::Gp v_casevalue = Jitcc.newUIntPtr("v_casevalue.uintptr"),
                v_casenull = Jitcc.newInt8("v_casenull.i8");
        EmitLoadFromArray(Jitcc, v_casevaluep, 0, v_casevalue, sizeof(Datum));
        EmitLoadFromArray(Jitcc, v_casenullp, 0, v_casenull, sizeof(bool));

        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
      } else {
        x86::Gp v_casevalue =
            emit_load_caseValue_datum_from_ExprContext(Jitcc, v_econtext);
        x86::Gp v_casenull =
            emit_load_caseValue_isNull_from_ExprContext(Jitcc, v_econtext);

        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
      }
      break;
    }
    case EEOP_MAKE_READONLY: {
      x86::Gp v_nullp = EmitLoadConstUIntPtr(Jitcc, "v_nullp.uintptr",
                                             op->d.make_readonly.isnull);
      x86::Gp v_null = Jitcc.newInt8("v_null.i8");

      EmitLoadFromArray(Jitcc, v_nullp, 0, v_null, sizeof(bool));

      /* store null isnull value in result */
      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr", op->resnull);
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_null, sizeof(bool));

      Jitcc.cmp(v_null, jit::imm(1));
      Jitcc.je(L_opblocks[opno + 1]);

      /* if value is not null, convert to RO datum */
      x86::Gp v_valuep = EmitLoadConstUIntPtr(Jitcc, "v_valuep.uintptr",
                                              op->d.make_readonly.value);
      x86::Gp v_value = Jitcc.newUIntPtr("v_value.uintptr");
      EmitLoadFromArray(Jitcc, v_valuep, 0, v_value, sizeof(Datum));
      jit::InvokeNode *InvokeMakeExpandedObjectReadOnly;
      Jitcc.invoke(&InvokeMakeExpandedObjectReadOnly,
                   jit::imm(MakeExpandedObjectReadOnlyInternal),
                   jit::FuncSignature::build<Datum, Datum>());
      InvokeMakeExpandedObjectReadOnly->setArg(0, v_value);
      InvokeMakeExpandedObjectReadOnly->setRet(0, v_value);

      x86::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr", op->resvalue);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_value, sizeof(Datum));
      break;
    }

    case EEOP_IOCOERCE: {
      FunctionCallInfo fcinfo_out = op->d.iocoerce.fcinfo_data_out,
                       fcinfo_in = op->d.iocoerce.fcinfo_data_in;
      jit::Label L_SkipOutputCall = Jitcc.newLabel(),
                 L_InputCall = Jitcc.newLabel();

      x86::Gp v_fcinfo_out = EmitLoadConstUIntPtr(Jitcc, "v_fcinfo_out.uintptr",
                                                  fcinfo_out),
              v_fcinfo_in =
                  EmitLoadConstUIntPtr(Jitcc, "v_fcinfo_in.uintptr", fcinfo_in);
      x86::Gp v_output = Jitcc.newUInt64("v_output.u64");

      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", op->resnull);
      x86::Gp v_resnull = Jitcc.newInt8("op.resnull.i8");
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      Jitcc.cmp(v_resnull, jit::imm(1));
      Jitcc.je(L_SkipOutputCall);
      {
        /* Not null, call output. */
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr", op->resvalue);
        x86::Gp v_resvalue = Jitcc.newUIntPtr("v_resvalue");
        EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
        StoreFuncArgValue(Jitcc, v_fcinfo_out, 0, v_resvalue);
        StoreFuncArgNull(Jitcc, v_fcinfo_out, 0, jit::imm(0));
        emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo_out,
                                                      jit::imm(0));

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
        if (op->d.iocoerce.finfo_in->fn_strict) {
          Jitcc.cmp(v_output, jit::imm(0));
          Jitcc.je(L_opblocks[opno + 1]);
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
            EmitLoadConstUIntPtr(Jitcc, "op.resvalue.uintptr", op->resvalue);
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
      FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
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
              EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", op->resnull);
          x86::Gp v_resvaluep =
              EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", op->resvalue);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
          if (opcode == EEOP_NOT_DISTINCT)
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(1), sizeof(Datum));
          else
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          Jitcc.jmp(L_opblocks[opno + 1]);
        }

        Jitcc.bind(L_AnyArgIsNull);
        {
          x86::Gp v_resnullp =
              EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", op->resnull);
          x86::Gp v_resvaluep =
              EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", op->resvalue);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
          if (opcode == EEOP_NOT_DISTINCT)
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          else
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(1), sizeof(Datum));
          Jitcc.jmp(L_opblocks[opno + 1]);
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
        if (opcode == EEOP_DISTINCT) {
          /* Must invert the result of "=" */
          x86::Gp v_tmpretval = Jitcc.newUInt64("v_tmpretval.u64");
          Jitcc.mov(v_tmpretval, v_retval);
          Jitcc.xor_(v_retval, v_retval);
          Jitcc.cmp(v_tmpretval, jit::imm(0));
          Jitcc.sete(v_retval);
        }
        x86::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", op->resnull);
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", op->resvalue);

        EmitStoreToArray(Jitcc, v_resnullp, 0, v_fcinfo_isnull, sizeof(bool));
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
      }
      break;
    }

    case EEOP_NULLIF: {
      FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
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
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", op->resnull);
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", op->resvalue);
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_argnull0, sizeof(bool));
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_arg0, sizeof(Datum));
        Jitcc.jmp(L_opblocks[opno + 1]);
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
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", op->resnull);
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", op->resvalue);
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
      FunctionCallInfo fcinfo = op->d.rowcompare_step.fcinfo_data;
      x86::Gp v_fcinfo =
          EmitLoadConstUIntPtr(Jitcc, "v_fcinfo.uintptr", fcinfo);
      jit::Label L_Null = Jitcc.newLabel();
      /*
       * If function is strict, and either arg is null, we're
       * done.
       */
      if (op->d.rowcompare_step.finfo->fn_strict) {
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
          EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", op->resvalue);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
      /* if result of function is NULL, force NULL result */
      Jitcc.cmp(v_fcinfo_isnull, jit::imm(0));
      Jitcc.jne(L_Null);
      /* if results equal, compare next, otherwise done */
      Jitcc.cmp(v_retval, jit::imm(0));
      Jitcc.je(L_opblocks[opno + 1]);
      Jitcc.jmp(L_opblocks[op->d.rowcompare_step.jumpdone]);

      Jitcc.bind(L_Null);
      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", op->resnull);
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
      Jitcc.jmp(L_opblocks[op->d.rowcompare_step.jumpnull]);
      break;
    }

    case EEOP_ROWCOMPARE_FINAL: {
      RowCompareType rctype = op->d.rowcompare_final.rctype;

      /*
       * Btree comparators return 32 bit results, need to be
       * careful about sign (used as a 64 bit value it's
       * otherwise wrong).
       */
      x86::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "op.resvaluep.uintptr", op->resvalue);
      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp.uintptr", op->resnull);
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
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      if (op->d.casetest.value) {
        x86::Gp v_casevaluep = EmitLoadConstUIntPtr(
                    Jitcc, "v_casevaluep.uintptr", op->d.casetest.value),
                v_casenullp = EmitLoadConstUIntPtr(Jitcc, "v_casenullp.uintptr",
                                                   op->d.casetest.isnull);
        x86::Gp v_casevalue = Jitcc.newUIntPtr("v_casevalue.uintptr"),
                v_casenull = Jitcc.newInt8("v_casenull.i8");
        EmitLoadFromArray(Jitcc, v_casevaluep, 0, v_casevalue, sizeof(Datum));
        EmitLoadFromArray(Jitcc, v_casenullp, 0, v_casenull, sizeof(bool));

        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
      } else {
        x86::Gp v_casevalue =
            emit_load_domainValue_datum_from_ExprContext(Jitcc, v_econtext);
        x86::Gp v_casenull =
            emit_load_domainValue_isNull_from_ExprContext(Jitcc, v_econtext);

        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
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
      x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                 op->resvalue),
              v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                op->resnull);
      EmitStoreToArray(Jitcc, v_resvaluep, 0,
                       jit::imm(op->d.hashdatum_initvalue.init_value),
                       sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

      break;
    }
    case EEOP_HASHDATUM_FIRST:
    case EEOP_HASHDATUM_FIRST_STRICT:
    case EEOP_HASHDATUM_NEXT32:
    case EEOP_HASHDATUM_NEXT32_STRICT: {
      jit::Label L_IfNull = Jitcc.newLabel();
      FunctionCallInfo fcinfo = op->d.hashdatum.fcinfo_data;
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
      if (opcode == EEOP_HASHDATUM_NEXT32) {
        /*
         * Fetch the previously hashed value from where the
         * EEOP_HASHDATUM_FIRST operation stored it.
         */
        x86::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr", op->resvalue);
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
        if (opcode == EEOP_HASHDATUM_NEXT32_STRICT) {
          /*
           * Fetch the previously hashed value from where the
           * EEOP_HASHDATUM_FIRST_STRICT operation stored it.
           */
          x86::Gp v_resvaluep =
              EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr", op->resvalue);
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
        if (opcode == EEOP_HASHDATUM_NEXT32 ||
            opcode == EEOP_HASHDATUM_NEXT32_STRICT)
          Jitcc.xor_(v_retval, v_prevhash);

        x86::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr",
                                                   op->resvalue),
                v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                  op->resnull);
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

        Jitcc.jmp(L_opblocks[opno + 1]);
      }

      Jitcc.bind(L_IfNull);
      {
        if (opcode == EEOP_HASHDATUM_FIRST_STRICT ||
            opcode == EEOP_HASHDATUM_NEXT32_STRICT) {
          /*
           * In strict node, NULL inputs result in NULL.  Save
           * the NULL result and goto jumpdone.
           */
          x86::Gp v_resvaluep = EmitLoadConstUIntPtr(
                      Jitcc, "v_resvaluep.uintptr", op->resvalue),
                  v_resnullp =
                      EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
          EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));

          Jitcc.jmp(L_opblocks[op->d.hashdatum.jumpdone]);
        } else {
          x86::Gp v_resvaluep = EmitLoadConstUIntPtr(
                      Jitcc, "v_resvaluep.uintptr", op->resvalue),
                  v_resnullp = EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr",
                                                    op->resnull);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

          if (opcode == EEOP_HASHDATUM_NEXT32) {
            /* Assert(v_prevhash != NULL) */
            EmitStoreToArray(Jitcc, v_resvaluep, 0, v_prevhash, sizeof(Datum));
          } else {
            Assert(opcode == EEOP_HASHDATUM_FIRST);
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          }

          Jitcc.jmp(L_opblocks[opno + 1]);
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
      JsonExprState *jsestate = op->d.jsonexpr.jsestate;
      x86::Gp v_ret = Jitcc.newInt32("v_ret.i32");
      /*
       * Call ExecEvalJsonExprPath().  It returns the address of
       * the step to perform next.
       */
      jit::InvokeNode *InvokeExecEvalJsonExprPath;
      Jitcc.invoke(&InvokeExecEvalJsonExprPath, jit::imm(ExecEvalJsonExprPath),
                   jit::FuncSignature::build<int, ExprState *, ExprEvalStep *,
                                             ExprContext *>());
      InvokeExecEvalJsonExprPath->setArg(0, v_state);
      InvokeExecEvalJsonExprPath->setArg(1, jit::imm(op));
      InvokeExecEvalJsonExprPath->setArg(2, v_econtext);
      InvokeExecEvalJsonExprPath->setRet(0, v_ret);

      /*
       * Build a switch to map the return value (v_ret above),
       * which is a runtime value of the step address to perform
       * next, to either jump_empty, jump_error,
       * jump_eval_coercion, or jump_end.
       */
      if (jsestate->jump_empty >= 0) {
        Jitcc.cmp(v_ret, jit::imm(jsestate->jump_empty));
        Jitcc.je(L_opblocks[jsestate->jump_empty]);
      }

      if (jsestate->jump_error >= 0) {
        Jitcc.cmp(v_ret, jit::imm(jsestate->jump_error));
        Jitcc.je(L_opblocks[jsestate->jump_error]);
      }

      if (jsestate->jump_eval_coercion >= 0) {
        Jitcc.cmp(v_ret, jit::imm(jsestate->jump_eval_coercion));
        Jitcc.je(L_opblocks[jsestate->jump_eval_coercion]);
      }

      Jitcc.jmp(L_opblocks[jsestate->jump_end]);
      break;
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
      /*
       * Returns a Datum whose value is the precomputed aggregate value
       * found in the given expression context.
       */
      x86::Gp v_aggvaluesp =
          emit_load_ecxt_aggvalues_from_ExprContext(Jitcc, v_econtext);
      x86::Gp v_aggnullsp =
          emit_load_ecxt_aggnulls_from_ExprContext(Jitcc, v_econtext);
      x86::Gp v_value = Jitcc.newUIntPtr("v_value.uintptr");
      x86::Gp v_isnull = Jitcc.newInt8("v_isnull.i8");

      /* load agg value / null */
      EmitLoadFromArray(Jitcc, v_aggvaluesp, op->d.aggref.aggno, v_value,
                        sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_aggnullsp, op->d.aggref.aggno, v_isnull,
                        sizeof(bool));

      /* and store result */
      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr", op->resnull);
      x86::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr", op->resvalue);
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_isnull, sizeof(bool));
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_value, sizeof(Datum));

      break;
    }

    case EEOP_GROUPING_FUNC: {
      BuildEvalXFunc2(ExecEvalGroupingFunc);
      break;
    }

    case EEOP_WINDOW_FUNC: {
      WindowFuncExprState *wfunc = op->d.window_func.wfstate;
      /*
       * At this point aggref->wfuncno is not yet set (it's set
       * up in ExecInitWindowAgg() after initializing the
       * expression). So load it from memory each time round.
       */
      x86::Gp v_wfuncnop =
          EmitLoadConstUIntPtr(Jitcc, "v_wfuncnop.uintptr", &wfunc->wfuncno);
      x86::Gp v_wfuncno = Jitcc.newInt32("v_wfuncno.i32");
      EmitLoadFromArray(Jitcc, v_wfuncnop, 0, v_wfuncno, sizeof(int32));
      x86::Gp v_aggvaluesp =
          emit_load_ecxt_aggvalues_from_ExprContext(Jitcc, v_econtext);
      x86::Gp v_aggnullsp =
          emit_load_ecxt_aggnulls_from_ExprContext(Jitcc, v_econtext);
      x86::Gp v_value = Jitcc.newUIntPtr("v_value.uintptr");
      x86::Gp v_isnull = Jitcc.newInt8("v_isnull.i8");

      /* load agg value / null */
      x86::Mem m_aggvaluesp = x86::ptr(v_aggvaluesp, v_wfuncno, 3);
      x86::Mem m_aggnullsp = x86::ptr(v_aggnullsp, v_wfuncno, 0);
      Jitcc.mov(v_value, m_aggvaluesp);
      Jitcc.mov(v_isnull, m_aggnullsp);

      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr", op->resnull);
      x86::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr", op->resvalue);
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_isnull, sizeof(bool));
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_value, sizeof(Datum));

      break;
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
      FunctionCallInfo fcinfo = op->d.agg_deserialize.fcinfo_data;
      x86::Gp v_fcinfo =
          EmitLoadConstUIntPtr(Jitcc, "v_fcinfo.uintptr", fcinfo);

      if (opcode == EEOP_AGG_STRICT_DESERIALIZE) {
        x86::Gp v_argnull0 = LoadFuncArgNull(Jitcc, v_fcinfo, 0);
        Jitcc.cmp(v_argnull0, jit::imm(1));
        Jitcc.je(L_opblocks[op->d.agg_deserialize.jumpnull]);
      }

      AggState *aggstate = castNode(AggState, State->parent);
      x86::Gp v_tmpcontext =
          EmitLoadConstUIntPtr(Jitcc, "v_tmpcontext.uintptr",
                               aggstate->tmpcontext->ecxt_per_tuple_memory);
      x86::Gp v_oldcontext = Jitcc.newUIntPtr("v_oldcontext.uintptr");
      jit::InvokeNode *InvokeMemoryContextSwitchTo;
      Jitcc.invoke(&InvokeMemoryContextSwitchTo,
                   jit::imm(MemoryContextSwitchTo),
                   jit::FuncSignature::build<MemoryContext, MemoryContext>());
      InvokeMemoryContextSwitchTo->setArg(0, v_tmpcontext);
      InvokeMemoryContextSwitchTo->setRet(0, v_oldcontext);

      jit::InvokeNode *PGFunc;
      x86::Gp v_retval = Jitcc.newUIntPtr("v_retval.uintptr");
      Jitcc.invoke(&PGFunc, jit::imm(fcinfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->setArg(0, fcinfo);
      PGFunc->setRet(0, v_retval);
      x86::Gp v_fcinfo_isnull =
          emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);

      InvokeMemoryContextSwitchTo = nullptr;
      Jitcc.invoke(&InvokeMemoryContextSwitchTo,
                   jit::imm(MemoryContextSwitchTo),
                   jit::FuncSignature::build<MemoryContext, MemoryContext>());
      InvokeMemoryContextSwitchTo->setArg(0, v_oldcontext);
      InvokeMemoryContextSwitchTo->setRet(0, v_oldcontext);

      x86::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp.uintptr", op->resnull);
      x86::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep.uintptr", op->resvalue);
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_fcinfo_isnull, sizeof(bool));
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));

      break;
    }

    case EEOP_AGG_STRICT_INPUT_CHECK_ARGS:
    case EEOP_AGG_STRICT_INPUT_CHECK_NULLS: {
      int nargs = op->d.agg_strict_input_check.nargs;
      NullableDatum *args = op->d.agg_strict_input_check.args;
      bool *nulls = op->d.agg_strict_input_check.nulls;

      Assert(nargs > 0);

      int jumpnull = op->d.agg_strict_input_check.jumpnull;
      x86::Gp v_argsp = EmitLoadConstUIntPtr(Jitcc, "v_argsp.uintptr", args);
      x86::Gp v_nullsp = EmitLoadConstUIntPtr(Jitcc, "v_nullsp.uintptr", nulls);

      /* strict function, check for NULL args */
      for (int argno = 0; argno < nargs; ++argno) {
        x86::Gp v_argisnull = Jitcc.newInt8("v_argisnull.i8");
        if (opcode == EEOP_AGG_STRICT_INPUT_CHECK_NULLS) {
          EmitLoadFromArray(Jitcc, v_nullsp, argno, v_argisnull, sizeof(bool));
        } else {
          x86::Mem m_argnisnull = x86::ptr(v_argsp,
                                           argno * sizeof(NullableDatum) +
                                               offsetof(NullableDatum, isnull),
                                           sizeof(bool));
          Jitcc.mov(v_argisnull, m_argnisnull);
        }

        Jitcc.cmp(v_argisnull, jit::imm(1));
        Jitcc.je(L_opblocks[jumpnull]);
      }

      break;
    }
    case EEOP_AGG_PLAIN_PERGROUP_NULLCHECK: {
      int jumpnull = op->d.agg_plain_pergroup_nullcheck.jumpnull;

      /*
       * pergroup_allaggs = aggstate->all_pergroups
       * [op->d.agg_plain_pergroup_nullcheck.setoff];
       */
      x86::Gp v_aggstatep = emit_load_parent_from_ExprState(Jitcc, v_state);
      x86::Gp v_allpergroupsp =
          emit_load_all_pergroups_from_AggState(Jitcc, v_aggstatep);
      x86::Gp v_pergroup_allaggs =
          Jitcc.newUIntPtr("v_pergroup_allaggs.uintptr");
      EmitLoadFromArray(Jitcc, v_allpergroupsp,
                        op->d.agg_plain_pergroup_nullcheck.setoff,
                        v_pergroup_allaggs, sizeof(Datum));
      Jitcc.cmp(v_pergroup_allaggs, jit::imm(0));
      Jitcc.je(L_opblocks[jumpnull]);
      break;
    }

    case EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYVAL:
    case EEOP_AGG_PLAIN_TRANS_STRICT_BYVAL:
    case EEOP_AGG_PLAIN_TRANS_BYVAL:
    case EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYREF:
    case EEOP_AGG_PLAIN_TRANS_STRICT_BYREF:
    case EEOP_AGG_PLAIN_TRANS_BYREF: {
      AggState *aggstate = castNode(AggState, State->parent);
      AggStatePerTrans pertrans = op->d.agg_trans.pertrans;
      FunctionCallInfo fcinfo = pertrans->transfn_fcinfo;
      x86::Gp v_aggstatep = emit_load_parent_from_ExprState(Jitcc, v_state);
      x86::Gp v_pertransp =
          EmitLoadConstUIntPtr(Jitcc, "v_pertransp.uintptr", pertrans);

      /*
       * pergroup = &aggstate->all_pergroups
       * [op->d.agg_trans.setoff] [op->d.agg_trans.transno];
       */
      int32 setoff = op->d.agg_trans.setoff;
      int32 transno = op->d.agg_trans.transno;
      x86::Gp v_pergroupp = Jitcc.newUIntPtr("v_pergroupp.uintptr");
      x86::Gp v_all_pergroupsp =
          emit_load_all_pergroups_from_AggState(Jitcc, v_aggstatep);
      EmitLoadFromArray(Jitcc, v_all_pergroupsp, setoff, v_pergroupp,
                        sizeof(AggStatePerGroup));
      Jitcc.add(v_pergroupp, jit::imm(transno * sizeof(AggStatePerGroupData)));

      if (opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYVAL ||
          opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYREF) {
        jit::Label L_NoInit = Jitcc.newLabel();
        x86::Gp v_notransvalue =
            emit_load_noTransValue_from_AggStatePerGroupData(Jitcc,
                                                             v_pergroupp);
        Jitcc.cmp(v_notransvalue, jit::imm(1));
        Jitcc.jne(L_NoInit);
        {
          /* init the transition value if necessary */
          x86::Gp v_aggcontext = EmitLoadConstUIntPtr(
              Jitcc, "v_aggcontext.uintptr", op->d.agg_trans.aggcontext);
          jit::InvokeNode *InvokeExecAggInitGroup;
          Jitcc.invoke(
              &InvokeExecAggInitGroup, jit::imm(ExecAggInitGroup),
              jit::FuncSignature::build<void, AggState *, AggStatePerTrans,
                                        AggStatePerGroup, ExprContext *>());
          InvokeExecAggInitGroup->setArg(0, v_aggstatep);
          InvokeExecAggInitGroup->setArg(1, v_pertransp);
          InvokeExecAggInitGroup->setArg(2, v_pergroupp);
          InvokeExecAggInitGroup->setArg(3, v_aggcontext);

          Jitcc.jmp(L_opblocks[opno + 1]);
        }

        Jitcc.bind(L_NoInit);
      }

      if (opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYVAL ||
          opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYREF ||
          opcode == EEOP_AGG_PLAIN_TRANS_STRICT_BYVAL ||
          opcode == EEOP_AGG_PLAIN_TRANS_STRICT_BYREF) {
        x86::Gp v_transnull =
            emit_load_transValueIsNull_from_AggStatePerGroupData(Jitcc,
                                                                 v_pergroupp);
        Jitcc.cmp(v_transnull, jit::imm(1));
        Jitcc.je(L_opblocks[opno + 1]);
      }

      x86::Gp v_fcinfo =
          EmitLoadConstUIntPtr(Jitcc, "v_fcinfo.uintptr", fcinfo);
      x86::Gp v_aggcontext = EmitLoadConstUIntPtr(Jitcc, "v_aggcontext.uintptr",
                                                  op->d.agg_trans.aggcontext);

      /* set aggstate globals */
      {
        /*
         * FIXME: I don't know why v_aggstatep is nullptr in -O2 if we don't
         * load it here. Need to investigate more.
         */
        v_aggstatep = emit_load_parent_from_ExprState(Jitcc, v_state);
      }
      emit_store_curaggcontext_to_AggState(Jitcc, v_aggstatep, v_aggcontext);
      emit_store_current_set_to_AggState(Jitcc, v_aggstatep,
                                         jit::imm(op->d.agg_trans.setno));
      emit_store_curpertrans_to_AggState(Jitcc, v_aggstatep, v_pertransp);

      /* invoke transition function in per-tuple context */
      x86::Gp v_tmpcontext =
          EmitLoadConstUIntPtr(Jitcc, "v_tmpcontext.uintptr",
                               aggstate->tmpcontext->ecxt_per_tuple_memory);
      x86::Gp v_oldcontext = Jitcc.newUIntPtr("v_oldcontext.uintptr");
      jit::InvokeNode *InvokeMemoryContextSwitchTo;
      Jitcc.invoke(&InvokeMemoryContextSwitchTo,
                   jit::imm(MemoryContextSwitchTo),
                   jit::FuncSignature::build<MemoryContext, MemoryContext>());
      InvokeMemoryContextSwitchTo->setArg(0, v_tmpcontext);
      InvokeMemoryContextSwitchTo->setRet(0, v_oldcontext);

      /* store transvalue in fcinfo->args[0] */
      x86::Gp v_transvalue =
          emit_load_transValue_from_AggStatePerGroupData(Jitcc, v_pergroupp);
      x86::Gp v_transnull =
          emit_load_transValueIsNull_from_AggStatePerGroupData(Jitcc,
                                                               v_pergroupp);
      StoreFuncArgValue(Jitcc, v_fcinfo, 0, v_transvalue);
      StoreFuncArgNull(Jitcc, v_fcinfo, 0, v_transnull);
      emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo,
                                                    jit::imm(0));

      x86::Gp v_retval = Jitcc.newUIntPtr("v_retval.uintptr");
      jit::InvokeNode *PGFunc;
      Jitcc.invoke(&PGFunc, jit::imm(fcinfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->setArg(0, v_fcinfo);
      PGFunc->setRet(0, v_retval);
      x86::Gp v_fcinfo_isnull =
          emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);

      /*
       * For pass-by-ref datatype, must copy the new value into
       * aggcontext and free the prior transValue.  But if
       * transfn returned a pointer to its first input, we don't
       * need to do anything.  Also, if transfn returned a
       * pointer to a R/W expanded object that is already a
       * child of the aggcontext, assume we can adopt that value
       * without copying it.
       */
      if (opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYREF ||
          opcode == EEOP_AGG_PLAIN_TRANS_STRICT_BYREF ||
          opcode == EEOP_AGG_PLAIN_TRANS_BYREF) {
        jit::Label L_NoCall = Jitcc.newLabel();
        x86::Gp v_transvalue =
            emit_load_transValue_from_AggStatePerGroupData(Jitcc, v_pergroupp);
        x86::Gp v_transnull =
            emit_load_transValueIsNull_from_AggStatePerGroupData(Jitcc,
                                                                 v_pergroupp);
        Jitcc.cmp(v_transvalue, v_retval);
        Jitcc.je(L_NoCall);

        /* store trans value */
        {
          /*
           * FIXME: It's seems v_transvalue is not properly loaded in -O3 and I
           * don't know why.
           */
          v_transvalue = emit_load_transValue_from_AggStatePerGroupData(
              Jitcc, v_pergroupp);
          v_transnull = emit_load_transValueIsNull_from_AggStatePerGroupData(
              Jitcc, v_pergroupp);
        }

        jit::InvokeNode *InvokeExecAggCopyTransValue;
        x86::Gp v_newval = Jitcc.newUIntPtr("v_newval.uintptr");
        Jitcc.invoke(
            &InvokeExecAggCopyTransValue, jit::imm(ExecAggCopyTransValue),
            jit::FuncSignature::build<Datum, AggState *, AggStatePerTrans,
                                      Datum, bool, Datum, bool>());
        InvokeExecAggCopyTransValue->setArg(0, v_aggstatep);
        InvokeExecAggCopyTransValue->setArg(1, v_pertransp);
        InvokeExecAggCopyTransValue->setArg(2, v_retval);
        InvokeExecAggCopyTransValue->setArg(3, v_fcinfo_isnull);
        InvokeExecAggCopyTransValue->setArg(4, v_transvalue);
        InvokeExecAggCopyTransValue->setArg(5, v_transnull);
        InvokeExecAggCopyTransValue->setRet(0, v_newval);

        /* store trans value */
        emit_store_transValue_to_AggStatePerGroupData(Jitcc, v_pergroupp,
                                                      v_newval);
        emit_store_transValueIsNull_to_AggStatePerGroupData(Jitcc, v_pergroupp,
                                                            v_fcinfo_isnull);

        InvokeMemoryContextSwitchTo = nullptr;
        Jitcc.invoke(&InvokeMemoryContextSwitchTo,
                     jit::imm(MemoryContextSwitchTo),
                     jit::FuncSignature::build<MemoryContext, MemoryContext>());
        InvokeMemoryContextSwitchTo->setArg(0, v_oldcontext);

        Jitcc.jmp(L_opblocks[opno + 1]);

        Jitcc.bind(L_NoCall);
      }

      /* store trans value */
      emit_store_transValue_to_AggStatePerGroupData(Jitcc, v_pergroupp,
                                                    v_retval);
      emit_store_transValueIsNull_to_AggStatePerGroupData(Jitcc, v_pergroupp,
                                                          v_fcinfo_isnull);

      InvokeMemoryContextSwitchTo = nullptr;
      Jitcc.invoke(&InvokeMemoryContextSwitchTo,
                   jit::imm(MemoryContextSwitchTo),
                   jit::FuncSignature::build<MemoryContext, MemoryContext>());
      InvokeMemoryContextSwitchTo->setArg(0, v_oldcontext);

      break;
    }
    case EEOP_AGG_PRESORTED_DISTINCT_SINGLE: {
      AggState *aggstate = castNode(AggState, State->parent);
      AggStatePerTrans pertrans = op->d.agg_presorted_distinctcheck.pertrans;
      int jumpdistinct = op->d.agg_presorted_distinctcheck.jumpdistinct;
      x86::Gp v_aggstatep =
          EmitLoadConstUIntPtr(Jitcc, "v_aggstate.uintptr", aggstate);
      x86::Gp v_pertrans =
          EmitLoadConstUIntPtr(Jitcc, "v_pertrans.uintptr", pertrans);
      x86::Gp v_retval = Jitcc.newInt8("v_retval.i8");
      jit::InvokeNode *InvokeExecEvalPreOrderedDistinctSingle;
      Jitcc.invoke(
          &InvokeExecEvalPreOrderedDistinctSingle,
          jit::imm(ExecEvalPreOrderedDistinctSingle),
          jit::FuncSignature::build<bool, AggState *, AggStatePerTrans>());
      InvokeExecEvalPreOrderedDistinctSingle->setArg(0, v_aggstatep);
      InvokeExecEvalPreOrderedDistinctSingle->setArg(1, v_pertrans);
      InvokeExecEvalPreOrderedDistinctSingle->setRet(0, v_retval);

      Jitcc.cmp(v_retval, 1);
      Jitcc.jne(L_opblocks[jumpdistinct]);

      break;
    }
    case EEOP_AGG_PRESORTED_DISTINCT_MULTI: {
      AggState *aggstate = castNode(AggState, State->parent);
      AggStatePerTrans pertrans = op->d.agg_presorted_distinctcheck.pertrans;
      int jumpdistinct = op->d.agg_presorted_distinctcheck.jumpdistinct;
      x86::Gp v_aggstatep =
          EmitLoadConstUIntPtr(Jitcc, "v_aggstate.uintptr", aggstate);
      x86::Gp v_pertrans =
          EmitLoadConstUIntPtr(Jitcc, "v_pertrans.uintptr", pertrans);
      x86::Gp v_retval = Jitcc.newInt8("v_retval.i8");
      jit::InvokeNode *InvokeExecEvalPreOrderedDistinctMulti;
      Jitcc.invoke(
          &InvokeExecEvalPreOrderedDistinctMulti,
          jit::imm(ExecEvalPreOrderedDistinctMulti),
          jit::FuncSignature::build<bool, AggState *, AggStatePerTrans>());
      InvokeExecEvalPreOrderedDistinctMulti->setArg(0, v_aggstatep);
      InvokeExecEvalPreOrderedDistinctMulti->setArg(1, v_pertrans);
      InvokeExecEvalPreOrderedDistinctMulti->setRet(0, v_retval);

      Jitcc.cmp(v_retval, 1);
      Jitcc.jne(L_opblocks[jumpdistinct]);

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
