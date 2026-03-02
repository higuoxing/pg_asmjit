#include "asmjit_common.h"

extern "C" {

static bool JitSessionInitialized = false;
static bool JitArchSupported = false;
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

  jit::Arch arch = Runtime.environment().arch();
  JitArchSupported = (arch == jit::Arch::kX86 || arch == jit::Arch::kX64 ||
                      arch == jit::Arch::kAArch64);
  if (!JitArchSupported)
    elog(LOG, "pg_asmjit: unsupported architecture, JIT disabled");
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

  /* Initialize session and check architecture support */
  JitInitializeSession();
  if (!JitArchSupported)
    return false;

  /* get or create JIT context */
  if (Parent->state->es_jit) {
    Context = (AsmJitContext *)Parent->state->es_jit;
  } else {
    Context = JitCreateContext(Parent->state->es_jit_flags);
    Parent->state->es_jit = &Context->base;
  }

  INSTR_TIME_SET_CURRENT(CodeGenStartTime);

  jit::CodeHolder Code;
  Code.init(Runtime.environment(), Runtime.cpu_features());
  arch::Compiler Jitcc(&Code);

  /*
   * Datum ExprStateEvalFunc(struct ExprState *expression,
   *                         struct ExprContext *econtext,
   *                         bool *isNull);
   */
  jit::FuncNode *JittedFunc = Jitcc.add_func(
      jit::FuncSignature::build<Datum, ExprState *, ExprContext *, bool *>());

  arch::Gp v_state = Jitcc.new_gp_ptr("v_state"),
           v_econtext = Jitcc.new_gp_ptr("v_econtext"),
           v_isnullp = Jitcc.new_gp_ptr("v_isnullp");

  JittedFunc->set_arg(0, v_state);
  JittedFunc->set_arg(1, v_econtext);
  JittedFunc->set_arg(2, v_isnullp);

  jit::Label *L_opblocks =
      (jit::Label *)palloc(State->steps_len * sizeof(jit::Label));
  for (int opno = 0; opno < State->steps_len; ++opno)
    L_opblocks[opno] = Jitcc.new_label();

  for (int opno = 0; opno < State->steps_len; ++opno) {
    ExprEvalStep *op = &State->steps[opno];
    ExprEvalOp opcode = ExecEvalStepOp(State, op);

    Jitcc.bind(L_opblocks[opno]);

#define BuildEvalXFunc2(Func)                                                  \
  do {                                                                         \
    jit::InvokeNode *JitFunc;                                                  \
    Jitcc.invoke(asmjit::Out(JitFunc), JIT_FN_PTR(Jitcc, Func),               \
                 jit::FuncSignature::build<void, ExprState *, ExprEvalStep *>()); \
    JitFunc->set_arg(0, v_state);                                              \
    JitFunc->set_arg(1, jit::imm(op));                                         \
  } while (0);

#define BuildEvalXFunc3(Func)                                                  \
  do {                                                                         \
    jit::InvokeNode *JitFunc;                                                  \
    Jitcc.invoke(asmjit::Out(JitFunc), JIT_FN_PTR(Jitcc, Func),               \
                 jit::FuncSignature::build<void, ExprState *, ExprEvalStep *,  \
                                           ExprContext *>());                  \
    JitFunc->set_arg(0, v_state);                                              \
    JitFunc->set_arg(1, jit::imm(op));                                         \
    JitFunc->set_arg(2, v_econtext);                                           \
  } while (0);

    switch (opcode) {
    case EEOP_DONE_RETURN: {
      /* Load expression->resvalue and expression->resnull */
      arch::Gp v_resvalue = emit_load_resvalue_from_ExprState(Jitcc, v_state),
               v_resnull = emit_load_resnull_from_ExprState(Jitcc, v_state);

      /* *isnull = expression->resnull */
      EmitStoreToArray(Jitcc, v_isnullp, 0, v_resnull, sizeof(bool));

      /* return expression->resvalue */
      Jitcc.ret(v_resvalue);
      Jitcc.end_func();
      break;
    }

    case EEOP_DONE_NO_RETURN: {
      /* Return a zero datum; the caller ignores the return value. */
      arch::Gp v_zero = Jitcc.new_gp_ptr("v_zero");
      EmitZero(Jitcc, v_zero);
      Jitcc.ret(v_zero);
      Jitcc.end_func();
      break;
    }
    case EEOP_INNER_FETCHSOME:
    case EEOP_OUTER_FETCHSOME:
    case EEOP_SCAN_FETCHSOME:
    case EEOP_OLD_FETCHSOME:
    case EEOP_NEW_FETCHSOME: {
      const TupleTableSlotOps *tts_ops =
          op->d.fetch.fixed ? op->d.fetch.kind : nullptr;
      TupleDesc desc = op->d.fetch.known_desc;
      TupleDeformingFunc jit_deform = nullptr;

      /* Step should not have been generated. */
      Assert(tts_ops != &TTSOpsVirtual);

      /* Compute the address of Slot->tts_nvalid */
      arch::Gp v_slot =
          opcode == EEOP_INNER_FETCHSOME
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, v_econtext)
              : (opcode == EEOP_OUTER_FETCHSOME
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  v_econtext)
                     : (opcode == EEOP_OLD_FETCHSOME
                            ? emit_load_ecxt_oldtuple_from_ExprContext(Jitcc,
                                                                       v_econtext)
                            : (opcode == EEOP_NEW_FETCHSOME
                                   ? emit_load_ecxt_newtuple_from_ExprContext(
                                         Jitcc, v_econtext)
                                   : emit_load_ecxt_scantuple_from_ExprContext(
                                         Jitcc, v_econtext))));

      arch::Gp v_nvalid =
          emit_load_tts_nvalid_from_TupleTableSlot(Jitcc, v_slot);

      /*
       * Check if all required attributes are available, or whether deforming is
       * required.
       */
      EmitCondJumpGE(Jitcc, v_nvalid, op->d.fetch.last_var,
                     L_opblocks[opno + 1]);

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
        Jitcc.invoke(asmjit::Out(SlotGetSomeAttrsInt), JIT_FN_PTR(Jitcc, jit_deform),
                     jit::FuncSignature::build<void, TupleTableSlot *>());
        SlotGetSomeAttrsInt->set_arg(0, v_slot);
      } else {
        Jitcc.invoke(asmjit::Out(SlotGetSomeAttrsInt), JIT_FN_PTR(Jitcc, slot_getsomeattrs_int),
                     jit::FuncSignature::build<void, TupleTableSlot *, int>());
        SlotGetSomeAttrsInt->set_arg(0, v_slot);
        SlotGetSomeAttrsInt->set_arg(1, jit::imm(op->d.fetch.last_var));
      }

      break;
    }

    case EEOP_INNER_VAR:
    case EEOP_OUTER_VAR:
    case EEOP_SCAN_VAR:
    case EEOP_OLD_VAR:
    case EEOP_NEW_VAR: {
      arch::Gp v_slot =
          opcode == EEOP_INNER_VAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, v_econtext)
              : (opcode == EEOP_OUTER_VAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  v_econtext)
                     : (opcode == EEOP_OLD_VAR
                            ? emit_load_ecxt_oldtuple_from_ExprContext(Jitcc,
                                                                       v_econtext)
                            : (opcode == EEOP_NEW_VAR
                                   ? emit_load_ecxt_newtuple_from_ExprContext(
                                         Jitcc, v_econtext)
                                   : emit_load_ecxt_scantuple_from_ExprContext(
                                         Jitcc, v_econtext))));

      arch::Gp v_values =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot),
               v_nulls =
                   emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_slot);

      int attnum = op->d.var.attnum;

      arch::Gp v_value = Jitcc.new_gp_ptr(), v_isnull = Jitcc.new_gp32();
      EmitLoadFromArray(Jitcc, v_values, attnum, v_value, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_nulls, attnum, v_isnull, sizeof(bool));

      arch::Gp v_resvaluep = EmitLoadConstUIntPtr(Jitcc, "v_resvaluep",
                                                  op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);

      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_value, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_isnull, sizeof(bool));

      break;
    }
    case EEOP_INNER_SYSVAR:
    case EEOP_OUTER_SYSVAR:
    case EEOP_SCAN_SYSVAR:
    case EEOP_OLD_SYSVAR:
    case EEOP_NEW_SYSVAR: {
      arch::Gp v_slot =
          opcode == EEOP_INNER_SYSVAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, v_econtext)
              : (opcode == EEOP_OUTER_SYSVAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  v_econtext)
                     : (opcode == EEOP_OLD_SYSVAR
                            ? emit_load_ecxt_oldtuple_from_ExprContext(Jitcc,
                                                                       v_econtext)
                            : (opcode == EEOP_NEW_SYSVAR
                                   ? emit_load_ecxt_newtuple_from_ExprContext(
                                         Jitcc, v_econtext)
                                   : emit_load_ecxt_scantuple_from_ExprContext(
                                         Jitcc, v_econtext))));

      jit::InvokeNode *ExecEvalSysVarFunc;
      Jitcc.invoke(asmjit::Out(ExecEvalSysVarFunc),
           JIT_FN_PTR(Jitcc, ExecEvalSysVar),
          jit::FuncSignature::build<void, ExprState *, ExprEvalStep *,
                                    ExprContext *, TupleTableSlot *>());
      ExecEvalSysVarFunc->set_arg(0, v_state);
      ExecEvalSysVarFunc->set_arg(1, jit::imm(op));
      ExecEvalSysVarFunc->set_arg(2, v_econtext);
      ExecEvalSysVarFunc->set_arg(3, v_slot);
      break;
    }

    case EEOP_WHOLEROW: {
      BuildEvalXFunc3(ExecEvalWholeRowVar);
      break;
    }

    case EEOP_ASSIGN_INNER_VAR:
    case EEOP_ASSIGN_OUTER_VAR:
    case EEOP_ASSIGN_SCAN_VAR:
    case EEOP_ASSIGN_OLD_VAR:
    case EEOP_ASSIGN_NEW_VAR: {
      arch::Gp v_slot =
          opcode == EEOP_ASSIGN_INNER_VAR
              ? emit_load_ecxt_innertuple_from_ExprContext(Jitcc, v_econtext)
              : (opcode == EEOP_ASSIGN_OUTER_VAR
                     ? emit_load_ecxt_outertuple_from_ExprContext(Jitcc,
                                                                  v_econtext)
                     : (opcode == EEOP_ASSIGN_OLD_VAR
                            ? emit_load_ecxt_oldtuple_from_ExprContext(Jitcc,
                                                                       v_econtext)
                            : (opcode == EEOP_ASSIGN_NEW_VAR
                                   ? emit_load_ecxt_newtuple_from_ExprContext(
                                         Jitcc, v_econtext)
                                   : emit_load_ecxt_scantuple_from_ExprContext(
                                         Jitcc, v_econtext))));

      arch::Gp v_values =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot),
               v_nulls =
                   emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_slot);

      int attnum = op->d.assign_var.attnum;

      /* Load data. */
      arch::Gp v_value = Jitcc.new_gp_ptr("v_value"),
               v_null = Jitcc.new_gp32("v_null");
      EmitLoadFromArray(Jitcc, v_values, attnum, v_value, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_nulls, attnum, v_null, sizeof(bool));

      /* Save the result. */
      int resultnum = op->d.assign_var.resultnum;
      arch::Gp v_resultslot =
          emit_load_resultslot_from_ExprState(Jitcc, v_state);
      arch::Gp v_rvaluep =
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
      arch::Gp v_rvalue = emit_load_resvalue_from_ExprState(Jitcc, v_state),
               v_risnull = emit_load_resnull_from_ExprState(Jitcc, v_state);

      arch::Gp v_resultslot =
          emit_load_resultslot_from_ExprState(Jitcc, v_state);
      arch::Gp v_tmpvaluep =
                  emit_load_tts_values_from_TupleTableSlot(Jitcc, v_resultslot),
               v_tmpisnullp =
                   emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_resultslot);

      EmitStoreToArray(Jitcc, v_tmpisnullp, resultnum, v_risnull, sizeof(bool));

      if (opcode == EEOP_ASSIGN_TMP_MAKE_RO) {
        EmitCondJumpEQ(Jitcc, v_risnull, 1, L_opblocks[opno + 1]);

        jit::InvokeNode *MakeExpandedObjectReadOnlyInternalFunc;
        Jitcc.invoke(asmjit::Out(MakeExpandedObjectReadOnlyInternalFunc),
                     JIT_FN_PTR(Jitcc, MakeExpandedObjectReadOnlyInternal),
                     jit::FuncSignature::build<Datum, Datum>());
        MakeExpandedObjectReadOnlyInternalFunc->set_arg(0, v_rvalue);
        MakeExpandedObjectReadOnlyInternalFunc->set_ret(0, v_rvalue);
      }

      /* Finally, store the result. */
      EmitStoreToArray(Jitcc, v_tmpvaluep, resultnum, v_rvalue, sizeof(Datum));
      break;
    }
    case EEOP_CONST: {
      arch::Gp v_constvalue = EmitLoadConstUInt64(Jitcc, "constval.value",
                                                  op->d.constval.value),
               v_constnull = EmitLoadConstUInt8(Jitcc, "constval.isnull",
                                                op->d.constval.isnull);

      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "op.resvalue", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "op.resnull", op->resnull);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_constvalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_constnull, sizeof(bool));

      break;
    }
    case EEOP_FUNCEXPR:
    case EEOP_FUNCEXPR_STRICT:
    case EEOP_FUNCEXPR_STRICT_1:
    case EEOP_FUNCEXPR_STRICT_2: {
      FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
      arch::Gp v_fcinfo = EmitLoadConstUIntPtr(Jitcc, "v_fcinfo", fcinfo);

      jit::Label L_InvokePGFunc = Jitcc.new_label();

      if (opcode == EEOP_FUNCEXPR_STRICT ||
          opcode == EEOP_FUNCEXPR_STRICT_1 ||
          opcode == EEOP_FUNCEXPR_STRICT_2) {
        jit::Label L_StrictFail = Jitcc.new_label();
        /* Should make sure that they're optimized beforehand. */
        int argnum = op->d.func.nargs;
        if (argnum == 0) {
          ereport(ERROR,
                  (errmsg("Argumentless strict functions are pointless")));
        }

        /* Check for NULL args for strict function. */
        for (int argno = 0; argno < argnum; ++argno) {
          arch::Gp v_argisnull = LoadFuncArgNull(Jitcc, v_fcinfo, argno);
          EmitCondJumpEQ(Jitcc, v_argisnull, 1, L_StrictFail);
        }

        EmitJump(Jitcc, L_InvokePGFunc);

        Jitcc.bind(L_StrictFail);
        /* Op->resnull = true */
        arch::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
        EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
        EmitJump(Jitcc, L_opblocks[opno + 1]);
      }

      /*
       * Before invoking PGFuncs, we should set FuncCallInfo->isnull to false.
       */
      Jitcc.bind(L_InvokePGFunc);
      emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo,
                                                    jit::imm(0));

      jit::InvokeNode *PGFunc;
      arch::Gp v_retval = Jitcc.new_gp_ptr("v_retval");
      Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->set_arg(0, v_fcinfo);
      PGFunc->set_ret(0, v_retval);

      /* Write result values. */
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);

      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
      arch::Gp v_fcinfo_isnull =
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
      arch::Gp v_boolanynullp = EmitLoadConstUIntPtr(
          Jitcc, "op.d.boolexpr.anynull", op->d.boolexpr.anynull);
      jit::Label L_BoolCheckFalse = Jitcc.new_label(),
                 L_BoolCont = Jitcc.new_label();

      if (opcode == EEOP_BOOL_AND_STEP_FIRST)
        EmitStoreToArray(Jitcc, v_boolanynullp, 0, jit::imm(0), sizeof(bool));

      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_boolvalue = Jitcc.new_gp_ptr("v_boolvalue"),
               v_boolnull = Jitcc.new_gp32("v_boolnull");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_boolvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_boolnull, sizeof(bool));

      /* check if current input is NULL */
      EmitCondJumpNE(Jitcc, v_boolnull, 1, L_BoolCheckFalse);
      {
        /* b_boolisnull */
        /* set boolanynull to true */
        EmitStoreToArray(Jitcc, v_boolanynullp, 0, jit::imm(1), sizeof(bool));
        EmitJump(Jitcc, L_BoolCont);
      }

      Jitcc.bind(L_BoolCheckFalse);
      {
        EmitCondJumpNE(Jitcc, v_boolvalue, 0, L_BoolCont);

        /* b_boolisfalse */
        EmitJump(Jitcc, L_opblocks[op->d.boolexpr.jumpdone]);
      }

      Jitcc.bind(L_BoolCont);
      {
        arch::Gp v_boolanynull = Jitcc.new_gp32("v_boolanynull");
        EmitLoadFromArray(Jitcc, v_boolanynullp, 0, v_boolanynull,
                          sizeof(bool));
        EmitCondJumpEQ(Jitcc, v_boolanynull, 0, L_opblocks[opno + 1]);
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
      arch::Gp v_boolanynullp = EmitLoadConstUIntPtr(
          Jitcc, "v_boolanynullp", op->d.boolexpr.anynull);
      jit::Label L_BoolCheckTrue = Jitcc.new_label(),
                 L_BoolCont = Jitcc.new_label();

      if (opcode == EEOP_BOOL_OR_STEP_FIRST)
        EmitStoreToArray(Jitcc, v_boolanynullp, 0, jit::imm(0), sizeof(bool));

      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_boolvalue = Jitcc.new_gp_ptr("v_boolvalue"),
               v_boolnull = Jitcc.new_gp32("v_boolnull");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_boolvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_boolnull, sizeof(bool));

      /* check if current input is NULL */
      EmitCondJumpNE(Jitcc, v_boolnull, 1, L_BoolCheckTrue);
      {
        /* b_boolisnull */
        /* set boolanynull to true */
        EmitStoreToArray(Jitcc, v_boolanynullp, 0, jit::imm(1), sizeof(bool));
        EmitJump(Jitcc, L_BoolCont);
      }

      Jitcc.bind(L_BoolCheckTrue);
      {
        EmitCondJumpNE(Jitcc, v_boolvalue, 1, L_BoolCont);

        /* b_boolistrue */
        EmitJump(Jitcc, L_opblocks[op->d.boolexpr.jumpdone]);
      }

      Jitcc.bind(L_BoolCont);
      {
        arch::Gp v_boolanynull = Jitcc.new_gp32("v_boolanynull");
        EmitLoadFromArray(Jitcc, v_boolanynullp, 0, v_boolanynull,
                          sizeof(bool));
        EmitCondJumpEQ(Jitcc, v_boolanynull, 0, L_opblocks[opno + 1]);
      }

      /* set resnull to true */
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
      /* reset resvalue */
      EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
      break;
    }

    case EEOP_BOOL_NOT_STEP: {
      arch::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue);
      arch::Gp v_boolvalue = Jitcc.new_gp_ptr("v_boolvalue");
      arch::Gp v_negbool = Jitcc.new_gp_ptr("v_negbool");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_boolvalue, sizeof(Datum));
      EmitSetEQ(Jitcc, v_negbool, v_boolvalue, 0);

      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_negbool, sizeof(Datum));
      break;
    }

    case EEOP_QUAL: {
      jit::Label L_HandleNullOrFalse = Jitcc.new_label();

      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resvalue = Jitcc.new_gp_ptr("v_resvalue"),
               v_resnull = Jitcc.new_gp32("v_resnull");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      EmitCondJumpEQ(Jitcc, v_resnull, 1, L_HandleNullOrFalse);
      EmitCondJumpEQ(Jitcc, v_resvalue, 0, L_HandleNullOrFalse);

      EmitJump(Jitcc, L_opblocks[opno + 1]);

      /* Handling null or false. */
      Jitcc.bind(L_HandleNullOrFalse);

      /* Set resnull and resvalue to false. */
      EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

      EmitJump(Jitcc, L_opblocks[op->d.qualexpr.jumpdone]);

      break;
    }

    case EEOP_JUMP: {
      EmitJump(Jitcc, L_opblocks[op->d.jump.jumpdone]);
      break;
    }

    case EEOP_JUMP_IF_NULL: {
      /* Transfer control if current result is null */
      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resnull = Jitcc.new_gp32("v_resnull");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
      EmitCondJumpEQ(Jitcc, v_resnull, 1, L_opblocks[op->d.jump.jumpdone]);

      break;
    }

    case EEOP_JUMP_IF_NOT_NULL: {
      /* Transfer control if current result is non-null */
      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resnull = Jitcc.new_gp32("v_resnull");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
      EmitCondJumpEQ(Jitcc, v_resnull, 0, L_opblocks[op->d.jump.jumpdone]);

      break;
    }

    case EEOP_JUMP_IF_NOT_TRUE: {
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resvalue = Jitcc.new_gp_ptr("v_resvalue"),
               v_resnull = Jitcc.new_gp32("v_resnull");

      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      /* Transfer control if current result is null or false */
      EmitCondJumpEQ(Jitcc, v_resnull, 1, L_opblocks[op->d.jump.jumpdone]);
      EmitCondJumpEQ(Jitcc, v_resvalue, 0, L_opblocks[op->d.jump.jumpdone]);

      break;
    }

    case EEOP_NULLTEST_ISNULL: {
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resvalue = Jitcc.new_gp_ptr("v_resvalue"),
               v_resnull = Jitcc.new_gp32("v_resnull");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
      EmitSetEQ(Jitcc, v_resvalue, v_resnull, 1);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

      break;
    }

    case EEOP_NULLTEST_ISNOTNULL: {
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resvalue = Jitcc.new_gp_ptr("v_resvalue"),
               v_resnull = Jitcc.new_gp32("v_resnull");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
      EmitSetEQ(Jitcc, v_resvalue, v_resnull, 0);

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
      jit::Label L_NotNull = Jitcc.new_label();
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resnull = Jitcc.new_gp32("resnull");

      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
      EmitCondJumpNE(Jitcc, v_resnull, 1, L_NotNull);
      /* result is null. */
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
      EmitStoreToArray(
          Jitcc, v_resvaluep, 0,
          (opcode == EEOP_BOOLTEST_IS_TRUE || opcode == EEOP_BOOLTEST_IS_FALSE)
              ? jit::imm(0)
              : jit::imm(1),
          sizeof(Datum));
      EmitJump(Jitcc, L_opblocks[opno + 1]);

      Jitcc.bind(L_NotNull);
      if (opcode == EEOP_BOOLTEST_IS_TRUE ||
          opcode == EEOP_BOOLTEST_IS_NOT_FALSE) {
        /*
         * if value is not null NULL, return value (already
         * set)
         */
      } else {
        arch::Gp v_resvalue = Jitcc.new_gp_ptr("v_resvalue");
        arch::Gp v_resvalue_is_false = Jitcc.new_gp_ptr("v_resvalue_is_false");
        EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
        EmitSetEQ(Jitcc, v_resvalue_is_false, v_resvalue, 0);
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
      arch::Gp v_retval = Jitcc.new_gp32("ret");
      Jitcc.invoke(asmjit::Out(InvokeSubscriptFunc),
           JIT_FN_PTR(Jitcc, op->d.sbsref_subscript.subscriptfunc),
          jit::FuncSignature::build<bool, ExprState *, struct ExprEvalStep *,
                                    ExprContext *>());
      InvokeSubscriptFunc->set_arg(0, v_state);
      InvokeSubscriptFunc->set_arg(1, jit::imm(op));
      InvokeSubscriptFunc->set_arg(2, v_econtext);
      InvokeSubscriptFunc->set_ret(0, v_retval);

      EmitCondJumpEQ(Jitcc, v_retval, 0,
                     L_opblocks[op->d.sbsref_subscript.jumpdone]);
      break;
    }

    case EEOP_SBSREF_OLD:
    case EEOP_SBSREF_ASSIGN:
    case EEOP_SBSREF_FETCH: {
      BuildEvalXFunc3(op->d.sbsref.subscriptfunc);
      break;
    }

    case EEOP_CASE_TESTVAL: {
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      if (op->d.casetest.value) {
        arch::Gp v_casevaluep = EmitLoadConstUIntPtr(
                    Jitcc, "v_casevaluep", op->d.casetest.value),
                 v_casenullp = EmitLoadConstUIntPtr(Jitcc, "v_casenullp",
                                                    op->d.casetest.isnull);
        arch::Gp v_casevalue = Jitcc.new_gp_ptr("v_casevalue"),
                 v_casenull = Jitcc.new_gp32("v_casenull");
        EmitLoadFromArray(Jitcc, v_casevaluep, 0, v_casevalue, sizeof(Datum));
        EmitLoadFromArray(Jitcc, v_casenullp, 0, v_casenull, sizeof(bool));

        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
      } else {
        arch::Gp v_casevalue =
            emit_load_caseValue_datum_from_ExprContext(Jitcc, v_econtext);
        arch::Gp v_casenull =
            emit_load_caseValue_isNull_from_ExprContext(Jitcc, v_econtext);

        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
      }
      break;
    }

    case EEOP_CASE_TESTVAL_EXT: {
      /* Always reads from econtext->caseValue_datum/isNull (no pointer). */
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_casevalue =
          emit_load_caseValue_datum_from_ExprContext(Jitcc, v_econtext);
      arch::Gp v_casenull =
          emit_load_caseValue_isNull_from_ExprContext(Jitcc, v_econtext);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
      break;
    }
    case EEOP_MAKE_READONLY: {
      arch::Gp v_nullp =
          EmitLoadConstUIntPtr(Jitcc, "v_nullp", op->d.make_readonly.isnull);
      arch::Gp v_null = Jitcc.new_gp32("v_null");

      EmitLoadFromArray(Jitcc, v_nullp, 0, v_null, sizeof(bool));

      /* store null isnull value in result */
      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_null, sizeof(bool));

      EmitCondJumpEQ(Jitcc, v_null, 1, L_opblocks[opno + 1]);

      /* if value is not null, convert to RO datum */
      arch::Gp v_valuep =
          EmitLoadConstUIntPtr(Jitcc, "v_valuep", op->d.make_readonly.value);
      arch::Gp v_value = Jitcc.new_gp_ptr("v_value");
      EmitLoadFromArray(Jitcc, v_valuep, 0, v_value, sizeof(Datum));
      jit::InvokeNode *InvokeMakeExpandedObjectReadOnly;
      Jitcc.invoke(asmjit::Out(InvokeMakeExpandedObjectReadOnly),
                   JIT_FN_PTR(Jitcc, MakeExpandedObjectReadOnlyInternal),
                   jit::FuncSignature::build<Datum, Datum>());
      InvokeMakeExpandedObjectReadOnly->set_arg(0, v_value);
      InvokeMakeExpandedObjectReadOnly->set_ret(0, v_value);

      arch::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_value, sizeof(Datum));
      break;
    }

    case EEOP_IOCOERCE: {
      FunctionCallInfo fcinfo_out = op->d.iocoerce.fcinfo_data_out,
                       fcinfo_in = op->d.iocoerce.fcinfo_data_in;
      jit::Label L_SkipOutputCall = Jitcc.new_label(),
                 L_InputCall = Jitcc.new_label();

      arch::Gp v_fcinfo_out =
                  EmitLoadConstUIntPtr(Jitcc, "v_fcinfo_out", fcinfo_out),
               v_fcinfo_in =
                   EmitLoadConstUIntPtr(Jitcc, "v_fcinfo_in", fcinfo_in);
      arch::Gp v_output = Jitcc.new_gp64("v_output");

      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp", op->resnull);
      arch::Gp v_resnull = Jitcc.new_gp32("op.resnull");
      EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));

      EmitCondJumpEQ(Jitcc, v_resnull, 1, L_SkipOutputCall);
      {
        /* Not null, call output. */
        arch::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvalue", op->resvalue);
        arch::Gp v_resvalue = Jitcc.new_gp_ptr("v_resvalue");
        EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_resvalue, sizeof(Datum));
        StoreFuncArgValue(Jitcc, v_fcinfo_out, 0, v_resvalue);
        StoreFuncArgNull(Jitcc, v_fcinfo_out, 0, jit::imm(0));
        emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo_out,
                                                      jit::imm(0));

        jit::InvokeNode *PGFunc;
        Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo_out->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->set_arg(0, v_fcinfo_out);
        PGFunc->set_ret(0, v_output);
        EmitJump(Jitcc, L_InputCall);
      }

      Jitcc.bind(L_SkipOutputCall);
      Jitcc.mov(v_output, jit::imm(0));

      Jitcc.bind(L_InputCall);
      {
        if (op->d.iocoerce.finfo_in->fn_strict) {
          EmitCondJumpEQ(Jitcc, v_output, 0, L_opblocks[opno + 1]);
        }
        EmitLoadFromArray(Jitcc, v_resnullp, 0, v_resnull, sizeof(bool));
        /* Call input function. */
        StoreFuncArgValue(Jitcc, v_fcinfo_in, 0, v_output);
        StoreFuncArgNull(Jitcc, v_fcinfo_in, 0, v_resnull);
        emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo_in,
                                                      jit::imm(0));
        jit::InvokeNode *PGFunc;
        Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo_in->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->set_arg(0, v_fcinfo_in);
        PGFunc->set_ret(0, v_output);

        arch::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvalue", op->resvalue);
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
      jit::Label L_NoArgIsNull = Jitcc.new_label(),
                 L_AnyArgIsNull = Jitcc.new_label();
      arch::Gp v_fcinfo = EmitLoadConstUIntPtr(Jitcc, "v_fcinfo", fcinfo);
      /* load args[0|1].isnull for both arguments */
      arch::Gp v_argnull0 = LoadFuncArgNull(Jitcc, v_fcinfo, 0),
               v_argnull1 = LoadFuncArgNull(Jitcc, v_fcinfo, 1);
      arch::Gp v_anyargisnull = Jitcc.new_gp32("v_anyargisnull");
      Jitcc.mov(v_anyargisnull, v_argnull0);
      EmitBitwiseOr(Jitcc, v_anyargisnull, v_argnull1);

      EmitCondJumpEQ(Jitcc, v_anyargisnull, 0, L_NoArgIsNull);
      {
        /* check both arguments */
        arch::Gp v_bothargisnull = Jitcc.new_gp32("v_bothargisnull");
        Jitcc.mov(v_bothargisnull, v_argnull0);
        EmitBitwiseAnd(Jitcc, v_bothargisnull, v_argnull1);
        EmitCondJumpNE(Jitcc, v_bothargisnull, 1, L_AnyArgIsNull);
        {
          arch::Gp v_resnullp =
              EmitLoadConstUIntPtr(Jitcc, "op.resnullp", op->resnull);
          arch::Gp v_resvaluep =
              EmitLoadConstUIntPtr(Jitcc, "op.resvaluep", op->resvalue);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
          if (opcode == EEOP_NOT_DISTINCT)
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(1), sizeof(Datum));
          else
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          EmitJump(Jitcc, L_opblocks[opno + 1]);
        }

        Jitcc.bind(L_AnyArgIsNull);
        {
          arch::Gp v_resnullp =
              EmitLoadConstUIntPtr(Jitcc, "op.resnullp", op->resnull);
          arch::Gp v_resvaluep =
              EmitLoadConstUIntPtr(Jitcc, "op.resvaluep", op->resvalue);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));
          if (opcode == EEOP_NOT_DISTINCT)
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          else
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(1), sizeof(Datum));
          EmitJump(Jitcc, L_opblocks[opno + 1]);
        }
      }

      Jitcc.bind(L_NoArgIsNull);
      {
        arch::Gp v_retval = Jitcc.new_gp64("v_retval");
        jit::InvokeNode *PGFunc;
        Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->set_arg(0, v_fcinfo);
        PGFunc->set_ret(0, v_retval);
        arch::Gp v_fcinfo_isnull =
            emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);
        if (opcode == EEOP_DISTINCT) {
          /* Must invert the result of "=" */
          arch::Gp v_tmpretval = Jitcc.new_gp64("v_tmpretval");
          Jitcc.mov(v_tmpretval, v_retval);
          EmitSetEQ(Jitcc, v_retval, v_tmpretval, 0);
        }
        arch::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp", op->resnull);
        arch::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep", op->resvalue);

        EmitStoreToArray(Jitcc, v_resnullp, 0, v_fcinfo_isnull, sizeof(bool));
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
      }
      break;
    }

    case EEOP_NULLIF: {
      FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
      jit::Label L_NonNull = Jitcc.new_label(), L_HasNull = Jitcc.new_label();
      arch::Gp v_fcinfo = EmitLoadConstUIntPtr(Jitcc, "v_fcinfo", fcinfo);

      /* if either argument is NULL they can't be equal */
      arch::Gp v_argnull0 = LoadFuncArgNull(Jitcc, v_fcinfo, 0);
      arch::Gp v_argnull1 = LoadFuncArgNull(Jitcc, v_fcinfo, 1);
      arch::Gp v_anyargisnull = Jitcc.new_gp32("v_anyargisnull");
      Jitcc.mov(v_anyargisnull, v_argnull0);
      EmitBitwiseOr(Jitcc, v_anyargisnull, v_argnull1);

      EmitCondJumpNE(Jitcc, v_anyargisnull, 1, L_NonNull);
      Jitcc.bind(L_HasNull);
      {
        arch::Gp v_arg0 = LoadFuncArgValue(Jitcc, v_fcinfo, 0);
        arch::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp", op->resnull);
        arch::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep", op->resvalue);
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_argnull0, sizeof(bool));
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_arg0, sizeof(Datum));
        EmitJump(Jitcc, L_opblocks[opno + 1]);
      }

      Jitcc.bind(L_NonNull);
      {
        arch::Gp v_retval = Jitcc.new_gp64("v_retval");
        jit::InvokeNode *PGFunc;
        Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->set_arg(0, v_fcinfo);
        PGFunc->set_ret(0, v_retval);
        arch::Gp v_fcinfo_isnull =
            emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);

        /*
         * If result not null, and arguments are equal return null
         * (same result as if there'd been NULLs, hence reuse
         * b_hasnull).
         */
        arch::Gp v_argsequal = Jitcc.new_gp64("v_argsequal");
        EmitSetEQ(Jitcc, v_argsequal, v_fcinfo_isnull, 0);
        EmitBitwiseAnd(Jitcc, v_argsequal, v_retval);

        EmitCondJumpNE(Jitcc, v_argsequal, 1, L_HasNull);

        /* build block setting result to NULL, if args are equal */
        arch::Gp v_resnullp =
            EmitLoadConstUIntPtr(Jitcc, "op.resnullp", op->resnull);
        arch::Gp v_resvaluep =
            EmitLoadConstUIntPtr(Jitcc, "op.resvaluep", op->resvalue);
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

    case EEOP_RETURNINGEXPR: {
      /*
       * If the OLD/NEW row doesn't exist (flagged by ExprState->flags &
       * nullflag), store a NULL result and jump to jumpdone; otherwise
       * fall through to the next op to evaluate the expression normally.
       */
      arch::Gp v_flags =
          emit_load_flags_from_ExprState(Jitcc, v_state);
      EmitBitwiseAndImm(Jitcc, v_flags,
                        (int64_t)(uint64_t)op->d.returningexpr.nullflag);

      jit::Label L_notnull = Jitcc.new_label();
      EmitCondJumpEQ(Jitcc, v_flags, 0, L_notnull);

      /* OLD/NEW row is NULL: write NULL result then jump to jumpdone. */
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_zero = Jitcc.new_gp_ptr("v_zero");
      arch::Gp v_true = EmitLoadConstInt32(Jitcc, "v_true", 1);
      EmitZero(Jitcc, v_zero);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_zero, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_true, sizeof(bool));
      EmitJump(Jitcc, L_opblocks[op->d.returningexpr.jumpdone]);

      Jitcc.bind(L_notnull);
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
      arch::Gp v_fcinfo = EmitLoadConstUIntPtr(Jitcc, "v_fcinfo", fcinfo);
      jit::Label L_Null = Jitcc.new_label();
      /*
       * If function is strict, and either arg is null, we're
       * done.
       */
      if (op->d.rowcompare_step.finfo->fn_strict) {
        arch::Gp v_argnull0 = LoadFuncArgNull(Jitcc, v_fcinfo, 0);
        arch::Gp v_argnull1 = LoadFuncArgNull(Jitcc, v_fcinfo, 1);
        arch::Gp v_anyargisnull = Jitcc.new_gp32("v_anyargisnull");
        Jitcc.mov(v_anyargisnull, v_argnull0);
        EmitBitwiseOr(Jitcc, v_anyargisnull, v_argnull1);
        EmitCondJumpEQ(Jitcc, v_anyargisnull, 1, L_Null);
      }

      arch::Gp v_retval = Jitcc.new_gp64("v_retval");
      jit::InvokeNode *PGFunc;
      Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->set_arg(0, v_fcinfo);
      PGFunc->set_ret(0, v_retval);
      arch::Gp v_fcinfo_isnull =
          emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);

      arch::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "op.resvaluep", op->resvalue);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
      /* if result of function is NULL, force NULL result */
      EmitCondJumpNE(Jitcc, v_fcinfo_isnull, 0, L_Null);
      /* if results equal, compare next, otherwise done */
      EmitCondJumpEQ(Jitcc, v_retval, 0, L_opblocks[opno + 1]);
      EmitJump(Jitcc, L_opblocks[op->d.rowcompare_step.jumpdone]);

      Jitcc.bind(L_Null);
      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp", op->resnull);
      EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
      EmitJump(Jitcc, L_opblocks[op->d.rowcompare_step.jumpnull]);
      break;
    }

    case EEOP_ROWCOMPARE_FINAL: {
      CompareType rctype = op->d.rowcompare_final.cmptype;

      /*
       * Btree comparators return 32 bit results, need to be
       * careful about sign (used as a 64 bit value it's
       * otherwise wrong).
       */
      arch::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "op.resvaluep", op->resvalue);
      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "op.resnullp", op->resnull);
      arch::Gp v_cmpop = Jitcc.new_gp32("v_cmpop");
      EmitLoadFromArray(Jitcc, v_resvaluep, 0, v_cmpop, sizeof(int32));
      arch::Gp v_cmpresult = Jitcc.new_gp64("v_cmpresult");

      switch (rctype) {
      case COMPARE_LT:
        EmitSetLT(Jitcc, v_cmpresult, v_cmpop);
        break;
      case COMPARE_LE:
        EmitSetLE(Jitcc, v_cmpresult, v_cmpop);
        break;
      case COMPARE_GT:
        EmitSetGT(Jitcc, v_cmpresult, v_cmpop);
        break;
      case COMPARE_GE:
        EmitSetGE(Jitcc, v_cmpresult, v_cmpop);
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
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      if (op->d.casetest.value) {
        arch::Gp v_casevaluep = EmitLoadConstUIntPtr(
                    Jitcc, "v_casevaluep", op->d.casetest.value),
                 v_casenullp = EmitLoadConstUIntPtr(Jitcc, "v_casenullp",
                                                    op->d.casetest.isnull);
        arch::Gp v_casevalue = Jitcc.new_gp_ptr("v_casevalue"),
                 v_casenull = Jitcc.new_gp32("v_casenull");
        EmitLoadFromArray(Jitcc, v_casevaluep, 0, v_casevalue, sizeof(Datum));
        EmitLoadFromArray(Jitcc, v_casenullp, 0, v_casenull, sizeof(bool));

        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
      } else {
        arch::Gp v_casevalue =
            emit_load_domainValue_datum_from_ExprContext(Jitcc, v_econtext);
        arch::Gp v_casenull =
            emit_load_domainValue_isNull_from_ExprContext(Jitcc, v_econtext);

        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
      }
      break;
    }

    case EEOP_DOMAIN_TESTVAL_EXT: {
      /* Always reads from econtext->domainValue_datum/isNull (no pointer). */
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_casevalue =
          emit_load_domainValue_datum_from_ExprContext(Jitcc, v_econtext);
      arch::Gp v_casenull =
          emit_load_domainValue_isNull_from_ExprContext(Jitcc, v_econtext);
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_casevalue, sizeof(Datum));
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_casenull, sizeof(bool));
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
      arch::Gp v_resvaluep =
                  EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
               v_resnullp =
                   EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
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
      jit::Label L_IfNull = Jitcc.new_label();
      FunctionCallInfo fcinfo = op->d.hashdatum.fcinfo_data;
      arch::Gp v_prevhash = Jitcc.new_gp_ptr("prevhash");
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
         * Read the intermediate hash result from op->d.hashdatum.iresult->value,
         * which points to the FIRST step's NullableDatum (not this step's resvalue).
         */
        arch::Gp v_iresultp =
            EmitLoadConstUIntPtr(Jitcc, "v_iresultp", &op->d.hashdatum.iresult->value);
        EmitLoadFromArray(Jitcc, v_iresultp, 0, v_prevhash, sizeof(Datum));

        /*
         * Rotate bits left by 1 bit.  Be careful not to
         * overflow uint32 when working with size_t.
         */
        arch::Gp v_tmp = Jitcc.new_gp64("v_tmp");
        Jitcc.mov(v_tmp, v_prevhash);
        EmitShlImm(Jitcc, v_tmp, 1);
        EmitBitwiseAndImm(Jitcc, v_tmp, 0xffffffff);
        EmitShrImm(Jitcc, v_prevhash, 31);
        EmitBitwiseOr(Jitcc, v_prevhash, v_tmp);
      }

      /* We expect the hash function to have 1 argument */
      if (fcinfo->nargs != 1)
        ereport(ERROR, (errmsg("incorrect number of function arguments")));

      arch::Gp v_fcinfo = EmitLoadConstUIntPtr(Jitcc, "fcinfo", fcinfo);
      /* emit code to check if the input parameter is NULL */
      arch::Gp v_argisnull = LoadFuncArgNull(Jitcc, v_fcinfo, 0);
      EmitCondJumpEQ(Jitcc, v_argisnull, 1, L_IfNull);
      {
        /* If not null. */
        if (opcode == EEOP_HASHDATUM_NEXT32_STRICT) {
          /*
           * Read the intermediate hash result from op->d.hashdatum.iresult->value.
           */
          arch::Gp v_iresultp =
              EmitLoadConstUIntPtr(Jitcc, "v_iresultp", &op->d.hashdatum.iresult->value);
          EmitLoadFromArray(Jitcc, v_iresultp, 0, v_prevhash, sizeof(Datum));

          arch::Gp v_tmp = Jitcc.new_gp64("v_tmp");
          Jitcc.mov(v_tmp, v_prevhash);
          EmitShlImm(Jitcc, v_tmp, 1);
          EmitBitwiseAndImm(Jitcc, v_tmp, 0xffffffff);
          EmitShrImm(Jitcc, v_prevhash, 31);
          EmitBitwiseOr(Jitcc, v_prevhash, v_tmp);
        }

        /* call the hash function */
        arch::Gp v_retval = Jitcc.new_gp64("v_retval");
        jit::InvokeNode *PGFunc;
        Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo->flinfo->fn_addr),
                     jit::FuncSignature::build<Datum, FunctionCallInfo>());
        PGFunc->set_arg(0, v_fcinfo);
        PGFunc->set_ret(0, v_retval);
        /*
         * For NEXT32 ops, XOR (^) the returned hash value with
         * the existing hash value.
         */
        if (opcode == EEOP_HASHDATUM_NEXT32 ||
            opcode == EEOP_HASHDATUM_NEXT32_STRICT)
          EmitBitwiseXor(Jitcc, v_retval, v_prevhash);

        arch::Gp v_resvaluep =
                    EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
                 v_resnullp =
                     EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
        EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));
        EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

        EmitJump(Jitcc, L_opblocks[opno + 1]);
      }

      Jitcc.bind(L_IfNull);
      {
        if (opcode == EEOP_HASHDATUM_FIRST_STRICT ||
            opcode == EEOP_HASHDATUM_NEXT32_STRICT) {
          /*
           * In strict node, NULL inputs result in NULL.  Save
           * the NULL result and goto jumpdone.
           */
          arch::Gp v_resvaluep =
                      EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
                   v_resnullp =
                       EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(1), sizeof(bool));
          EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));

          EmitJump(Jitcc, L_opblocks[op->d.hashdatum.jumpdone]);
        } else {
          arch::Gp v_resvaluep =
                      EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue),
                   v_resnullp =
                       EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
          EmitStoreToArray(Jitcc, v_resnullp, 0, jit::imm(0), sizeof(bool));

          if (opcode == EEOP_HASHDATUM_NEXT32) {
            /* Assert(v_prevhash != NULL) */
            EmitStoreToArray(Jitcc, v_resvaluep, 0, v_prevhash, sizeof(Datum));
          } else {
            Assert(opcode == EEOP_HASHDATUM_FIRST);
            EmitStoreToArray(Jitcc, v_resvaluep, 0, jit::imm(0), sizeof(Datum));
          }

          EmitJump(Jitcc, L_opblocks[opno + 1]);
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
      arch::Gp v_ret = Jitcc.new_gp32("v_ret");
      /*
       * Call ExecEvalJsonExprPath().  It returns the address of
       * the step to perform next.
       */
      jit::InvokeNode *InvokeExecEvalJsonExprPath;
      Jitcc.invoke(asmjit::Out(InvokeExecEvalJsonExprPath), JIT_FN_PTR(Jitcc, ExecEvalJsonExprPath),
                   jit::FuncSignature::build<int, ExprState *, ExprEvalStep *,
                                             ExprContext *>());
      InvokeExecEvalJsonExprPath->set_arg(0, v_state);
      InvokeExecEvalJsonExprPath->set_arg(1, jit::imm(op));
      InvokeExecEvalJsonExprPath->set_arg(2, v_econtext);
      InvokeExecEvalJsonExprPath->set_ret(0, v_ret);

      /*
       * Build a switch to map the return value (v_ret above),
       * which is a runtime value of the step address to perform
       * next, to either jump_empty, jump_error,
       * jump_eval_coercion, or jump_end.
       */
      if (jsestate->jump_empty >= 0) {
        EmitCondJumpEQ(Jitcc, v_ret, jsestate->jump_empty,
                       L_opblocks[jsestate->jump_empty]);
      }

      if (jsestate->jump_error >= 0) {
        EmitCondJumpEQ(Jitcc, v_ret, jsestate->jump_error,
                       L_opblocks[jsestate->jump_error]);
      }

      if (jsestate->jump_eval_coercion >= 0) {
        EmitCondJumpEQ(Jitcc, v_ret, jsestate->jump_eval_coercion,
                       L_opblocks[jsestate->jump_eval_coercion]);
      }

      EmitJump(Jitcc, L_opblocks[jsestate->jump_end]);
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
      arch::Gp v_aggvaluesp =
          emit_load_ecxt_aggvalues_from_ExprContext(Jitcc, v_econtext);
      arch::Gp v_aggnullsp =
          emit_load_ecxt_aggnulls_from_ExprContext(Jitcc, v_econtext);
      arch::Gp v_value = Jitcc.new_gp_ptr("v_value");
      arch::Gp v_isnull = Jitcc.new_gp32("v_isnull");

      /* load agg value / null */
      EmitLoadFromArray(Jitcc, v_aggvaluesp, op->d.aggref.aggno, v_value,
                        sizeof(Datum));
      EmitLoadFromArray(Jitcc, v_aggnullsp, op->d.aggref.aggno, v_isnull,
                        sizeof(bool));

      /* and store result */
      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue);
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
      arch::Gp v_wfuncnop =
          EmitLoadConstUIntPtr(Jitcc, "v_wfuncnop", &wfunc->wfuncno);
      arch::Gp v_wfuncno = Jitcc.new_gp32("v_wfuncno");
      EmitLoadFromArray(Jitcc, v_wfuncnop, 0, v_wfuncno, sizeof(int32));
      arch::Gp v_aggvaluesp =
          emit_load_ecxt_aggvalues_from_ExprContext(Jitcc, v_econtext);
      arch::Gp v_aggnullsp =
          emit_load_ecxt_aggnulls_from_ExprContext(Jitcc, v_econtext);
      arch::Gp v_value = Jitcc.new_gp_ptr("v_value");
      arch::Gp v_isnull = Jitcc.new_gp32("v_isnull");

      /*
       * Load values[wfuncno] and nulls[wfuncno] using indexed addressing.
       * Datum is pointer-sized, bool is 1 byte.
       */
      {
        /* v_value = aggvalues[wfuncno] (each slot is sizeof(Datum) bytes) */
        arch::Gp v_wfuncno64 = Jitcc.new_gp64("v_wfuncno64");
        EmitSignExtend32to64(Jitcc, v_wfuncno64, v_wfuncno);
        /* multiply by sizeof(Datum) = 8 on 64-bit */
        arch::Gp v_offset = Jitcc.new_gp64("v_offset");
        Jitcc.mov(v_offset, v_wfuncno64);
        EmitShlImm(Jitcc, v_offset, 3); /* << 3 = * 8 */
        arch::Gp v_valueptr = Jitcc.new_gp_ptr("v_valueptr");
        Jitcc.mov(v_valueptr, v_aggvaluesp);
        EmitAddReg(Jitcc, v_valueptr, v_offset);
        EmitLoadFromArray(Jitcc, v_valueptr, 0, v_value, sizeof(Datum));

        /* v_isnull = aggnulls[wfuncno] (each slot is 1 byte) */
        arch::Gp v_nullptr = Jitcc.new_gp_ptr("v_nullptr");
        Jitcc.mov(v_nullptr, v_aggnullsp);
        EmitAddReg(Jitcc, v_nullptr, v_wfuncno64);
        EmitLoadFromArray(Jitcc, v_nullptr, 0, v_isnull, sizeof(bool));
      }

      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue);
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
      arch::Gp v_fcinfo = EmitLoadConstUIntPtr(Jitcc, "v_fcinfo", fcinfo);

      if (opcode == EEOP_AGG_STRICT_DESERIALIZE) {
        arch::Gp v_argnull0 = LoadFuncArgNull(Jitcc, v_fcinfo, 0);
        EmitCondJumpEQ(Jitcc, v_argnull0, 1,
                       L_opblocks[op->d.agg_deserialize.jumpnull]);
      }

      AggState *aggstate = castNode(AggState, State->parent);
      arch::Gp v_tmpcontext =
          EmitLoadConstUIntPtr(Jitcc, "v_tmpcontext",
                               aggstate->tmpcontext->ecxt_per_tuple_memory);
      arch::Gp v_oldcontext = Jitcc.new_gp_ptr("v_oldcontext");
      jit::InvokeNode *InvokeMemoryContextSwitchTo;
      Jitcc.invoke(asmjit::Out(InvokeMemoryContextSwitchTo),
                   JIT_FN_PTR(Jitcc, MemoryContextSwitchTo),
                   jit::FuncSignature::build<MemoryContext, MemoryContext>());
      InvokeMemoryContextSwitchTo->set_arg(0, v_tmpcontext);
      InvokeMemoryContextSwitchTo->set_ret(0, v_oldcontext);

      jit::InvokeNode *PGFunc;
      arch::Gp v_retval = Jitcc.new_gp_ptr("v_retval");
      Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->set_arg(0, fcinfo);
      PGFunc->set_ret(0, v_retval);
      arch::Gp v_fcinfo_isnull =
          emit_load_isnull_from_FunctionCallInfoBaseData(Jitcc, v_fcinfo);

      InvokeMemoryContextSwitchTo = nullptr;
      Jitcc.invoke(asmjit::Out(InvokeMemoryContextSwitchTo),
                   JIT_FN_PTR(Jitcc, MemoryContextSwitchTo),
                   jit::FuncSignature::build<MemoryContext, MemoryContext>());
      InvokeMemoryContextSwitchTo->set_arg(0, v_oldcontext);
      InvokeMemoryContextSwitchTo->set_ret(0, v_oldcontext);

      arch::Gp v_resnullp =
          EmitLoadConstUIntPtr(Jitcc, "v_resnullp", op->resnull);
      arch::Gp v_resvaluep =
          EmitLoadConstUIntPtr(Jitcc, "v_resvaluep", op->resvalue);
      EmitStoreToArray(Jitcc, v_resnullp, 0, v_fcinfo_isnull, sizeof(bool));
      EmitStoreToArray(Jitcc, v_resvaluep, 0, v_retval, sizeof(Datum));

      break;
    }

    case EEOP_AGG_STRICT_INPUT_CHECK_ARGS:
    case EEOP_AGG_STRICT_INPUT_CHECK_ARGS_1:
    case EEOP_AGG_STRICT_INPUT_CHECK_NULLS: {
      int nargs = op->d.agg_strict_input_check.nargs;
      NullableDatum *args = op->d.agg_strict_input_check.args;
      bool *nulls = op->d.agg_strict_input_check.nulls;

      Assert(nargs > 0);

      int jumpnull = op->d.agg_strict_input_check.jumpnull;
      arch::Gp v_argsp = EmitLoadConstUIntPtr(Jitcc, "v_argsp", args);
      arch::Gp v_nullsp = EmitLoadConstUIntPtr(Jitcc, "v_nullsp", nulls);

      /* strict function, check for NULL args */
      for (int argno = 0; argno < nargs; ++argno) {
        arch::Gp v_argisnull = Jitcc.new_gp32("v_argisnull");
        if (opcode == EEOP_AGG_STRICT_INPUT_CHECK_NULLS) {
          EmitLoadFromArray(Jitcc, v_nullsp, argno, v_argisnull, sizeof(bool));
        } else {
          /* Load isnull field from NullableDatum array */
          size_t off = (size_t)argno * sizeof(NullableDatum) +
                       offsetof(NullableDatum, isnull);
          arch::Gp v_argsp_tmp = EmitLoadConstUIntPtr(Jitcc, "v_argsp_tmp",
                                                       args);
          EmitLoadFromFlexibleArray(Jitcc, v_argsp_tmp, off, 0, v_argisnull,
                                    sizeof(bool));
        }

        EmitCondJumpEQ(Jitcc, v_argisnull, 1, L_opblocks[jumpnull]);
      }

      break;
    }
    case EEOP_AGG_PLAIN_PERGROUP_NULLCHECK: {
      int jumpnull = op->d.agg_plain_pergroup_nullcheck.jumpnull;

      arch::Gp v_aggstatep = emit_load_parent_from_ExprState(Jitcc, v_state);
      arch::Gp v_allpergroupsp =
          emit_load_all_pergroups_from_AggState(Jitcc, v_aggstatep);
      arch::Gp v_pergroup_allaggs = Jitcc.new_gp_ptr("v_pergroup_allaggs");
      EmitLoadFromArray(Jitcc, v_allpergroupsp,
                        op->d.agg_plain_pergroup_nullcheck.setoff,
                        v_pergroup_allaggs, sizeof(Datum));
      EmitCondJumpEQ(Jitcc, v_pergroup_allaggs, 0, L_opblocks[jumpnull]);
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
      arch::Gp v_aggstatep = emit_load_parent_from_ExprState(Jitcc, v_state);
      arch::Gp v_pertransp =
          EmitLoadConstUIntPtr(Jitcc, "v_pertransp", pertrans);

      /*
       * pergroup = &aggstate->all_pergroups
       * [op->d.agg_trans.setoff] [op->d.agg_trans.transno];
       */
      int32 setoff = op->d.agg_trans.setoff;
      int32 transno = op->d.agg_trans.transno;
      arch::Gp v_pergroupp = Jitcc.new_gp_ptr("v_pergroupp");
      arch::Gp v_all_pergroupsp =
          emit_load_all_pergroups_from_AggState(Jitcc, v_aggstatep);
      EmitLoadFromArray(Jitcc, v_all_pergroupsp, setoff, v_pergroupp,
                        sizeof(AggStatePerGroup));
      EmitAddImm(Jitcc, v_pergroupp,
                 (int64_t)transno * sizeof(AggStatePerGroupData));

      if (opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYVAL ||
          opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYREF) {
        jit::Label L_NoInit = Jitcc.new_label();
        arch::Gp v_notransvalue =
            emit_load_noTransValue_from_AggStatePerGroupData(Jitcc,
                                                             v_pergroupp);
        EmitCondJumpNE(Jitcc, v_notransvalue, 1, L_NoInit);
        {
          /* init the transition value if necessary */
          arch::Gp v_aggcontext = EmitLoadConstUIntPtr(
              Jitcc, "v_aggcontext", op->d.agg_trans.aggcontext);
          jit::InvokeNode *InvokeExecAggInitGroup;
          Jitcc.invoke(asmjit::Out(InvokeExecAggInitGroup),
               JIT_FN_PTR(Jitcc, ExecAggInitGroup),
              jit::FuncSignature::build<void, AggState *, AggStatePerTrans,
                                        AggStatePerGroup, ExprContext *>());
          InvokeExecAggInitGroup->set_arg(0, v_aggstatep);
          InvokeExecAggInitGroup->set_arg(1, v_pertransp);
          InvokeExecAggInitGroup->set_arg(2, v_pergroupp);
          InvokeExecAggInitGroup->set_arg(3, v_aggcontext);

          EmitJump(Jitcc, L_opblocks[opno + 1]);
        }

        Jitcc.bind(L_NoInit);
      }

      if (opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYVAL ||
          opcode == EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYREF ||
          opcode == EEOP_AGG_PLAIN_TRANS_STRICT_BYVAL ||
          opcode == EEOP_AGG_PLAIN_TRANS_STRICT_BYREF) {
        arch::Gp v_transnull =
            emit_load_transValueIsNull_from_AggStatePerGroupData(Jitcc,
                                                                 v_pergroupp);
        EmitCondJumpEQ(Jitcc, v_transnull, 1, L_opblocks[opno + 1]);
      }

      arch::Gp v_fcinfo =
          EmitLoadConstUIntPtr(Jitcc, "v_fcinfo", fcinfo);
      arch::Gp v_aggcontext = EmitLoadConstUIntPtr(Jitcc, "v_aggcontext",
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
      arch::Gp v_tmpcontext =
          EmitLoadConstUIntPtr(Jitcc, "v_tmpcontext",
                               aggstate->tmpcontext->ecxt_per_tuple_memory);
      arch::Gp v_oldcontext = Jitcc.new_gp_ptr("v_oldcontext");
      jit::InvokeNode *InvokeMemoryContextSwitchTo;
      Jitcc.invoke(asmjit::Out(InvokeMemoryContextSwitchTo),
                   JIT_FN_PTR(Jitcc, MemoryContextSwitchTo),
                   jit::FuncSignature::build<MemoryContext, MemoryContext>());
      InvokeMemoryContextSwitchTo->set_arg(0, v_tmpcontext);
      InvokeMemoryContextSwitchTo->set_ret(0, v_oldcontext);

      /* store transvalue in fcinfo->args[0] */
      arch::Gp v_transvalue =
          emit_load_transValue_from_AggStatePerGroupData(Jitcc, v_pergroupp);
      arch::Gp v_transnull =
          emit_load_transValueIsNull_from_AggStatePerGroupData(Jitcc,
                                                               v_pergroupp);
      StoreFuncArgValue(Jitcc, v_fcinfo, 0, v_transvalue);
      StoreFuncArgNull(Jitcc, v_fcinfo, 0, v_transnull);
      emit_store_isnull_to_FunctionCallInfoBaseData(Jitcc, v_fcinfo,
                                                    jit::imm(0));

      arch::Gp v_retval = Jitcc.new_gp_ptr("v_retval");
      jit::InvokeNode *PGFunc;
      Jitcc.invoke(asmjit::Out(PGFunc), JIT_FN_PTR(Jitcc, fcinfo->flinfo->fn_addr),
                   jit::FuncSignature::build<Datum, FunctionCallInfo>());
      PGFunc->set_arg(0, v_fcinfo);
      PGFunc->set_ret(0, v_retval);
      arch::Gp v_fcinfo_isnull =
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
        jit::Label L_NoCall = Jitcc.new_label();
        arch::Gp v_transvalue2 =
            emit_load_transValue_from_AggStatePerGroupData(Jitcc, v_pergroupp);
        arch::Gp v_transnull2 =
            emit_load_transValueIsNull_from_AggStatePerGroupData(Jitcc,
                                                                 v_pergroupp);
        EmitCondJumpRegEQ(Jitcc, v_transvalue2, v_retval, L_NoCall);

        /* store trans value */
        {
          /*
           * FIXME: It's seems v_transvalue is not properly loaded in -O3 and I
           * don't know why.
           */
          v_transvalue2 = emit_load_transValue_from_AggStatePerGroupData(
              Jitcc, v_pergroupp);
          v_transnull2 = emit_load_transValueIsNull_from_AggStatePerGroupData(
              Jitcc, v_pergroupp);
        }

        jit::InvokeNode *InvokeExecAggCopyTransValue;
        arch::Gp v_newval = Jitcc.new_gp_ptr("v_newval");
        Jitcc.invoke(asmjit::Out(InvokeExecAggCopyTransValue),
             JIT_FN_PTR(Jitcc, ExecAggCopyTransValue),
            jit::FuncSignature::build<Datum, AggState *, AggStatePerTrans,
                                      Datum, bool, Datum, bool>());
        InvokeExecAggCopyTransValue->set_arg(0, v_aggstatep);
        InvokeExecAggCopyTransValue->set_arg(1, v_pertransp);
        InvokeExecAggCopyTransValue->set_arg(2, v_retval);
        InvokeExecAggCopyTransValue->set_arg(3, v_fcinfo_isnull);
        InvokeExecAggCopyTransValue->set_arg(4, v_transvalue2);
        InvokeExecAggCopyTransValue->set_arg(5, v_transnull2);
        InvokeExecAggCopyTransValue->set_ret(0, v_newval);

        /* store trans value */
        emit_store_transValue_to_AggStatePerGroupData(Jitcc, v_pergroupp,
                                                      v_newval);
        emit_store_transValueIsNull_to_AggStatePerGroupData(Jitcc, v_pergroupp,
                                                            v_fcinfo_isnull);

        InvokeMemoryContextSwitchTo = nullptr;
        Jitcc.invoke(asmjit::Out(InvokeMemoryContextSwitchTo),
                     JIT_FN_PTR(Jitcc, MemoryContextSwitchTo),
                     jit::FuncSignature::build<MemoryContext, MemoryContext>());
        InvokeMemoryContextSwitchTo->set_arg(0, v_oldcontext);

        EmitJump(Jitcc, L_opblocks[opno + 1]);

        Jitcc.bind(L_NoCall);
      }

      /* store trans value */
      emit_store_transValue_to_AggStatePerGroupData(Jitcc, v_pergroupp,
                                                    v_retval);
      emit_store_transValueIsNull_to_AggStatePerGroupData(Jitcc, v_pergroupp,
                                                          v_fcinfo_isnull);

      InvokeMemoryContextSwitchTo = nullptr;
      Jitcc.invoke(asmjit::Out(InvokeMemoryContextSwitchTo),
                   JIT_FN_PTR(Jitcc, MemoryContextSwitchTo),
                   jit::FuncSignature::build<MemoryContext, MemoryContext>());
      InvokeMemoryContextSwitchTo->set_arg(0, v_oldcontext);

      break;
    }
    case EEOP_AGG_PRESORTED_DISTINCT_SINGLE: {
      AggState *aggstate = castNode(AggState, State->parent);
      AggStatePerTrans pertrans = op->d.agg_presorted_distinctcheck.pertrans;
      int jumpdistinct = op->d.agg_presorted_distinctcheck.jumpdistinct;
      arch::Gp v_aggstatep =
          EmitLoadConstUIntPtr(Jitcc, "v_aggstate", aggstate);
      arch::Gp v_pertrans =
          EmitLoadConstUIntPtr(Jitcc, "v_pertrans", pertrans);
      arch::Gp v_retval = Jitcc.new_gp32("v_retval");
      jit::InvokeNode *InvokeExecEvalPreOrderedDistinctSingle;
      Jitcc.invoke(asmjit::Out(InvokeExecEvalPreOrderedDistinctSingle),
          JIT_FN_PTR(Jitcc, ExecEvalPreOrderedDistinctSingle),
          jit::FuncSignature::build<bool, AggState *, AggStatePerTrans>());
      InvokeExecEvalPreOrderedDistinctSingle->set_arg(0, v_aggstatep);
      InvokeExecEvalPreOrderedDistinctSingle->set_arg(1, v_pertrans);
      InvokeExecEvalPreOrderedDistinctSingle->set_ret(0, v_retval);

      EmitCondJumpNE(Jitcc, v_retval, 1, L_opblocks[jumpdistinct]);

      break;
    }
    case EEOP_AGG_PRESORTED_DISTINCT_MULTI: {
      AggState *aggstate = castNode(AggState, State->parent);
      AggStatePerTrans pertrans = op->d.agg_presorted_distinctcheck.pertrans;
      int jumpdistinct = op->d.agg_presorted_distinctcheck.jumpdistinct;
      arch::Gp v_aggstatep =
          EmitLoadConstUIntPtr(Jitcc, "v_aggstate", aggstate);
      arch::Gp v_pertrans =
          EmitLoadConstUIntPtr(Jitcc, "v_pertrans", pertrans);
      arch::Gp v_retval = Jitcc.new_gp32("v_retval");
      jit::InvokeNode *InvokeExecEvalPreOrderedDistinctMulti;
      Jitcc.invoke(asmjit::Out(InvokeExecEvalPreOrderedDistinctMulti),
          JIT_FN_PTR(Jitcc, ExecEvalPreOrderedDistinctMulti),
          jit::FuncSignature::build<bool, AggState *, AggStatePerTrans>());
      InvokeExecEvalPreOrderedDistinctMulti->set_arg(0, v_aggstatep);
      InvokeExecEvalPreOrderedDistinctMulti->set_arg(1, v_pertrans);
      InvokeExecEvalPreOrderedDistinctMulti->set_ret(0, v_retval);

      EmitCondJumpNE(Jitcc, v_retval, 1, L_opblocks[jumpdistinct]);

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

  jit::Error finalize_err = Jitcc.finalize();
  if (finalize_err != jit::kErrorOk) {
    ereport(LOG, (errmsg("AsmJit finalize failed: %s",
                         jit::DebugUtils::error_as_string(finalize_err))));
    return false;
  }

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
  if (err != jit::kErrorOk) {
    ereport(LOG,
            (errmsg("Jit failed: %s", jit::DebugUtils::error_as_string(err))));
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
