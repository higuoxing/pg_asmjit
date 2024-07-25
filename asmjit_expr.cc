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

#define TYPES_INFO(struct_type, member_type, member_name, reg_type)            \
  static inline x86::Gp load_##member_name##_from_##struct_type(               \
      x86::Compiler &cc, x86::Gp &object_addr) {                               \
    x86::Mem member_ptr = x86::ptr(                                            \
        object_addr, offsetof(struct_type, member_name), sizeof(member_type)); \
    x86::Gp member = cc.new##reg_type();                                       \
    cc.mov(member, member_ptr);                                                \
    return member;                                                             \
  }
#include "jit_types_info.inc"
#undef TYPES_INFO

static inline void JitLoadFromArray(x86::Compiler &cc, x86::Gp &Array,
                                    size_t Index, x86::Gp &Elem,
                                    size_t ElemSize) {
  x86::Mem ElemPtr = x86::ptr(Array, Index * ElemSize, sizeof(ElemSize));
  cc.mov(Elem, ElemPtr);
}

static inline void JitStoreToArray(x86::Compiler &cc, x86::Gp &Array,
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

  /*
   * Addresses for
   *   expression->resvalue,
   *   expression->resnull,
   *   expression->resultslot,
   */
  x86::Mem StateResValuePtr = x86::ptr(
               Expression, offsetof(ExprState, resvalue), sizeof(Datum)),
           StateResNullPtr =
               x86::ptr(Expression, offsetof(ExprState, resnull), sizeof(bool)),
           StateResultSlotPtr =
               x86::ptr(Expression, offsetof(ExprState, resultslot),
                        sizeof(TupleTableSlot *));

  /*
   * Addresses for
   *   econtext->ecxt_scantuple,
   *   econtext->ecxt_innertuple,
   *   econtext->ecxt_outertuple,
   */
  x86::Mem EContextScanTuplePtr =
               x86::ptr(EContext, offsetof(ExprContext, ecxt_scantuple),
                        sizeof(TupleTableSlot *)),
           EContextInnerTuplePtr =
               x86::ptr(EContext, offsetof(ExprContext, ecxt_innertuple),
                        sizeof(TupleTableSlot *)),
           EContextOuterTuplePtr =
               x86::ptr(EContext, offsetof(ExprContext, ecxt_outertuple),
                        sizeof(TupleTableSlot *));

  /*
   * Addresses for
   *   econtext->ecxt_scantuple->tts_values,
   *   econtext->ext_scantuple->tts_isnull,
   *   econtext->ecxt_innertuple->tts_values,
   *   econtext->ext_innertuple->tts_isnull,
   *   econtext->ecxt_outertuple->tts_values,
   *   econtext->ext_outertuple->tts_isnull,
   *   expression->resultslot->tts_values,
   *   expression->resultslot->tts_isnull,
   */
  x86::Gp ScanTuple = load_ecxt_scantuple_from_ExprContext(Jitcc, EContext),
          InnerTuple = load_ecxt_innertuple_from_ExprContext(Jitcc, EContext),
          OuterTuple = load_ecxt_outertuple_from_ExprContext(Jitcc, EContext);

  x86::Gp ResultTuple = Jitcc.newUIntPtr();
  Jitcc.mov(ResultTuple, StateResultSlotPtr);

  x86::Mem ScanTupleValuesPtr =
               x86::ptr(ScanTuple, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           ScanTupleIsNullsPtr = x86::ptr(
               ScanTuple, offsetof(TupleTableSlot, tts_isnull), sizeof(bool *)),
           InnerTupleValuesPtr =
               x86::ptr(InnerTuple, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           InnerTupleIsNullsPtr =
               x86::ptr(InnerTuple, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *)),
           OuterTupleValuesPtr =
               x86::ptr(OuterTuple, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           OutertupleIsNullsPtr =
               x86::ptr(OuterTuple, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *)),
           ResultTupleValuesPtr =
               x86::ptr(ResultTuple, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           ResultTupleIsNullsPtr =
               x86::ptr(ResultTuple, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *));

  x86::Mem StateParentPtr =
      x86::ptr(Expression, offsetof(ExprState, parent), sizeof(PlanState *));
  x86::Gp StateParent = Jitcc.newUIntPtr();
  Jitcc.mov(StateParent, StateParentPtr);

  jit::Label *Opblocks =
      (jit::Label *)palloc(State->steps_len * sizeof(jit::Label));
  for (int OpIndex = 0; OpIndex < State->steps_len; ++OpIndex)
    Opblocks[OpIndex] = Jitcc.newLabel();

  for (int OpIndex = 0; OpIndex < State->steps_len; ++OpIndex) {
    ExprEvalStep *Op = &State->steps[OpIndex];
    ExprEvalOp Opcode = ExecEvalStepOp(State, Op);

    Jitcc.bind(Opblocks[OpIndex]);

    x86::Gp ResValue = Jitcc.newUIntPtr(), ResNull = Jitcc.newUIntPtr();
    Jitcc.mov(ResValue, jit::imm(Op->resvalue));
    Jitcc.mov(ResNull, jit::imm(Op->resnull));
    x86::Mem ResValuePtr = x86::ptr(ResValue, 0, sizeof(Datum)),
             ResNullPtr = x86::ptr(ResNull, 0, sizeof(bool));

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
      x86::Gp TempStateResValue = Jitcc.newUIntPtr(),
              TempStateResNull = Jitcc.newInt8();
      Jitcc.mov(TempStateResValue, StateResValuePtr);
      Jitcc.mov(TempStateResNull, StateResNullPtr);

      /* *isnull = expression->resnull */
      x86::Mem IsNullPtr = x86::ptr(IsNull, 0, sizeof(bool));
      Jitcc.mov(IsNullPtr, TempStateResNull);

      /* return expression->resvalue */
      Jitcc.ret(TempStateResValue);
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

      x86::Mem SlotPtr =
          Opcode == EEOP_INNER_FETCHSOME
              ? EContextInnerTuplePtr
              : (Opcode == EEOP_OUTER_FETCHSOME ? EContextOuterTuplePtr
                                                : EContextScanTuplePtr);

      /* Compute the address of Slot->tts_nvalid */
      x86::Gp Slot = Jitcc.newUIntPtr();
      Jitcc.mov(Slot, SlotPtr);

      x86::Gp TtsNvalid = load_tts_nvalid_from_TupleTableSlot(Jitcc, Slot);

      /*
       * Check if all required attributes are available, or whether deforming is
       * required.
       */
      Jitcc.cmp(TtsNvalid, jit::imm(Op->d.fetch.last_var));
      Jitcc.jge(Opblocks[OpIndex + 1]);

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
      x86::Mem SlotValuesPtr =
          Opcode == EEOP_INNER_VAR
              ? InnerTupleValuesPtr
              : (Opcode == EEOP_OUTER_VAR ? OuterTupleValuesPtr
                                          : ScanTupleValuesPtr);
      x86::Mem SlotIsNullsPtr =
          Opcode == EEOP_INNER_VAR
              ? InnerTupleIsNullsPtr
              : (Opcode == EEOP_OUTER_VAR ? OutertupleIsNullsPtr
                                          : ScanTupleIsNullsPtr);
      x86::Gp SlotValues = Jitcc.newUIntPtr(), SlotIsNulls = Jitcc.newUIntPtr();
      Jitcc.mov(SlotValues, SlotValuesPtr);
      Jitcc.mov(SlotIsNulls, SlotIsNullsPtr);

      int Attrnum = Op->d.var.attnum;

      x86::Gp SlotValue = Jitcc.newUIntPtr(), SlotIsNull = Jitcc.newInt8();
      JitLoadFromArray(Jitcc, SlotValues, Attrnum, SlotValue, sizeof(Datum));
      JitLoadFromArray(Jitcc, SlotIsNulls, Attrnum, SlotIsNull, sizeof(bool));

      Jitcc.mov(ResValuePtr, SlotValue);
      Jitcc.mov(ResNullPtr, SlotIsNull);

      break;
    }
    case EEOP_INNER_SYSVAR:
    case EEOP_OUTER_SYSVAR:
    case EEOP_SCAN_SYSVAR: {
      x86::Mem SlotPtr =
          Opcode == EEOP_INNER_VAR
              ? EContextInnerTuplePtr
              : (Opcode == EEOP_OUTER_SYSVAR ? EContextOuterTuplePtr
                                             : EContextScanTuplePtr);
      x86::Gp Slot = Jitcc.newUIntPtr();
      Jitcc.mov(Slot, SlotPtr);

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
      x86::Mem SlotValuesPtr =
          Opcode == EEOP_ASSIGN_INNER_VAR
              ? InnerTupleValuesPtr
              : (Opcode == EEOP_ASSIGN_OUTER_VAR ? OuterTupleValuesPtr
                                                 : ScanTupleValuesPtr);
      x86::Mem SlotIsNullsPtr =
          Opcode == EEOP_ASSIGN_INNER_VAR
              ? InnerTupleIsNullsPtr
              : (Opcode == EEOP_ASSIGN_OUTER_VAR ? OutertupleIsNullsPtr
                                                 : ScanTupleIsNullsPtr);

      x86::Gp SlotValues = Jitcc.newUIntPtr(), SlotIsNulls = Jitcc.newUIntPtr();
      Jitcc.mov(SlotValues, SlotValuesPtr);
      Jitcc.mov(SlotIsNulls, SlotIsNullsPtr);

      int Attrnum = Op->d.assign_var.attnum;

      /* Load data. */
      x86::Gp SlotValue = Jitcc.newUIntPtr(), SlotIsNull = Jitcc.newInt8();
      JitLoadFromArray(Jitcc, SlotValues, Attrnum, SlotValue, sizeof(Datum));
      JitLoadFromArray(Jitcc, SlotIsNulls, Attrnum, SlotIsNull, sizeof(bool));

      /* Save the result. */
      int Resultnum = Op->d.assign_var.resultnum;
      Jitcc.mov(SlotValues, ResultTupleValuesPtr);
      Jitcc.mov(SlotIsNulls, ResultTupleIsNullsPtr);

      JitStoreToArray(Jitcc, SlotValues, Resultnum, SlotValue, sizeof(Datum));
      JitStoreToArray(Jitcc, SlotIsNulls, Resultnum, SlotIsNull, sizeof(bool));

      break;
    }

    case EEOP_ASSIGN_TMP:
    case EEOP_ASSIGN_TMP_MAKE_RO: {
      size_t ResultNum = Op->d.assign_tmp.resultnum;

      /* Load expression->resvalue and expression->resnull */
      x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
              TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, StateResValuePtr);
      Jitcc.mov(TempStateResnull, StateResNullPtr);

      /*
       * Compute the addresses of
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      x86::Gp ResultValues = Jitcc.newUIntPtr(),
              ResultIsNulls = Jitcc.newUIntPtr();
      Jitcc.mov(ResultValues, ResultTupleValuesPtr);
      Jitcc.mov(ResultIsNulls, ResultTupleIsNullsPtr);

      /*
       * Store nullness.
       */
      JitStoreToArray(Jitcc, ResultIsNulls, ResultNum, TempStateResnull,
                      sizeof(bool));

      if (Opcode == EEOP_ASSIGN_TMP_MAKE_RO) {
        Jitcc.cmp(TempStateResnull, jit::imm(1));
        Jitcc.je(Opblocks[OpIndex + 1]);

        jit::InvokeNode *MakeExpandedObjectReadOnlyInternalFunc;
        Jitcc.invoke(&MakeExpandedObjectReadOnlyInternalFunc,
                     jit::imm(MakeExpandedObjectReadOnlyInternal),
                     jit::FuncSignature::build<Datum, Datum>());
        MakeExpandedObjectReadOnlyInternalFunc->setArg(0, TempStateResvalue);
        MakeExpandedObjectReadOnlyInternalFunc->setRet(0, TempStateResvalue);
      }

      /* Finally, store the result. */
      JitStoreToArray(Jitcc, ResultValues, ResultNum, TempStateResvalue,
                      sizeof(Datum));
      break;
    }
    case EEOP_CONST: {
      x86::Gp ConstVal = Jitcc.newUIntPtr(), ConstNull = Jitcc.newInt8();

      Jitcc.mov(ConstVal, jit::imm(Op->d.constval.value));
      Jitcc.mov(ConstNull, jit::imm(Op->d.constval.isnull));

      /*
       * Store Op->d.constval.value to Op->resvalue.
       * Store Op->d.constval.isnull to Op->resnull.
       */
      Jitcc.mov(ResValuePtr, ConstVal);
      Jitcc.mov(ResNullPtr, ConstNull);

      break;
    }
    case EEOP_FUNCEXPR:
    case EEOP_FUNCEXPR_STRICT: {
      FunctionCallInfo FuncCallInfo = Op->d.func.fcinfo_data;

      x86::Gp FuncCallInfoAddr = Jitcc.newUIntPtr();
      Jitcc.mov(FuncCallInfoAddr, FuncCallInfo);

      jit::Label InvokePGFunc = Jitcc.newLabel();

      if (Opcode == EEOP_FUNCEXPR_STRICT) {
        jit::Label StrictFail = Jitcc.newLabel();
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
          Jitcc.je(StrictFail);
        }

        Jitcc.jmp(InvokePGFunc);

        Jitcc.bind(StrictFail);
        /* Op->resnull = true */
        Jitcc.mov(ResNullPtr, jit::imm(1));
        Jitcc.jmp(Opblocks[OpIndex + 1]);
      }

      /*
       * Before invoking PGFuncs, we should set FuncCallInfo->isnull to false.
       */
      Jitcc.bind(InvokePGFunc);
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
      Jitcc.mov(ResValuePtr, RetValue);
      x86::Gp FuncCallInfoIsNull = Jitcc.newInt8();
      Jitcc.mov(FuncCallInfoIsNull, FuncCallInfoIsNullPtr);
      Jitcc.mov(ResNullPtr, FuncCallInfoIsNull);

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
      jit::Label HandleNullOrFalse = Jitcc.newLabel();

      x86::Gp Resvalue = Jitcc.newUIntPtr(), Resnull = Jitcc.newInt8();
      Jitcc.mov(Resvalue, ResValuePtr);
      Jitcc.mov(Resnull, ResNullPtr);

      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.je(HandleNullOrFalse);

      Jitcc.cmp(Resvalue, jit::imm(0));
      Jitcc.je(HandleNullOrFalse);

      Jitcc.jmp(Opblocks[OpIndex + 1]);

      /* Handling null or false. */
      Jitcc.bind(HandleNullOrFalse);

      /* Set resnull and resvalue to false. */
      Jitcc.mov(ResValuePtr, jit::imm(0));
      Jitcc.mov(ResNullPtr, jit::imm(0));

      Jitcc.jmp(Opblocks[Op->d.qualexpr.jumpdone]);

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
