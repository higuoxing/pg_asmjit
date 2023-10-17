#if __cplusplus > 199711L
/*
 * The register storage type is deprecated in C++11.
 */
#define register
#endif

#include <asmjit/asmjit.h>

namespace jit = asmjit;
namespace x86 = asmjit::x86;

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"

#include "executor/execExpr.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "jit/jit.h"
#include "nodes/execnodes.h"
#include "nodes/pg_list.h"
#include "portability/instr_time.h"
#include "storage/ipc.h"
#include "utils/memutils.h"
#include "utils/palloc.h"
#include "utils/resowner.h"
#include "utils/resowner_private.h"

PG_MODULE_MAGIC;

static bool JitSessionInitialized = false;
static jit::JitRuntime Runtime;

typedef struct AsmJitContext {
  JitContext base;
  List *funcs;
} AsmJitContext;

static void JitResetAfterError(void) { /* TODO. */
}

static void JitReleaseContext(JitContext *Ctx) {
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
}

static void JitInitializeSession(void) {
  if (JitSessionInitialized)
    return;

  JitSessionInitialized = true;
}

static AsmJitContext *JitCreateContext(int JitFlags) {
  JitInitializeSession();

  ResourceOwnerEnlargeJIT(CurrentResourceOwner);

  AsmJitContext *Context = (AsmJitContext *)MemoryContextAllocZero(
      TopMemoryContext, sizeof(AsmJitContext));
  Context->base.flags = JitFlags;
  Context->funcs = NIL;

  /* ensure cleanup */
  Context->base.resowner = CurrentResourceOwner;
  ResourceOwnerRememberJIT(CurrentResourceOwner, PointerGetDatum(Context));

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

static bool JitCompileExpr(ExprState *State) {
  PlanState *Parent = State->parent;
  AsmJitContext *Context = nullptr;
  instr_time CodeGenStartTime, CodeGenEndTime;

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
      jit::FuncSignatureT<Datum, ExprState *, ExprContext *, bool *>());

  x86::Gp ExpressionAddr = Jitcc.newUIntPtr("expression"),
          EContextAddr = Jitcc.newUIntPtr("econtext"),
          IsNullAddr = Jitcc.newUIntPtr("isnull");

  JittedFunc->setArg(0, ExpressionAddr);
  JittedFunc->setArg(1, EContextAddr);
  JittedFunc->setArg(2, IsNullAddr);

  /*
   * Addresses for
   *   expression->resvalue,
   *   expression->resnull,
   */
  x86::Mem v_StateResvalue = x86::ptr(
               ExpressionAddr, offsetof(ExprState, resvalue), sizeof(Datum)),
           v_StateResnull = x86::ptr(
               ExpressionAddr, offsetof(ExprState, resnull), sizeof(bool));

  /*
   * Addresses for
   *   econtext->ecxt_scantuple,
   *   econtext->ecxt_innertuple,
   *   econtext->ecxt_outertuple,
   */
  x86::Mem v_EContextScantuple =
               x86::ptr(EContextAddr, offsetof(ExprContext, ecxt_scantuple),
                        sizeof(TupleTableSlot *)),
           v_EContextInnertuple =
               x86::ptr(EContextAddr, offsetof(ExprContext, ecxt_innertuple),
                        sizeof(TupleTableSlot *)),
           v_EContextOutertuple =
               x86::ptr(EContextAddr, offsetof(ExprContext, ecxt_outertuple),
                        sizeof(TupleTableSlot *));

  /*
   * Addresses for
   *   econtext->ecxt_scantuple->tts_values,
   *   econtext->ext_scantuple->tts_isnull,
   *   econtext->ecxt_innertuple->tts_values,
   *   econtext->ext_innertuple->tts_isnull,
   *   econtext->ecxt_outertuple->tts_values,
   *   econtext->ext_outertuple->tts_isnull,
   */
  x86::Gp ScantupleAddr = Jitcc.newUIntPtr(),
          InnertupleAddr = Jitcc.newUIntPtr(),
          OutertupleAddr = Jitcc.newUIntPtr();
  Jitcc.mov(ScantupleAddr, v_EContextScantuple);
  Jitcc.mov(InnertupleAddr, v_EContextInnertuple);
  Jitcc.mov(OutertupleAddr, v_EContextOutertuple);
  x86::Mem v_ScantupleValues =
               x86::ptr(ScantupleAddr, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           v_ScantupleIsnulls =
               x86::ptr(ScantupleAddr, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *)),
           v_InnertupleValues =
               x86::ptr(InnertupleAddr, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           v_InnertupleIsnulls =
               x86::ptr(InnertupleAddr, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *)),
           v_OutertupleValues =
               x86::ptr(OutertupleAddr, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           v_OutertupleIsnulls =
               x86::ptr(OutertupleAddr, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *));

  x86::Mem v_StateParent = x86::ptr(ExpressionAddr, offsetof(ExprState, parent),
                                    sizeof(PlanState *));
  x86::Gp ParentAddr = Jitcc.newUIntPtr();
  Jitcc.mov(ParentAddr, v_StateParent);

  jit::Label *Opblocks =
      (jit::Label *)palloc(State->steps_len * sizeof(jit::Label));
  for (int OpIndex = 0; OpIndex < State->steps_len; ++OpIndex)
    Opblocks[OpIndex] = Jitcc.newLabel();

  for (int OpIndex = 0; OpIndex < State->steps_len; ++OpIndex) {
    ExprEvalStep *Op = &State->steps[OpIndex];
    ExprEvalOp Opcode = ExecEvalStepOp(State, Op);

    Jitcc.bind(Opblocks[OpIndex]);

    x86::Gp ResvalueAddr = Jitcc.newUIntPtr(), ResnullAddr = Jitcc.newUIntPtr();
    Jitcc.mov(ResvalueAddr, jit::imm(Op->resvalue));
    Jitcc.mov(ResnullAddr, jit::imm(Op->resnull));
    x86::Mem v_Resvalue = x86::ptr(ResvalueAddr, 0, sizeof(Datum)),
             v_Resnull = x86::ptr(ResnullAddr, 0, sizeof(bool));

    switch (Opcode) {
    case EEOP_DONE: {
      /* Load expression->resvalue and expression->resnull */
      x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
              TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, v_StateResvalue);
      Jitcc.mov(TempStateResnull, v_StateResnull);

      /* *isnull = expression->resnull */
      x86::Mem v_IsNull = x86::ptr(IsNullAddr, 0, sizeof(bool));
      Jitcc.mov(v_IsNull, TempStateResnull);

      /* return expression->resvalue */
      Jitcc.ret(TempStateResvalue);
      Jitcc.endFunc();
      break;
    }
    case EEOP_INNER_FETCHSOME:
    case EEOP_OUTER_FETCHSOME:
    case EEOP_SCAN_FETCHSOME: {
      /* Step should not have been generated. */
      Assert(TtsOps != &TTSOpsVirtual);

      x86::Mem v_Slot =
          Opcode == EEOP_INNER_FETCHSOME
              ? v_EContextInnertuple
              : (Opcode == EEOP_OUTER_FETCHSOME ? v_EContextOutertuple
                                                : v_EContextScantuple);

      /* Compute the address of Slot->tts_nvalid */
      x86::Gp SlotAddr = Jitcc.newUIntPtr();
      Jitcc.mov(SlotAddr, v_Slot);
      x86::Mem v_TtsNvalid = x86::ptr(
          SlotAddr, offsetof(TupleTableSlot, tts_nvalid), sizeof(AttrNumber));
      x86::Gp TtsNvalid = Jitcc.newInt16();
      Jitcc.mov(TtsNvalid, v_TtsNvalid);

      /*
       * Check if all required attributes are available, or whether deforming is
       * required.
       */
      Jitcc.cmp(TtsNvalid, jit::imm(Op->d.fetch.last_var));
      Jitcc.jge(Opblocks[OpIndex + 1]);

      /*
       * TODO: Add support for JITing the deforming process.
       */
      jit::InvokeNode *SlotGetSomeAttrsInt;
      Jitcc.invoke(&SlotGetSomeAttrsInt, jit::imm(slot_getsomeattrs_int),
                   jit::FuncSignatureT<void, TupleTableSlot *, int>());
      SlotGetSomeAttrsInt->setArg(0, SlotAddr);
      SlotGetSomeAttrsInt->setArg(1, jit::imm(Op->d.fetch.last_var));
      break;
    }

    case EEOP_INNER_VAR:
    case EEOP_OUTER_VAR:
    case EEOP_SCAN_VAR: {
      x86::Mem v_SlotValues =
          Opcode == EEOP_INNER_VAR
              ? v_InnertupleValues
              : (Opcode == EEOP_OUTER_VAR ? v_OutertupleValues
                                          : v_ScantupleValues);
      x86::Mem v_SlotIsnulls =
          Opcode == EEOP_INNER_VAR
              ? v_InnertupleIsnulls
              : (Opcode == EEOP_OUTER_VAR ? v_OutertupleIsnulls
                                          : v_ScantupleIsnulls);
      x86::Gp SlotValuesAddr = Jitcc.newUIntPtr(),
              SlotIsnullsAddr = Jitcc.newUIntPtr();
      Jitcc.mov(SlotValuesAddr, v_SlotValues);
      Jitcc.mov(SlotIsnullsAddr, v_SlotIsnulls);

      int Attrnum = Op->d.var.attnum;
      x86::Mem v_SlotValue =
          x86::ptr(SlotValuesAddr, Attrnum * sizeof(Datum), sizeof(Datum));
      x86::Mem v_SlotIsnull =
          x86::ptr(SlotIsnullsAddr, Attrnum * sizeof(bool), sizeof(bool));

      x86::Gp SlotValue = Jitcc.newUIntPtr(), SlotIsnull = Jitcc.newInt8();
      Jitcc.mov(SlotValue, v_SlotValue);
      Jitcc.mov(SlotIsnull, v_SlotIsnull);

      Jitcc.mov(v_Resvalue, SlotValue);
      Jitcc.mov(v_Resnull, SlotIsnull);

      break;
    }
    case EEOP_ASSIGN_TMP: {
      size_t ResultNum = Op->d.assign_tmp.resultnum;

      /* Load expression->resvalue and expression->resnull */
      x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
              TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, v_StateResvalue);
      Jitcc.mov(TempStateResnull, v_StateResnull);

      /*
       * Compute the addresses of expression->resultslot->tts_values and
       * expression->resultslot->tts_isnull
       */
      x86::Mem v_StateResultslot =
          x86::ptr(ExpressionAddr, offsetof(ExprState, resultslot),
                   sizeof(TupleTableSlot *));
      x86::Gp StateResultslotAddr = Jitcc.newUInt64();
      Jitcc.mov(StateResultslotAddr, v_StateResultslot);
      x86::Mem v_TtsValues = x86::ptr(StateResultslotAddr,
                                      offsetof(TupleTableSlot, tts_values),
                                      sizeof(Datum *)),
               v_TtsIsnulls = x86::ptr(StateResultslotAddr,
                                       offsetof(TupleTableSlot, tts_isnull),
                                       sizeof(bool *));

      /*
       * Compute the addresses of
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      x86::Gp TtsValueAddr = Jitcc.newUIntPtr(),
              TtsIsnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(TtsValueAddr, v_TtsValues);
      Jitcc.mov(TtsIsnullAddr, v_TtsIsnulls);
      x86::Mem v_TtsValue = x86::ptr(TtsValueAddr, sizeof(Datum) * ResultNum,
                                     sizeof(Datum)),
               v_TtsIsnull = x86::ptr(TtsIsnullAddr, sizeof(bool) * ResultNum,
                                      sizeof(bool));

      /*
       * Store expression->resvalue and expression->resnull to
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      Jitcc.mov(v_TtsValue, TempStateResvalue);
      Jitcc.mov(v_TtsIsnull, TempStateResnull);

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
      Jitcc.mov(v_Resvalue, ConstVal);
      Jitcc.mov(v_Resnull, ConstNull);

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
          x86::Mem v_FuncCallInfoArgNIsNull =
              x86::ptr(FuncCallInfoAddr,
                       offsetof(FunctionCallInfoBaseData, args) +
                           ArgIndex * sizeof(NullableDatum) +
                           offsetof(NullableDatum, isnull),
                       sizeof(bool));
          x86::Gp FuncCallInfoArgNIsNull = Jitcc.newInt8();
          Jitcc.mov(FuncCallInfoArgNIsNull, v_FuncCallInfoArgNIsNull);
          Jitcc.cmp(FuncCallInfoArgNIsNull, jit::imm(1));
          Jitcc.je(StrictFail);
        }

        Jitcc.jmp(InvokePGFunc);

        Jitcc.bind(StrictFail);
        /* Op->resnull = true */
        Jitcc.mov(v_Resnull, jit::imm(1));
        Jitcc.jmp(Opblocks[OpIndex + 1]);
      }

      /*
       * Before invoking PGFuncs, we should set FuncCallInfo->isnull to false.
       */
      Jitcc.bind(InvokePGFunc);
      x86::Mem v_FuncCallInfoIsNull =
          x86::ptr(FuncCallInfoAddr, offsetof(FunctionCallInfoBaseData, isnull),
                   sizeof(bool));
      Jitcc.mov(v_FuncCallInfoIsNull, jit::imm(0));

      jit::InvokeNode *PGFunc;
      x86::Gp RetValue = Jitcc.newUIntPtr();
      Jitcc.invoke(&PGFunc, jit::imm(FuncCallInfo->flinfo->fn_addr),
                   jit::FuncSignatureT<Datum, FunctionCallInfo>());
      PGFunc->setArg(0, FuncCallInfo);
      PGFunc->setRet(0, RetValue);

      /* Write result values. */
      Jitcc.mov(v_Resvalue, RetValue);
      x86::Gp FuncCallInfoIsNull = Jitcc.newInt8();
      Jitcc.mov(FuncCallInfoIsNull, v_FuncCallInfoIsNull);
      Jitcc.mov(v_Resnull, FuncCallInfoIsNull);

      break;
    }

    case EEOP_QUAL: {
      jit::Label HandleNullOrFalse = Jitcc.newLabel();

      x86::Gp Resvalue = Jitcc.newUIntPtr(), Resnull = Jitcc.newInt8();
      Jitcc.mov(Resvalue, v_Resvalue);
      Jitcc.mov(Resnull, v_Resnull);

      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.je(HandleNullOrFalse);

      Jitcc.cmp(Resvalue, jit::imm(0));
      Jitcc.je(HandleNullOrFalse);

      Jitcc.jmp(Opblocks[OpIndex + 1]);

      /* Handling null or false. */
      Jitcc.bind(HandleNullOrFalse);

      /* Set resnull and resvalue to false. */
      Jitcc.mov(v_Resvalue, jit::imm(0));
      Jitcc.mov(v_Resnull, jit::imm(0));

      Jitcc.jmp(Opblocks[Op->d.qualexpr.jumpdone]);

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

  instr_time CodeEmissionStartTime, CodeEmissionEndTime;
  ExprStateEvalFunc EvalFunc;
  INSTR_TIME_SET_CURRENT(CodeEmissionStartTime);
  jit::Error err = Runtime.add(&EvalFunc, &Code);
  INSTR_TIME_SET_CURRENT(CodeEmissionEndTime);
  INSTR_TIME_ACCUM_DIFF(Context->base.instr.emission_counter,
                        CodeEmissionEndTime, CodeEmissionStartTime);
  if (err) {
    ereport(LOG,
            (errmsg("Jit failed: %s", jit::DebugUtils::errorAsString(err))));
    return false;
  }

  {
    MemoryContext OldContext = MemoryContextSwitchTo(TopMemoryContext);
    State->evalfunc = ExecCompiledExpr;
    State->evalfunc_private = (void *)EvalFunc;
    Context->funcs = lappend(Context->funcs, (void *)EvalFunc);
    Context->base.instr.created_functions++;
    MemoryContextSwitchTo(OldContext);
  }

  INSTR_TIME_SET_CURRENT(CodeGenEndTime);
  INSTR_TIME_ACCUM_DIFF(Context->base.instr.generation_counter, CodeGenEndTime,
                        CodeGenStartTime);

  return true;
}

void _PG_jit_provider_init(JitProviderCallbacks *cb) {
  cb->reset_after_error = JitResetAfterError;
  cb->release_context = JitReleaseContext;
  cb->compile_expr = JitCompileExpr;
}

#ifdef __cplusplus
}
#endif
