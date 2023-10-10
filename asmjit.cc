#if __cplusplus > 199711L
#define register // Deprecated in C++11.
#endif           // #if __cplusplus > 199711L

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

static void JitResetAfterError(void) {
  elog(NOTICE, "TODO: reset after error");
}

static void JitReleaseContext(JitContext *Ctx) {
  AsmJitContext *Context = (AsmJitContext *)Ctx;
  ListCell *cell;

  foreach (cell, Context->funcs) {
    ExprStateEvalFunc EvalFunc = (ExprStateEvalFunc)lfirst(cell);
    Runtime.release(EvalFunc);
  }
  Context->funcs = NIL;
}

static void JitInitializeSession(void) {
  if (JitSessionInitialized)
    return;

  elog(NOTICE, "Jit initialize session");

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

static pg_attribute_always_inline void
ExecAggPlainTransByVal(AggState *aggstate, AggStatePerTrans pertrans,
                       AggStatePerGroup pergroup, ExprContext *aggcontext,
                       int setno) {
  FunctionCallInfo fcinfo = pertrans->transfn_fcinfo;
  MemoryContext oldContext;
  Datum newVal;

  /* cf. select_current_set() */
  aggstate->curaggcontext = aggcontext;
  aggstate->current_set = setno;

  /* set up aggstate->curpertrans for AggGetAggref() */
  aggstate->curpertrans = pertrans;

  /* invoke transition function in per-tuple context */
  oldContext =
      MemoryContextSwitchTo(aggstate->tmpcontext->ecxt_per_tuple_memory);

  fcinfo->args[0].value = pergroup->transValue;
  fcinfo->args[0].isnull = pergroup->transValueIsNull;
  fcinfo->isnull = false; /* just in case transfn doesn't set it */

  newVal = FunctionCallInvoke(fcinfo);

  pergroup->transValue = newVal;
  pergroup->transValueIsNull = fcinfo->isnull;

  MemoryContextSwitchTo(oldContext);
}

void InterpretAggPlainTransStrictByVal(ExprState *State, ExprEvalStep *Op) {
  AggState *aggstate = castNode(AggState, State->parent);
  AggStatePerTrans pertrans = Op->d.agg_trans.pertrans;
  AggStatePerGroup pergroup =
      &aggstate->all_pergroups[Op->d.agg_trans.setoff][Op->d.agg_trans.transno];

  Assert(pertrans->transtypeByVal);

  if (pergroup->noTransValue) {
    /* If transValue has not yet been initialized, do so now. */
    ExecAggInitGroup(aggstate, pertrans, pergroup, Op->d.agg_trans.aggcontext);
    /* copied trans value from input, done this round */
  } else if (likely(!pergroup->transValueIsNull)) {
    /* invoke transition function, unless prevented by strictness */
    ExecAggPlainTransByVal(aggstate, pertrans, pergroup,
                           Op->d.agg_trans.aggcontext, Op->d.agg_trans.setno);
  }
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

  x86::Gp expressionp = Jitcc.newUIntPtr("expression"),
          econtextp = Jitcc.newUIntPtr("econtext"),
          isnullp = Jitcc.newUIntPtr("isnull");

  JittedFunc->setArg(0, expressionp);
  JittedFunc->setArg(1, econtextp);
  JittedFunc->setArg(2, isnullp);

  /*
   * expression->resvalue and expression->resnull.
   */
  x86::Mem StateResvalue = x86::ptr(expressionp, offsetof(ExprState, resvalue),
                                    sizeof(Datum)),
           StateResnull = x86::ptr(expressionp, offsetof(ExprState, resnull),
                                   sizeof(bool));

  /*
   * econtext->ecxt_scantuple, econtext->ecxt_innertuple and
   * econtext->ecxt_outertuple.
   */
  x86::Mem EContextScantuple =
               x86::ptr(econtextp, offsetof(ExprContext, ecxt_scantuple),
                        sizeof(TupleTableSlot *)),
           EContextInnertuple =
               x86::ptr(econtextp, offsetof(ExprContext, ecxt_innertuple),
                        sizeof(TupleTableSlot *)),
           EContextOutertuple =
               x86::ptr(econtextp, offsetof(ExprContext, ecxt_outertuple),
                        sizeof(TupleTableSlot *));

  /*
   * econtext->ecxt_scantuple->tts_values, econtext->ext_scantuple->tts_isnull,
   */
  x86::Gp ScantupleAddr = Jitcc.newUIntPtr(),
          InnertupleAddr = Jitcc.newUIntPtr(),
          OutertupleAddr = Jitcc.newUIntPtr();
  Jitcc.mov(ScantupleAddr, EContextScantuple);
  Jitcc.mov(InnertupleAddr, EContextInnertuple);
  Jitcc.mov(OutertupleAddr, EContextOutertuple);
  x86::Mem ScantupleValues =
               x86::ptr(ScantupleAddr, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           ScantupleIsnulls =
               x86::ptr(ScantupleAddr, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *)),
           InnertupleValues =
               x86::ptr(InnertupleAddr, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           InnertupleIsnulls =
               x86::ptr(InnertupleAddr, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *)),
           OutertupleValues =
               x86::ptr(OutertupleAddr, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
           OutertupleIsnulls =
               x86::ptr(OutertupleAddr, offsetof(TupleTableSlot, tts_isnull),
                        sizeof(bool *));

  x86::Mem ParentAddrPtr =
      x86::ptr(expressionp, offsetof(ExprState, parent), sizeof(PlanState *));
  x86::Gp ParentAddr = Jitcc.newUIntPtr();
  Jitcc.mov(ParentAddr, ParentAddrPtr);

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
    x86::Mem ResvaluePtr = x86::ptr(ResvalueAddr, 0, sizeof(Datum)),
             ResnullPtr = x86::ptr(ResnullAddr, 0, sizeof(bool));

    switch (Opcode) {
    case EEOP_DONE: {
      /* Load expression->resvalue and expression->resnull */
      x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
              TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, StateResvalue);
      Jitcc.mov(TempStateResnull, StateResnull);

      /* *isnull = expression->resnull */
      x86::Mem IsNull = x86::ptr(isnullp, 0, sizeof(bool));
      Jitcc.mov(IsNull, TempStateResnull);

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

      x86::Mem Slot =
          Opcode == EEOP_INNER_FETCHSOME
              ? EContextInnertuple
              : (Opcode == EEOP_OUTER_FETCHSOME ? EContextOutertuple
                                                : EContextScantuple);

      /* Compute the address of Slot->tts_nvalid */
      x86::Gp SlotAddr = Jitcc.newUIntPtr();
      Jitcc.mov(SlotAddr, Slot);
      x86::Mem TtsNvalidPtr = x86::ptr(
          SlotAddr, offsetof(TupleTableSlot, tts_nvalid), sizeof(AttrNumber));
      x86::Gp TtsNvalid = Jitcc.newInt16();
      Jitcc.mov(TtsNvalid, TtsNvalidPtr);

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
      x86::Mem SlotValues =
          Opcode == EEOP_INNER_VAR
              ? InnertupleValues
              : (Opcode == EEOP_OUTER_VAR ? OutertupleValues : ScantupleValues);
      x86::Mem SlotIsnulls =
          Opcode == EEOP_INNER_VAR
              ? InnertupleIsnulls
              : (Opcode == EEOP_OUTER_VAR ? OutertupleIsnulls
                                          : ScantupleIsnulls);
      x86::Gp SlotValuesAddr = Jitcc.newUIntPtr(),
              SlotIsnullsAddr = Jitcc.newUIntPtr();
      Jitcc.mov(SlotValuesAddr, SlotValues);
      Jitcc.mov(SlotIsnullsAddr, SlotIsnulls);

      int Attrnum = Op->d.var.attnum;
      x86::Mem SlotValuePtr =
          x86::ptr(SlotValuesAddr, Attrnum * sizeof(Datum), sizeof(Datum));
      x86::Mem SlotIsnullPtr =
          x86::ptr(SlotIsnullsAddr, Attrnum * sizeof(bool), sizeof(bool));

      x86::Gp SlotValue = Jitcc.newUIntPtr(), SlotIsnull = Jitcc.newInt8();
      Jitcc.mov(SlotValue, SlotValuePtr);
      Jitcc.mov(SlotIsnull, SlotIsnullPtr);

      Jitcc.mov(ResvaluePtr, SlotValue);
      Jitcc.mov(ResnullPtr, SlotIsnull);

      break;
    }

    case EEOP_ASSIGN_TMP: {
      size_t ResultNum = Op->d.assign_tmp.resultnum;

      /* Load expression->resvalue and expression->resnull */
      x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
              TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, StateResvalue);
      Jitcc.mov(TempStateResnull, StateResnull);

      /*
       * Compute the addresses of expression->resultslot->tts_values and
       * expression->resultslot->tts_isnull
       */
      x86::Mem StateResultslot =
          x86::ptr(expressionp, offsetof(ExprState, resultslot),
                   sizeof(TupleTableSlot *));
      x86::Gp StateResultslotAddr = Jitcc.newUInt64();
      Jitcc.mov(StateResultslotAddr, StateResultslot);
      x86::Mem TtsValues = x86::ptr(StateResultslotAddr,
                                    offsetof(TupleTableSlot, tts_values),
                                    sizeof(Datum *)),
               TtsIsnulls = x86::ptr(StateResultslotAddr,
                                     offsetof(TupleTableSlot, tts_isnull),
                                     sizeof(bool *));

      /*
       * Compute the addresses of
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      x86::Gp TtsValueAddr = Jitcc.newUIntPtr(),
              TtsIsnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(TtsValueAddr, TtsValues);
      Jitcc.mov(TtsIsnullAddr, TtsIsnulls);
      x86::Mem TtsValue = x86::ptr(TtsValueAddr, sizeof(Datum) * ResultNum,
                                   sizeof(Datum)),
               TtsIsnull = x86::ptr(TtsIsnullAddr, sizeof(bool) * ResultNum,
                                    sizeof(bool));

      /*
       * Store expression->resvalue and expression->resnull to
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      Jitcc.mov(TtsValue, TempStateResvalue);
      Jitcc.mov(TtsIsnull, TempStateResnull);

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
      Jitcc.mov(ResvaluePtr, ConstVal);
      Jitcc.mov(ResnullPtr, ConstNull);

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
        Jitcc.mov(ResnullPtr, jit::imm(1));
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
                   jit::FuncSignatureT<Datum, FunctionCallInfo>());
      PGFunc->setArg(0, FuncCallInfo);
      PGFunc->setRet(0, RetValue);

      /* Write result values. */
      Jitcc.mov(ResvaluePtr, RetValue);
      x86::Gp FuncCallInfoIsNull = Jitcc.newInt8();
      Jitcc.mov(FuncCallInfoIsNull, FuncCallInfoIsNullPtr);
      Jitcc.mov(ResnullPtr, FuncCallInfoIsNull);

      break;
    }

    case EEOP_QUAL: {
      jit::Label HandleNullOrFalse = Jitcc.newLabel();

      x86::Gp Resvalue = Jitcc.newUIntPtr(), Resnull = Jitcc.newInt8();
      Jitcc.mov(Resvalue, ResvaluePtr);
      Jitcc.mov(Resnull, ResnullPtr);

      Jitcc.cmp(Resnull, jit::imm(1));
      Jitcc.je(HandleNullOrFalse);

      Jitcc.cmp(Resvalue, jit::imm(0));
      Jitcc.je(HandleNullOrFalse);

      Jitcc.jmp(Opblocks[OpIndex + 1]);

      /* Handling null or false. */
      Jitcc.bind(HandleNullOrFalse);

      /* Set resnull and resvalue to false. */
      Jitcc.mov(ResvaluePtr, jit::imm(0));
      Jitcc.mov(ResnullPtr, jit::imm(0));

      break;
    }

    case EEOP_AGGREF: {
      int AggNo = Op->d.aggref.aggno;

      x86::Mem AggValues0AddrPtr =
                   x86::ptr(econtextp, offsetof(ExprContext, ecxt_aggvalues),
                            sizeof(Datum *)),
               AggNulls0AddrPtr =
                   x86::ptr(econtextp, offsetof(ExprContext, ecxt_aggnulls),
                            sizeof(bool *));
      x86::Gp AggValues0Addr = Jitcc.newUIntPtr(),
              AggNulls0Addr = Jitcc.newUIntPtr();
      Jitcc.mov(AggValues0Addr, AggValues0AddrPtr);
      Jitcc.mov(AggNulls0Addr, AggNulls0AddrPtr);

      x86::Mem AggValuePtr = x86::ptr(AggValues0Addr, AggNo * sizeof(Datum),
                                      sizeof(Datum)),
               AggNullPtr =
                   x86::ptr(AggNulls0Addr, AggNo * sizeof(bool), sizeof(bool));
      x86::Gp Value = Jitcc.newUIntPtr(), Null = Jitcc.newInt8();
      Jitcc.mov(Value, AggValuePtr);
      Jitcc.mov(Null, AggNullPtr);

      /*
       * *op->resvalue = econtext->ecxt_aggvalues[aggno];
       * *op->resnull = econtext->ecxt_aggnulls[aggno];
       */
      Jitcc.mov(ResvaluePtr, Value);
      Jitcc.mov(ResnullPtr, Null);

      break;
    }

    case EEOP_AGG_STRICT_INPUT_CHECK_ARGS:
    case EEOP_AGG_STRICT_INPUT_CHECK_NULLS: {
      int NArgs = Op->d.agg_strict_input_check.nargs;
      NullableDatum *Args = Op->d.agg_strict_input_check.args;
      bool *Nulls = Op->d.agg_strict_input_check.nulls;
      int Jumpnull = Op->d.agg_strict_input_check.jumpnull;

      Assert(nargs > 0);

      x86::Gp Args0Addr = Jitcc.newUIntPtr(), Nulls0Addr = Jitcc.newUIntPtr();
      Jitcc.mov(Args0Addr, Args);
      Jitcc.mov(Nulls0Addr, Nulls);

      /* Strict function, check for NULL arguments. */
      for (int ArgIndex = 0; ArgIndex < NArgs; ++ArgIndex) {
        x86::Mem ArgIsNullPtr =
            Opcode == EEOP_AGG_STRICT_INPUT_CHECK_ARGS
                ? x86::ptr(Args0Addr,
                           ArgIndex * sizeof(NullableDatum) +
                               offsetof(NullableDatum, isnull),
                           sizeof(bool))
                : x86::ptr(Nulls0Addr, ArgIndex * sizeof(bool), sizeof(bool));
        x86::Gp ArgIsNull = Jitcc.newInt8();
        Jitcc.mov(ArgIsNull, ArgIsNullPtr);
        Jitcc.cmp(ArgIsNull, jit::imm(1));
        Jitcc.je(Opblocks[Jumpnull]);
      }
      break;
    }

    case EEOP_AGG_PLAIN_TRANS_STRICT_BYVAL: {
      /*
       * I'm lazy :p
       */
      jit::InvokeNode *InterpretAggPlainTransStrictByValFunc;
      Jitcc.invoke(&InterpretAggPlainTransStrictByValFunc,
                   InterpretAggPlainTransStrictByVal,
                   jit::FuncSignatureT<void, ExprState *, ExprEvalStep *>());
      InterpretAggPlainTransStrictByValFunc->setArg(0, expressionp);
      InterpretAggPlainTransStrictByValFunc->setArg(1, jit::imm(Op));
      break;
    }

    case EEOP_LAST: {
      Assert(false);
      break;
    }

    default:
      ereport(NOTICE,
              (errmsg("Jit for operator (%d) is not supported", Opcode)));
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
    ereport(ERROR,
            (errmsg("Jit failed: %s", jit::DebugUtils::errorAsString(err))));
    return false;
  }

  {
    State->evalfunc = ExecCompiledExpr;
    State->evalfunc_private = (void *)EvalFunc;
    Context->funcs = lappend(Context->funcs, (void *)EvalFunc);
    Context->base.instr.created_functions++;
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
