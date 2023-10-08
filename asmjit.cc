#if __cplusplus > 199711L
#define register // Deprecated in C++11.
#endif           // #if __cplusplus > 199711L

#include <asmjit/asmjit.h>

namespace jit = asmjit;

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
  jit::x86::Compiler Jitcc(&Code);

  /*
   * Datum ExprStateEvalFunc(struct ExprState *expression,
   *                         struct ExprContext *econtext,
   *                         bool *isNull);
   */
  jit::FuncNode *JittedFunc = Jitcc.addFunc(
      jit::FuncSignatureT<Datum, ExprState *, ExprContext *, bool *>());

  jit::x86::Gp expressionp = Jitcc.newUIntPtr("expression"),
               econtextp = Jitcc.newUIntPtr("econtext"),
               isnullp = Jitcc.newUIntPtr("isnull");

  JittedFunc->setArg(0, expressionp);
  JittedFunc->setArg(1, econtextp);
  JittedFunc->setArg(2, isnullp);

  /*
   * expression->resvalue and expression->resnull.
   */
  jit::x86::Mem StateResvalue = jit::x86::ptr(
                    expressionp, offsetof(ExprState, resvalue), sizeof(Datum)),
                StateResnull = jit::x86::ptr(
                    expressionp, offsetof(ExprState, resnull), sizeof(bool));

  /*
   * econtext->ecxt_scantuple, econtext->ecxt_innertuple and
   * econtext->ecxt_outertuple.
   */
  jit::x86::Mem EContextScantuple = jit::x86::ptr(
                    econtextp, offsetof(ExprContext, ecxt_scantuple),
                    sizeof(TupleTableSlot *)),
                EContextInnertuple = jit::x86::ptr(
                    econtextp, offsetof(ExprContext, ecxt_innertuple),
                    sizeof(TupleTableSlot *)),
                EContextOutertuple = jit::x86::ptr(
                    econtextp, offsetof(ExprContext, ecxt_outertuple),
                    sizeof(TupleTableSlot *));

  /*
   * econtext->ecxt_scantuple->tts_values, econtext->ext_scantuple->tts_isnull,
   */
  jit::x86::Gp ScantupleAddr = Jitcc.newUIntPtr(),
               InnertupleAddr = Jitcc.newUIntPtr(),
               OutertupleAddr = Jitcc.newUIntPtr();
  Jitcc.mov(ScantupleAddr, EContextScantuple);
  Jitcc.mov(InnertupleAddr, EContextInnertuple);
  Jitcc.mov(OutertupleAddr, EContextOutertuple);
  jit::x86::Mem
      ScantupleValues = jit::x86::ptr(
          ScantupleAddr, offsetof(TupleTableSlot, tts_values), sizeof(Datum *)),
      ScantupleIsnulls = jit::x86::ptr(
          ScantupleAddr, offsetof(TupleTableSlot, tts_isnull), sizeof(bool *)),
      InnertupleValues =
          jit::x86::ptr(InnertupleAddr, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
      InnertupleIsnulls = jit::x86::ptr(
          InnertupleAddr, offsetof(TupleTableSlot, tts_isnull), sizeof(bool *)),
      OutertupleValues =
          jit::x86::ptr(OutertupleAddr, offsetof(TupleTableSlot, tts_values),
                        sizeof(Datum *)),
      OutertupleIsnulls = jit::x86::ptr(
          OutertupleAddr, offsetof(TupleTableSlot, tts_isnull), sizeof(bool *));

  jit::Label *Opblocks =
      (jit::Label *)palloc(State->steps_len * sizeof(jit::Label));
  for (int OpIndex = 0; OpIndex < State->steps_len; ++OpIndex)
    Opblocks[OpIndex] = Jitcc.newLabel();

  for (int OpIndex = 0; OpIndex < State->steps_len; ++OpIndex) {
    ExprEvalStep *Op = &State->steps[OpIndex];
    ExprEvalOp Opcode = ExecEvalStepOp(State, Op);

    Jitcc.bind(Opblocks[OpIndex]);

    switch (Opcode) {
    case EEOP_DONE: {
      /* Load expression->resvalue and expression->resnull */
      jit::x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
                   TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, StateResvalue);
      Jitcc.mov(TempStateResnull, StateResnull);

      /* *isnull = expression->resnull */
      jit::x86::Mem IsNull = jit::x86::ptr(isnullp, 0, sizeof(bool));
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

      jit::x86::Mem Slot =
          Opcode == EEOP_INNER_FETCHSOME
              ? EContextInnertuple
              : (Opcode == EEOP_OUTER_FETCHSOME ? EContextOutertuple
                                                : EContextScantuple);

      /* Compute the address of Slot->tts_nvalid */
      jit::x86::Gp SlotAddr = Jitcc.newUIntPtr();
      Jitcc.mov(SlotAddr, Slot);
      jit::x86::Mem TtsNvalidPtr = jit::x86::ptr(
          SlotAddr, offsetof(TupleTableSlot, tts_nvalid), sizeof(AttrNumber));
      jit::x86::Gp TtsNvalid = Jitcc.newInt16();
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

      /* Jump to the next op block. */
      Jitcc.jmp(Opblocks[OpIndex + 1]);
      break;
    }

    case EEOP_INNER_VAR:
    case EEOP_OUTER_VAR:
    case EEOP_SCAN_VAR: {
      jit::x86::Mem SlotValues =
          Opcode == EEOP_INNER_VAR
              ? InnertupleValues
              : (Opcode == EEOP_OUTER_VAR ? OutertupleValues : ScantupleValues);
      jit::x86::Mem SlotIsnulls =
          Opcode == EEOP_INNER_VAR
              ? InnertupleIsnulls
              : (Opcode == EEOP_OUTER_VAR ? OutertupleIsnulls
                                          : ScantupleIsnulls);
      jit::x86::Gp SlotValuesAddr = Jitcc.newUIntPtr(),
                   SlotIsnullsAddr = Jitcc.newUIntPtr();
      Jitcc.mov(SlotValuesAddr, SlotValues);
      Jitcc.mov(SlotIsnullsAddr, SlotIsnulls);

      int Attrnum = Op->d.var.attnum;
      jit::x86::Mem SlotValuePtr =
          jit::x86::ptr(SlotValuesAddr, Attrnum * sizeof(Datum), sizeof(Datum));
      jit::x86::Mem SlotIsnullPtr =
          jit::x86::ptr(SlotIsnullsAddr, Attrnum * sizeof(bool), sizeof(bool));

      jit::x86::Gp SlotValue = Jitcc.newUIntPtr(), SlotIsnull = Jitcc.newInt8();
      Jitcc.mov(SlotValue, SlotValuePtr);
      Jitcc.mov(SlotIsnull, SlotIsnullPtr);

      /* Compute the addresses of op->resvalue and op->resnull */
      jit::x86::Gp ResvalueAddr = Jitcc.newUIntPtr();
      jit::x86::Gp ResnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(ResvalueAddr, jit::imm(Op->resvalue));
      Jitcc.mov(ResnullAddr, jit::imm(Op->resnull));
      jit::x86::Mem Resvalue = jit::x86::ptr(ResvalueAddr, 0, sizeof(Datum));
      jit::x86::Mem Resnull = jit::x86::ptr(ResnullAddr, 0, sizeof(bool));

      Jitcc.mov(Resvalue, SlotValue);
      Jitcc.mov(Resnull, SlotIsnull);

      /* Jump to the next op block. */
      Jitcc.jmp(Opblocks[OpIndex + 1]);
      break;
    }

    case EEOP_ASSIGN_TMP: {
      size_t ResultNum = Op->d.assign_tmp.resultnum;

      /* Load expression->resvalue and expression->resnull */
      jit::x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
                   TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, StateResvalue);
      Jitcc.mov(TempStateResnull, StateResnull);

      /*
       * Compute the addresses of expression->resultslot->tts_values and
       * expression->resultslot->tts_isnull
       */
      jit::x86::Mem StateResultslot =
          jit::x86::ptr(expressionp, offsetof(ExprState, resultslot),
                        sizeof(TupleTableSlot *));
      jit::x86::Gp StateResultslotAddr = Jitcc.newUInt64();
      Jitcc.mov(StateResultslotAddr, StateResultslot);
      jit::x86::Mem TtsValues = jit::x86::ptr(
                        StateResultslotAddr,
                        offsetof(TupleTableSlot, tts_values), sizeof(Datum *)),
                    TtsIsnulls = jit::x86::ptr(
                        StateResultslotAddr,
                        offsetof(TupleTableSlot, tts_isnull), sizeof(bool *));

      /*
       * Compute the addresses of
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      jit::x86::Gp TtsValueAddr = Jitcc.newUIntPtr(),
                   TtsIsnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(TtsValueAddr, TtsValues);
      Jitcc.mov(TtsIsnullAddr, TtsIsnulls);
      jit::x86::Mem TtsValue = jit::x86::ptr(
                        TtsValueAddr, sizeof(Datum) * ResultNum, sizeof(Datum)),
                    TtsIsnull = jit::x86::ptr(
                        TtsIsnullAddr, sizeof(bool) * ResultNum, sizeof(bool));

      /*
       * Store expression->resvalue and expression->resnull to
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      Jitcc.mov(TtsValue, TempStateResvalue);
      Jitcc.mov(TtsIsnull, TempStateResnull);

      /* Jump to the next op block. */
      Jitcc.jmp(Opblocks[OpIndex + 1]);
      break;
    }
    case EEOP_CONST: {
      jit::x86::Gp ConstVal = Jitcc.newUIntPtr(), ConstNull = Jitcc.newInt8();

      Jitcc.mov(ConstVal, jit::imm(Op->d.constval.value));
      Jitcc.mov(ConstNull, jit::imm(Op->d.constval.isnull));

      /* Compute the addresses of op->resvalue and op->resnull */
      jit::x86::Gp ResvalueAddr = Jitcc.newUIntPtr();
      jit::x86::Gp ResnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(ResvalueAddr, jit::imm(Op->resvalue));
      Jitcc.mov(ResnullAddr, jit::imm(Op->resnull));
      jit::x86::Mem Resvalue = jit::x86::ptr(ResvalueAddr, 0, sizeof(Datum));
      jit::x86::Mem Resnull = jit::x86::ptr(ResnullAddr, 0, sizeof(bool));

      /*
       * Store Op->d.constval.value to Op->resvalue.
       * Store Op->d.constval.isnull to Op->resnull.
       */
      Jitcc.mov(Resvalue, ConstVal);
      Jitcc.mov(Resnull, ConstNull);

      /* Jump to the next op block. */
      Jitcc.jmp(Opblocks[OpIndex + 1]);
      break;
    }

    case EEOP_FUNCEXPR:
    case EEOP_FUNCEXPR_STRICT: {
      FunctionCallInfo FuncCallInfo = Op->d.func.fcinfo_data;

      /*
       * Compute the addresses of Op->resvalue and Op->resnull.
       */
      jit::x86::Gp ResvalueAddr = Jitcc.newUIntPtr(),
                   ResnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(ResvalueAddr, jit::imm(Op->resvalue));
      Jitcc.mov(ResnullAddr, jit::imm(Op->resnull));
      jit::x86::Mem ResvaluePtr = jit::x86::ptr(ResvalueAddr, 0, sizeof(Datum)),
                    ResnullPtr = jit::x86::ptr(ResnullAddr, 0, sizeof(bool));

      jit::x86::Gp FuncCallInfoAddr = Jitcc.newUIntPtr();
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
          jit::x86::Mem FuncCallInfoArgNIsNullPtr =
              jit::x86::ptr(FuncCallInfoAddr,
                            offsetof(FunctionCallInfoBaseData, args) +
                                ArgIndex * sizeof(NullableDatum) +
                                offsetof(NullableDatum, isnull),
                            sizeof(bool));
          jit::x86::Gp FuncCallInfoArgNIsNull = Jitcc.newInt8();
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
      jit::x86::Mem FuncCallInfoIsNullPtr = jit::x86::ptr(
          FuncCallInfoAddr, offsetof(FunctionCallInfoBaseData, isnull),
          sizeof(bool));
      Jitcc.mov(FuncCallInfoIsNullPtr, jit::imm(0));

      jit::InvokeNode *PGFunc;
      jit::x86::Gp RetValue = Jitcc.newUIntPtr();
      Jitcc.invoke(&PGFunc, jit::imm(FuncCallInfo->flinfo->fn_addr),
                   jit::FuncSignatureT<Datum, FunctionCallInfo>());
      PGFunc->setArg(0, FuncCallInfo);
      PGFunc->setRet(0, RetValue);

      /* Write result values. */
      Jitcc.mov(ResvaluePtr, RetValue);
      jit::x86::Gp FuncCallInfoIsNull = Jitcc.newInt8();
      Jitcc.mov(FuncCallInfoIsNull, FuncCallInfoIsNullPtr);
      Jitcc.mov(ResnullPtr, FuncCallInfoIsNull);

      /* Jump to the next op block. */
      Jitcc.jmp(Opblocks[OpIndex + 1]);
      break;
    }

    case EEOP_QUAL: {
      jit::Label HandleNullOrFalse = Jitcc.newLabel();

      jit::x86::Gp ResvalueAddr = Jitcc.newUIntPtr(),
                   ResnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(ResvalueAddr, jit::imm(Op->resvalue));
      Jitcc.mov(ResnullAddr, jit::imm(Op->resnull));
      jit::x86::Mem ResvaluePtr = jit::x86::ptr(ResvalueAddr, 0, sizeof(Datum)),
                    ResnullPtr = jit::x86::ptr(ResnullAddr, 0, sizeof(bool));

      jit::x86::Gp Resvalue = Jitcc.newUIntPtr(), Resnull = Jitcc.newInt8();
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

      /* Jump to the next op block. */
      Jitcc.jmp(Opblocks[OpIndex + 1]);
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
