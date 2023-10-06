#if __cplusplus > 199711L
#define register // Deprecated in C++11.
#endif           // #if __cplusplus > 199711L

#include <asmjit/asmjit.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"

#include "executor/execExpr.h"
#include "fmgr.h"
#include "jit/jit.h"
#include "nodes/execnodes.h"
#include "nodes/pg_list.h"
#include "utils/memutils.h"
#include "utils/palloc.h"
#include "utils/resowner.h"
#include "utils/resowner_private.h"

PG_MODULE_MAGIC;

static bool JitSessionInitialized = false;
static asmjit::JitRuntime Runtime;

typedef struct AsmJitContext {
  JitContext base;
  List *funcs;
} AsmJitContext;

static void JitResetAfterError(void) { elog(NOTICE, "reset after error"); }

static void JitReleaseContext(JitContext *Ctx) {
  AsmJitContext *Context = (AsmJitContext *)Ctx;
  ListCell *cell;

  foreach (cell, Context->funcs) {
    ExprStateEvalFunc EvalFunc = (ExprStateEvalFunc)lfirst(cell);
    Runtime.release(EvalFunc);
  }
  Context->funcs = NIL;
  elog(NOTICE, "release context");
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

  asmjit::CodeHolder Code;
  Code.init(Runtime.environment(), Runtime.cpuFeatures());
  asmjit::x86::Compiler Jitcc(&Code);

  /*
   * Datum ExprStateEvalFunc(struct ExprState *expression,
   *                         struct ExprContext *econtext,
   *                         bool *isNull);
   */
  asmjit::FuncNode *JittedFunc = Jitcc.addFunc(
      asmjit::FuncSignatureT<Datum, ExprState *, ExprContext *, bool *>());

  asmjit::x86::Gp expressionp = Jitcc.newUIntPtr("expression"),
                  econtextp = Jitcc.newUIntPtr("econtext"),
                  isnullp = Jitcc.newUIntPtr("isnull");

  JittedFunc->setArg(0, expressionp);
  JittedFunc->setArg(1, econtextp);
  JittedFunc->setArg(2, isnullp);

  /*
   * expression->resvalue and expression->resnull.
   */
  asmjit::x86::Mem StateResvalue = asmjit::x86::ptr(
                       expressionp, offsetof(ExprState, resvalue),
                       sizeof(Datum)),
                   StateResnull = asmjit::x86::ptr(
                       expressionp, offsetof(ExprState, resnull), sizeof(bool));

  asmjit::Label *Opblocks =
      (asmjit::Label *)palloc(State->steps_len * sizeof(asmjit::Label));
  for (size_t OpIndex = 0; OpIndex < State->steps_len; ++OpIndex)
    Opblocks[OpIndex] = Jitcc.newLabel();

  for (size_t OpIndex = 0; OpIndex < State->steps_len; ++OpIndex) {
    ExprEvalStep *Op = &State->steps[OpIndex];
    ExprEvalOp Opcode = ExecEvalStepOp(State, Op);

    Jitcc.bind(Opblocks[OpIndex]);

    switch (Opcode) {
    case EEOP_DONE: {
      /* Load expression->resvalue and expression->resnull */
      asmjit::x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
                      TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, StateResvalue);
      Jitcc.mov(TempStateResnull, StateResnull);

      /* *isnull = expression->resnull */
      asmjit::x86::Mem IsNull = asmjit::x86::ptr(isnullp, 0, sizeof(bool));
      Jitcc.mov(IsNull, TempStateResnull);

      /* return expression->resvalue */
      Jitcc.ret(TempStateResvalue);
      Jitcc.endFunc();
      break;
    }
    case EEOP_ASSIGN_TMP: {
      size_t ResultNum = Op->d.assign_tmp.resultnum;

      /* Load expression->resvalue and expression->resnull */
      asmjit::x86::Gp TempStateResvalue = Jitcc.newUIntPtr(),
                      TempStateResnull = Jitcc.newInt8();
      Jitcc.mov(TempStateResvalue, StateResvalue);
      Jitcc.mov(TempStateResnull, StateResnull);

      /*
       * Compute the addresses of expression->resultslot->tts_values and
       * expression->resultslot->tts_isnull
       */
      asmjit::x86::Mem StateResultslot =
          asmjit::x86::ptr(expressionp, offsetof(ExprState, resultslot),
                           sizeof(TupleTableSlot *));
      asmjit::x86::Gp StateResultslotAddr = Jitcc.newUInt64();
      Jitcc.mov(StateResultslotAddr, StateResultslot);
      asmjit::x86::Mem TtsValues = asmjit::x86::ptr(
                           StateResultslotAddr,
                           offsetof(TupleTableSlot, tts_values),
                           sizeof(Datum *)),
                       TtsIsnulls = asmjit::x86::ptr(
                           StateResultslotAddr,
                           offsetof(TupleTableSlot, tts_isnull),
                           sizeof(bool *));
      asmjit::x86::Gp TtsValueAddr = Jitcc.newUIntPtr(),
                      TtsIsnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(TtsValueAddr, TtsValues);
      Jitcc.mov(TtsIsnullAddr, TtsIsnulls);
      asmjit::x86::Mem TtsValue = asmjit::x86::ptr(TtsValueAddr,
                                                   sizeof(Datum) * ResultNum,
                                                   sizeof(Datum)),
                       TtsIsnull = asmjit::x86::ptr(TtsIsnullAddr,
                                                    sizeof(bool) * ResultNum,
                                                    sizeof(bool));

      /*
       * Store expression->resvalue and expression->resnull to
       * expression->resultslot->tts_values[ResultNum] and
       * expression->resultslot->tts_isnull[ResultNum]
       */
      Jitcc.mov(TtsValue, TempStateResvalue);
      Jitcc.mov(TtsIsnull, TempStateResnull);

      Jitcc.jmp(Opblocks[OpIndex + 1]);
      break;
    }
    case EEOP_CONST: {
      asmjit::x86::Gp ConstVal = Jitcc.newUIntPtr(),
                      ConstNull = Jitcc.newInt8();

      Jitcc.mov(ConstVal, asmjit::imm(Op->d.constval.value));
      Jitcc.mov(ConstNull, asmjit::imm(Op->d.constval.isnull));

      /* Compute the addresses of op->resvalue and op->resnull */
      asmjit::x86::Gp ResvalueAddr = Jitcc.newUIntPtr();
      asmjit::x86::Gp ResnullAddr = Jitcc.newUIntPtr();
      Jitcc.mov(ResvalueAddr, asmjit::imm(Op->resvalue));
      Jitcc.mov(ResnullAddr, asmjit::imm(Op->resnull));
      asmjit::x86::Mem Resvalue =
          asmjit::x86::ptr(ResvalueAddr, 0, sizeof(Datum));
      asmjit::x86::Mem Resnull = asmjit::x86::ptr(ResnullAddr, 0, sizeof(bool));

      /*
       * Store Op->d.constval.value to Op->resvalue.
       * Store Op->d.constval.isnull to Op->resnull.
       */
      Jitcc.mov(Resvalue, ConstVal);
      Jitcc.mov(Resnull, ConstNull);

      Jitcc.jmp(Opblocks[OpIndex + 1]);
      break;
    }
    default:
      ereport(NOTICE,
              (errmsg("Jit for operator (%d) is not supported", Opcode)));
      return false;
    }
  }

  Jitcc.finalize();

  ExprStateEvalFunc EvalFunc;
  asmjit::Error err = Runtime.add(&EvalFunc, &Code);
  if (err) {
    ereport(NOTICE,
            (errmsg("Jit failed: %s", asmjit::DebugUtils::errorAsString(err))));
    return false;
  }

  {
    State->evalfunc = ExecCompiledExpr;
    State->evalfunc_private = (void *)EvalFunc;
    Context->funcs = lappend(Context->funcs, (void *)EvalFunc);
  }

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
