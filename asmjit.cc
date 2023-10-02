#include <asmjit/asmjit.h>

#ifdef __cplusplus
extern "C" {
#endif
  
#include "postgres.h"

#include "fmgr.h"
#include "jit/jit.h"
#include "nodes/execnodes.h"
#include "utils/memutils.h"
#include "utils/palloc.h"
#include "utils/resowner.h"
#include "utils/resowner_private.h"

PG_MODULE_MAGIC;

static bool AsmjitSessionInitialized = false;
static asmjit::JitRuntime AsmjitRuntime;
static asmjit::CodeHolder AsmjitCodeHolder;

typedef struct AsmJitContext {
  JitContext base;
} AsmJitContext;

static void AsmjitResetAfterError(void) { elog(LOG, "reset after error"); }

static void AsmjitReleaseContext(JitContext *Context) {
  elog(LOG, "release context");
}

static void AsmjitInitializeSession(void) {
  if (AsmjitSessionInitialized)
    return;

  AsmjitCodeHolder.init(AsmjitRuntime.environment(),
                        AsmjitRuntime.cpuFeatures());
  AsmjitSessionInitialized = true;
}

static AsmJitContext *AsmjitCreateContext(int JitFlags) {
  AsmjitInitializeSession();

  ResourceOwnerEnlargeJIT(CurrentResourceOwner);

  AsmJitContext *Context = (AsmJitContext *)MemoryContextAllocZero(
      TopMemoryContext, sizeof(AsmJitContext));
  Context->base.flags = JitFlags;

  /* ensure cleanup */
  Context->base.resowner = CurrentResourceOwner;
  ResourceOwnerRememberJIT(CurrentResourceOwner, PointerGetDatum(Context));

  return Context;
}

static bool AsmjitCompileExpr(ExprState *State) {
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
    Context = AsmjitCreateContext(Parent->state->es_jit_flags);
    Parent->state->es_jit = &Context->base;
  }

  return false;
}

void _PG_jit_provider_init(JitProviderCallbacks *cb) {
  cb->reset_after_error = AsmjitResetAfterError;
  cb->release_context = AsmjitReleaseContext;
  cb->compile_expr = AsmjitCompileExpr;
}

#ifdef __cplusplus
}
#endif
