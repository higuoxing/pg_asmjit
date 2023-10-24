#include "asmjit_common.h"

extern "C" {

PG_MODULE_MAGIC;

void _PG_jit_provider_init(JitProviderCallbacks *cb) {
  cb->reset_after_error = AsmJitResetAfterError;
  cb->release_context = AsmJitReleaseContext;
  cb->compile_expr = AsmJitCompileExpr;
}

}
