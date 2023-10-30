#include "asmjit_common.h"

namespace x86 = jit::x86;

extern "C" {
TupleDeformFunc CompileTupleDeformFunc(JitContext *Ctx,
                                       jit::JitRuntime &Runtime, TupleDesc Desc,
                                       const TupleTableSlotOps *Ops,
                                       int Nattrs) {
  jit::CodeHolder Code;
  Code.init(Runtime.environment(), Runtime.cpuFeatures());
  x86::Compiler Jitcc(&Code);

  /* Codes go here... */

  TupleDeformFunc JittedFunc = nullptr;
  jit::Error Err = Runtime.add(&JittedFunc, &Code);
  if (Err) {
    ereport(LOG, (errmsg("Jit failed for tuple deforming: %s",
                         jit::DebugUtils::errorAsString(Err))));
    return nullptr;
  }

  return JittedFunc;
}
}
