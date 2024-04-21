#include "asmjit/core/compiler.h"
#include "asmjit/core/func.h"
#include "asmjit_common.h"

namespace x86 = asmjit::x86;
namespace jit = asmjit;

extern "C" {

TupleDeformingFunc CompileTupleDeformingFunc(AsmJitContext *Context,
                                             jit::JitRuntime &Runtime,
                                             TupleDesc Desc,
                                             const TupleTableSlotOps *TtsOps,
                                             int NAttrs) {
  /* virtual tuples never need deforming, so don't generate code */
  if (TtsOps == &TTSOpsVirtual)
    return nullptr;

  /* decline to JIT for slot types we don't know to handle */
  if (TtsOps != &TTSOpsHeapTuple && TtsOps != &TTSOpsBufferHeapTuple &&
      TtsOps != &TTSOpsMinimalTuple)
    return nullptr;

  /*
   * FIXME: Is any way to get rid of this code holder?
   */
  jit::CodeHolder Code;
  Code.init(Runtime.environment(), Runtime.cpuFeatures());
  x86::Compiler Jitcc(&Code);

  /*
   * void (*TupleDeformingFunc) (TupleTableSlot *);
   */
  jit::FuncNode *JittedDeformingFunc =
      Jitcc.addFunc(jit::FuncSignature::build<void, TupleTableSlot *>());
  x86::Gp SlotAddr = Jitcc.newUIntPtr();
  JittedDeformingFunc->setArg(0, SlotAddr);

  x86::Mem v_TtsOps = x86::ptr(SlotAddr, offsetof(TupleTableSlot, tts_ops),
                               sizeof(TupleTableSlotOps *));
  x86::Gp TtsOpsAddr = Jitcc.newUIntPtr();
  Jitcc.mov(TtsOpsAddr, v_TtsOps);

  x86::Mem v_GetSomeAttrs = x86::ptr(
      TtsOpsAddr, offsetof(TupleTableSlotOps, getsomeattrs), sizeof(uintptr_t));
  x86::Gp GetSomeAttrsAddr = Jitcc.newUIntPtr();
  Jitcc.mov(GetSomeAttrsAddr, v_GetSomeAttrs);
  jit::InvokeNode *GetSomeAttrsInvoke;
  Jitcc.invoke(&GetSomeAttrsInvoke, GetSomeAttrsAddr,
               jit::FuncSignature::build<void, TupleTableSlot *, int>());
  GetSomeAttrsInvoke->setArg(0, SlotAddr);
  GetSomeAttrsInvoke->setArg(1, jit::imm(NAttrs));

  jit::Label Out = Jitcc.newLabel();

  x86::Mem v_TtsNValid = x86::ptr(
      SlotAddr, offsetof(TupleTableSlot, tts_nvalid), sizeof(AttrNumber));
  x86::Gp TtsNValid = Jitcc.newInt32();
  Jitcc.movzx(TtsNValid, v_TtsNValid);

  Jitcc.cmp(TtsNValid, jit::imm(NAttrs));
  Jitcc.jge(Out);

  jit::InvokeNode *GetMissingAttrsInvoke;
  Jitcc.invoke(&GetMissingAttrsInvoke, jit::imm(slot_getmissingattrs),
               jit::FuncSignature::build<void, TupleTableSlot *, int, int>());
  GetMissingAttrsInvoke->setArg(0, SlotAddr);
  GetMissingAttrsInvoke->setArg(1, TtsNValid);
  GetMissingAttrsInvoke->setArg(2, jit::imm(NAttrs));

  Jitcc.mov(v_TtsNValid, jit::imm(NAttrs));

  Jitcc.bind(Out);
  Jitcc.ret();
  Jitcc.endFunc();
  Jitcc.finalize();

  return (TupleDeformingFunc)EmitJittedFunction(Context, Code);
}
}
