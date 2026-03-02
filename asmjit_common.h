#ifndef _PG_ASMJIT_H_
#define _PG_ASMJIT_H_

#include <asmjit/asmjit.h>

extern "C" {
#if __cplusplus > 199711L
#define register
#endif

#include "postgres.h"

#include "access/htup_details.h"
#include "access/tupdesc_details.h"
#include "catalog/pg_attribute.h"
#include "executor/execExpr.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "jit/jit.h"
#include "nodes/execnodes.h"
#include "nodes/pg_list.h"
#include "portability/instr_time.h"
#include "storage/ipc.h"
#include "utils/expandeddatum.h"
#include "utils/memutils.h"
#include "utils/palloc.h"
#include "utils/resowner.h"

namespace jit = asmjit;

typedef struct AsmJitContext {
  JitContext base;
  ResourceOwner resowner;
  List *funcs;
} AsmJitContext;

extern bool AsmJitCompileExpr(ExprState *State);
extern void AsmJitReleaseContext(JitContext *Ctx);
extern void AsmJitResetAfterError(void);

extern void *EmitJittedFunction(AsmJitContext *Context, jit::CodeHolder &Code);

typedef void (*TupleDeformingFunc)(TupleTableSlot *);
TupleDeformingFunc CompileTupleDeformingFunc(AsmJitContext *Context,
                                             jit::JitRuntime &Runtime,
                                             TupleDesc Desc,
                                             const TupleTableSlotOps *TtsOps,
                                             int NAttrs);
}

/*
 * Include the architecture-specific emit helpers.  These define:
 *   - namespace arch (= asmjit::x86 or asmjit::a64)
 *   - All TYPES_INFO-based emit_load/emit_store helpers
 *   - EmitLoadConst* helpers
 *   - EmitLoadFromArray / EmitStoreToArray / EmitLoadFromFlexibleArray /
 *     EmitStoreToFlexibleArray
 *   - LoadFuncArgNull / LoadFuncArgValue / StoreFuncArgNull / StoreFuncArgValue
 *   - EmitCondJumpEQ/NE/GE/LE/GT/RegEQ/RegNE, EmitJump
 *   - EmitSetEQ/LT/LE/GT/GE
 *   - EmitZero, EmitBitwiseOr/And/AndImm/Xor
 *   - EmitAddImm/AddReg, EmitInc
 *   - EmitShlImm/ShrImm
 *   - EmitZeroExtend8/16/32to64, EmitSignExtend8to64/16to32/32to64
 */
#if defined(__aarch64__) || defined(_M_ARM64)
#include "asmjit_a64_common.h"
#else
#include "asmjit_x86_common.h"
#endif

/*
 * JIT_FN_PTR(cc, fn) — architecture-aware function-address operand.
 *
 * On x86/x86-64, AsmJit can encode CALL/JMP with a 32-bit PC-relative
 * displacement and handles absolute-to-relative relocations automatically,
 * so passing jit::imm(fn) directly to invoke() is fine.
 *
 * On AArch64, the BLR (branch-link-register) instruction only accepts a
 * 64-bit GP register operand.  AsmJit's a64::Compiler invoke() uses kIdBlr
 * for all calls, so if the target operand is an Imm the assembler rejects it
 * with kInvalidInstruction.  We therefore pre-load the function address into
 * a scratch GP register and pass that register to invoke().
 */
#if defined(__aarch64__) || defined(_M_ARM64)
static inline arch::Gp JitFnAddr(arch::Compiler &cc, void *fn) {
  arch::Gp r = cc.new_gp_ptr("fn_addr");
  cc.mov(r, asmjit::imm((uintptr_t)fn));
  return r;
}
#define JIT_FN_PTR(cc_, fn_) JitFnAddr((cc_), (void *)(uintptr_t)(fn_))
#else
/* x86: pass the address as an immediate; AsmJit handles the relocation. */
#define JIT_FN_PTR(cc_, fn_) jit::imm(fn_)
#endif

#endif /* _PG_ASMJIT_H_ */
