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

#define TYPES_INFO(struct_type, member_type, member_name, reg_type)            \
  static inline jit::x86::Gp emit_load_##member_name##_from_##struct_type(     \
      jit::x86::Compiler &cc, jit::x86::Gp &object_addr) {                     \
    jit::x86::Mem member_ptr = jit::x86::ptr(                                  \
        object_addr, offsetof(struct_type, member_name), sizeof(member_type)); \
    jit::x86::Gp member = cc.new##reg_type(#struct_type "_" #member_name);     \
    cc.mov(member, member_ptr);                                                \
    return member;                                                             \
  }                                                                            \
  template <typename Op>                                                       \
  static inline void emit_store_##member_name##_to_##struct_type(              \
      jit::x86::Compiler &cc, jit::x86::Gp &object_addr, Op &&val) {           \
    jit::x86::Mem member_ptr = jit::x86::ptr(                                  \
        object_addr, offsetof(struct_type, member_name), sizeof(member_type)); \
    cc.mov(member_ptr, val);                                                   \
  }
#include "jit_types_info.inc"
#undef TYPES_INFO

#define LOAD_STORE_CONST(CType, JitType)                                       \
  static inline jit::x86::Gp EmitLoadConst##JitType(                           \
      jit::x86::Compiler &cc, const char *fmt, CType c) {                      \
    jit::x86::Gp JitReg = cc.new##JitType(fmt);                                \
    cc.mov(JitReg, jit::imm(c));                                               \
    return JitReg;                                                             \
  }
LOAD_STORE_CONST(uint8, UInt8)
LOAD_STORE_CONST(int8, Int8)
LOAD_STORE_CONST(uint32, UInt32)
LOAD_STORE_CONST(int32, Int32)
LOAD_STORE_CONST(void *, UIntPtr)
LOAD_STORE_CONST(intptr_t, IntPtr)
LOAD_STORE_CONST(uint64, UInt64)
LOAD_STORE_CONST(int64, Int64)
#undef LOAD_STORE_CONST

static inline void EmitLoadFromArray(jit::x86::Compiler &cc,
                                     jit::x86::Gp &Array, size_t Index,
                                     jit::x86::Gp &Elem, size_t ElemSize) {
  jit::x86::Mem ElemPtr = jit::x86::ptr(Array, Index * ElemSize, ElemSize);
  cc.mov(Elem, ElemPtr);
}

template <typename T>
static inline void EmitStoreToArray(jit::x86::Compiler &cc, jit::x86::Gp &Array,
                                    size_t Index, const T &Elem,
                                    size_t ElemSize) {
  jit::x86::Mem ElemPtr = jit::x86::ptr(Array, Index * ElemSize, ElemSize);
  cc.mov(ElemPtr, Elem);
}

/* TODO: Combine with EmitLoadFromArray. */
static inline void EmitLoadFromFlexibleArray(jit::x86::Compiler &cc,
                                             jit::x86::Gp &ObjectAddr,
                                             size_t ArrayOff, size_t Index,
                                             jit::x86::Gp &Elem,
                                             size_t ElemSize) {
  jit::x86::Mem ElemPtr =
      jit::x86::ptr(ObjectAddr, ArrayOff + Index * ElemSize, ElemSize);
  cc.mov(Elem, ElemPtr);
}

template <typename T>
static inline void EmitStoreToFlexibleArray(jit::x86::Compiler &cc,
                                            jit::x86::Gp &ObjectAddr,
                                            size_t ArrayOff, size_t Index,
                                            const T &Elem, size_t ElemSize) {
  jit::x86::Mem ElemPtr =
      jit::x86::ptr(ObjectAddr, ArrayOff + Index * ElemSize, ElemSize);
  cc.mov(ElemPtr, Elem);
}

#endif
