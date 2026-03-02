/*
 * asmjit_x86_common.h
 *
 * x86/x86-64 specific emit helpers for pg_asmjit. Included by
 * asmjit_common.h when compiling on x86 targets.
 */
#ifndef ASMJIT_X86_COMMON_H
#define ASMJIT_X86_COMMON_H

#include <asmjit/x86.h>

namespace arch = asmjit::x86;

/*
 * Per-member load/store helpers for PostgreSQL structs.
 *
 * Generates:
 *   emit_load_<member>_from_<struct>(cc, base) -> Gp
 *   emit_store_<member>_to_<struct>(cc, base, val)
 */
#define TYPES_INFO(struct_type, member_type, member_name, reg_type)            \
  static inline asmjit::x86::Gp emit_load_##member_name##_from_##struct_type( \
      asmjit::x86::Compiler &cc, asmjit::x86::Gp &object_addr) {              \
    asmjit::x86::Mem member_ptr = asmjit::x86::ptr(                           \
        object_addr, offsetof(struct_type, member_name), sizeof(member_type)); \
    asmjit::x86::Gp member = cc.new##reg_type(#struct_type "_" #member_name); \
    cc.mov(member, member_ptr);                                                \
    return member;                                                             \
  }                                                                            \
  template <typename Op>                                                       \
  static inline void emit_store_##member_name##_to_##struct_type(              \
      asmjit::x86::Compiler &cc, asmjit::x86::Gp &object_addr, Op &&val) {    \
    asmjit::x86::Mem member_ptr = asmjit::x86::ptr(                           \
        object_addr, offsetof(struct_type, member_name), sizeof(member_type)); \
    cc.mov(member_ptr, val);                                                   \
  }
#include "jit_types_info.inc"
#undef TYPES_INFO

/*
 * Emit helpers to load/store constant-sized typed registers.
 */
#define LOAD_STORE_CONST(CType, JitType)                                       \
  static inline asmjit::x86::Gp EmitLoadConst##JitType(                       \
      asmjit::x86::Compiler &cc, const char *name, CType c) {                 \
    asmjit::x86::Gp reg = cc.new##JitType(name);                              \
    cc.mov(reg, asmjit::imm(c));                                               \
    return reg;                                                                \
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

static inline void EmitLoadFromArray(asmjit::x86::Compiler &cc,
                                     asmjit::x86::Gp &Array, size_t Index,
                                     asmjit::x86::Gp &Elem, size_t ElemSize) {
  asmjit::x86::Mem ElemPtr =
      asmjit::x86::ptr(Array, (int32_t)(Index * ElemSize), (uint32_t)ElemSize);
  cc.mov(Elem, ElemPtr);
}

template <typename T>
static inline void EmitStoreToArray(asmjit::x86::Compiler &cc,
                                    asmjit::x86::Gp &Array, size_t Index,
                                    const T &Elem, size_t ElemSize) {
  asmjit::x86::Mem ElemPtr =
      asmjit::x86::ptr(Array, (int32_t)(Index * ElemSize), (uint32_t)ElemSize);
  cc.mov(ElemPtr, Elem);
}

static inline void EmitLoadFromFlexibleArray(asmjit::x86::Compiler &cc,
                                             asmjit::x86::Gp &ObjectAddr,
                                             size_t ArrayOff, size_t Index,
                                             asmjit::x86::Gp &Elem,
                                             size_t ElemSize) {
  asmjit::x86::Mem ElemPtr = asmjit::x86::ptr(
      ObjectAddr, (int32_t)(ArrayOff + Index * ElemSize), (uint32_t)ElemSize);
  cc.mov(Elem, ElemPtr);
}

template <typename T>
static inline void EmitStoreToFlexibleArray(asmjit::x86::Compiler &cc,
                                            asmjit::x86::Gp &ObjectAddr,
                                            size_t ArrayOff, size_t Index,
                                            const T &Elem, size_t ElemSize) {
  asmjit::x86::Mem ElemPtr = asmjit::x86::ptr(
      ObjectAddr, (int32_t)(ArrayOff + Index * ElemSize), (uint32_t)ElemSize);
  cc.mov(ElemPtr, Elem);
}

static inline asmjit::x86::Gp LoadFuncArgNull(asmjit::x86::Compiler &cc,
                                              asmjit::x86::Gp &v_fcinfo,
                                              size_t argno) {
  asmjit::x86::Gp v_argnull = cc.newInt8("v_argnull.i8");
  EmitLoadFromFlexibleArray(cc, v_fcinfo,
                            offsetof(FunctionCallInfoBaseData, args) +
                                argno * sizeof(NullableDatum) +
                                offsetof(NullableDatum, isnull),
                            0, v_argnull, sizeof(bool));
  return v_argnull;
}

static inline asmjit::x86::Gp LoadFuncArgValue(asmjit::x86::Compiler &cc,
                                               asmjit::x86::Gp &v_fcinfo,
                                               size_t argno) {
  asmjit::x86::Gp v_argvalue = cc.newUIntPtr("v_argvalue.uintptr");
  EmitLoadFromFlexibleArray(cc, v_fcinfo,
                            offsetof(FunctionCallInfoBaseData, args) +
                                argno * sizeof(NullableDatum) +
                                offsetof(NullableDatum, value),
                            0, v_argvalue, sizeof(Datum));
  return v_argvalue;
}

template <typename T>
static inline void StoreFuncArgNull(asmjit::x86::Compiler &cc,
                                    asmjit::x86::Gp &v_fcinfo, size_t argno,
                                    const T &v_val) {
  EmitStoreToFlexibleArray(cc, v_fcinfo,
                           offsetof(FunctionCallInfoBaseData, args) +
                               argno * sizeof(NullableDatum) +
                               offsetof(NullableDatum, isnull),
                           0, v_val, sizeof(bool));
}

template <typename T>
static inline void StoreFuncArgValue(asmjit::x86::Compiler &cc,
                                     asmjit::x86::Gp &v_fcinfo, size_t argno,
                                     const T &v_val) {
  EmitStoreToFlexibleArray(cc, v_fcinfo,
                           offsetof(FunctionCallInfoBaseData, args) +
                               argno * sizeof(NullableDatum) +
                               offsetof(NullableDatum, value),
                           0, v_val, sizeof(Datum));
}

/*
 * Architecture-neutral wrappers for instruction patterns that differ
 * substantially between x86 and AArch64.
 */

/* Emit: if (reg == cmp_imm) goto label */
static inline void EmitCondJumpEQ(asmjit::x86::Compiler &cc,
                                  asmjit::x86::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  cc.cmp(reg, asmjit::imm(cmp_imm));
  cc.je(label);
}

/* Emit: if (reg != cmp_imm) goto label */
static inline void EmitCondJumpNE(asmjit::x86::Compiler &cc,
                                  asmjit::x86::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  cc.cmp(reg, asmjit::imm(cmp_imm));
  cc.jne(label);
}

/* Emit: if (reg >= cmp_imm) goto label (signed) */
static inline void EmitCondJumpGE(asmjit::x86::Compiler &cc,
                                  asmjit::x86::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  cc.cmp(reg, asmjit::imm(cmp_imm));
  cc.jge(label);
}

/* Emit: if (reg <= cmp_imm) goto label (signed) */
static inline void EmitCondJumpLE(asmjit::x86::Compiler &cc,
                                  asmjit::x86::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  cc.cmp(reg, asmjit::imm(cmp_imm));
  cc.jle(label);
}

/* Emit: if (reg > cmp_imm) goto label (signed) */
static inline void EmitCondJumpGT(asmjit::x86::Compiler &cc,
                                  asmjit::x86::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  cc.cmp(reg, asmjit::imm(cmp_imm));
  cc.jg(label);
}

/* Emit: if (reg != rhs_reg) goto label */
static inline void EmitCondJumpRegNE(asmjit::x86::Compiler &cc,
                                     asmjit::x86::Gp &lhs,
                                     asmjit::x86::Gp &rhs,
                                     asmjit::Label &label) {
  cc.cmp(lhs, rhs);
  cc.jne(label);
}

/* Emit: if (reg == rhs_reg) goto label */
static inline void EmitCondJumpRegEQ(asmjit::x86::Compiler &cc,
                                     asmjit::x86::Gp &lhs,
                                     asmjit::x86::Gp &rhs,
                                     asmjit::Label &label) {
  cc.cmp(lhs, rhs);
  cc.je(label);
}

/* Emit unconditional jump */
static inline void EmitJump(asmjit::x86::Compiler &cc,
                            asmjit::Label &label) {
  cc.jmp(label);
}

/* dst = (reg == imm) ? 1 : 0 */
static inline void EmitSetEQ(asmjit::x86::Compiler &cc,
                             asmjit::x86::Gp &dst, asmjit::x86::Gp &reg,
                             int64_t cmp_imm) {
  cc.xor_(dst, dst);
  cc.cmp(reg, asmjit::imm(cmp_imm));
  cc.sete(dst);
}

/* dst = (reg < 0) ? 1 : 0  (signed, after cmp with 0) */
static inline void EmitSetLT(asmjit::x86::Compiler &cc,
                             asmjit::x86::Gp &dst, asmjit::x86::Gp &reg) {
  cc.xor_(dst, dst);
  cc.cmp(reg, asmjit::imm(0));
  cc.setl(dst);
}

static inline void EmitSetLE(asmjit::x86::Compiler &cc,
                             asmjit::x86::Gp &dst, asmjit::x86::Gp &reg) {
  cc.xor_(dst, dst);
  cc.cmp(reg, asmjit::imm(0));
  cc.setle(dst);
}

static inline void EmitSetGT(asmjit::x86::Compiler &cc,
                             asmjit::x86::Gp &dst, asmjit::x86::Gp &reg) {
  cc.xor_(dst, dst);
  cc.cmp(reg, asmjit::imm(0));
  cc.setg(dst);
}

static inline void EmitSetGE(asmjit::x86::Compiler &cc,
                             asmjit::x86::Gp &dst, asmjit::x86::Gp &reg) {
  cc.xor_(dst, dst);
  cc.cmp(reg, asmjit::imm(0));
  cc.setge(dst);
}

/* Zero a register */
static inline void EmitZero(asmjit::x86::Compiler &cc,
                            asmjit::x86::Gp &reg) {
  cc.xor_(reg, reg);
}

/* dst |= src (2-operand x86 style) */
static inline void EmitBitwiseOr(asmjit::x86::Compiler &cc,
                                 asmjit::x86::Gp &dst,
                                 asmjit::x86::Gp &src) {
  cc.or_(dst, src);
}

/* dst &= src (2-operand x86 style) */
static inline void EmitBitwiseAnd(asmjit::x86::Compiler &cc,
                                  asmjit::x86::Gp &dst,
                                  asmjit::x86::Gp &src) {
  cc.and_(dst, src);
}

/* dst &= imm */
static inline void EmitBitwiseAndImm(asmjit::x86::Compiler &cc,
                                     asmjit::x86::Gp &dst, int64_t imm_val) {
  cc.and_(dst, asmjit::imm(imm_val));
}

/* dst |= imm */
static inline void EmitBitwiseOrImm(asmjit::x86::Compiler &cc,
                                    asmjit::x86::Gp &dst, int64_t imm_val) {
  cc.or_(dst, asmjit::imm(imm_val));
}

/* dst ^= src */
static inline void EmitBitwiseXor(asmjit::x86::Compiler &cc,
                                  asmjit::x86::Gp &dst,
                                  asmjit::x86::Gp &src) {
  cc.xor_(dst, src);
}

/* dst += imm */
static inline void EmitAddImm(asmjit::x86::Compiler &cc,
                              asmjit::x86::Gp &dst, int64_t imm_val) {
  cc.add(dst, asmjit::imm(imm_val));
}

/* dst += src */
static inline void EmitAddReg(asmjit::x86::Compiler &cc,
                              asmjit::x86::Gp &dst,
                              asmjit::x86::Gp &src) {
  cc.add(dst, src);
}

/* dst++ */
static inline void EmitInc(asmjit::x86::Compiler &cc,
                           asmjit::x86::Gp &dst) {
  cc.inc(dst);
}

/* Shift left by immediate */
static inline void EmitShlImm(asmjit::x86::Compiler &cc,
                              asmjit::x86::Gp &dst, uint32_t shift) {
  cc.shl(dst, asmjit::imm(shift));
}

/* Logical shift right by immediate */
static inline void EmitShrImm(asmjit::x86::Compiler &cc,
                              asmjit::x86::Gp &dst, uint32_t shift) {
  cc.shr(dst, asmjit::imm(shift));
}

/* Zero-extend 8-bit -> native width */
static inline void EmitZeroExtend8(asmjit::x86::Compiler &cc,
                                   asmjit::x86::Gp &dst,
                                   asmjit::x86::Gp &src) {
  cc.movzx(dst, src);
}

/* Zero-extend 16-bit -> 32-bit */
static inline void EmitZeroExtend16(asmjit::x86::Compiler &cc,
                                    asmjit::x86::Gp &dst,
                                    asmjit::x86::Gp &src) {
  cc.movzx(dst, src);
}

/* Sign-extend 8-bit -> 64-bit */
static inline void EmitSignExtend8to64(asmjit::x86::Compiler &cc,
                                       asmjit::x86::Gp &dst,
                                       asmjit::x86::Gp &src) {
  cc.movsx(dst, src);
}

/* Sign-extend 16-bit -> 32-bit (for movsxd-like sign extension) */
static inline void EmitSignExtend16to32(asmjit::x86::Compiler &cc,
                                        asmjit::x86::Gp &dst,
                                        asmjit::x86::Gp &src) {
  cc.movsxd(dst, src);
}

/* Sign-extend 32-bit -> 64-bit */
static inline void EmitSignExtend32to64(asmjit::x86::Compiler &cc,
                                        asmjit::x86::Gp &dst,
                                        asmjit::x86::Gp &src) {
  cc.movsxd(dst, src);
}

/* Zero-extend 32-bit -> 64-bit (nop on x86-64, upper bits are zeroed by mov) */
static inline void EmitZeroExtend32to64(asmjit::x86::Compiler &cc,
                                        asmjit::x86::Gp &dst,
                                        asmjit::x86::Gp &src) {
  cc.movzx(dst, src);
}

#endif /* ASMJIT_X86_COMMON_H */
