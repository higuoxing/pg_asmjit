/*
 * asmjit_a64_common.h
 *
 * AArch64 specific emit helpers for pg_asmjit. Included by asmjit_common.h
 * when compiling on ARM64 targets.
 */
#ifndef ASMJIT_A64_COMMON_H
#define ASMJIT_A64_COMMON_H

#include <asmjit/a64.h>

namespace arch = asmjit::a64;

/*
 * Per-member load/store helpers for PostgreSQL structs.
 *
 * On AArch64 we use ldr/str with offset addressing. The register width is
 * determined by the register type passed (Gp32 vs Gp64), so we use
 * new_gp_ptr() for pointer-sized values and new_gp32() for narrower ones.
 * We always load into 64-bit registers (sign/zero extension is handled by
 * the appropriate ldr variant) to avoid aliasing issues in the compiler.
 *
 * The "reg_type" column in jit_types_info.inc is the old x86 camelCase name
 * (UIntPtr, Int8, UInt16, etc.).  We map those to a64 new_gp_ptr or new_gp32
 * as appropriate.  For simplicity, all sub-64-bit integral values use a 32-bit
 * virtual register; the compiler handles spilling/allocation.
 */
#define JIT_A64_NEW_REG_UIntPtr(cc, name)  (cc).new_gp_ptr(name)
#define JIT_A64_NEW_REG_IntPtr(cc, name)   (cc).new_gp_ptr(name)
#define JIT_A64_NEW_REG_UInt64(cc, name)   (cc).new_gp64(name)
#define JIT_A64_NEW_REG_Int64(cc, name)    (cc).new_gp64(name)
#define JIT_A64_NEW_REG_UInt32(cc, name)   (cc).new_gp32(name)
#define JIT_A64_NEW_REG_Int32(cc, name)    (cc).new_gp32(name)
#define JIT_A64_NEW_REG_UInt16(cc, name)   (cc).new_gp32(name)
#define JIT_A64_NEW_REG_Int16(cc, name)    (cc).new_gp32(name)
#define JIT_A64_NEW_REG_UInt8(cc, name)    (cc).new_gp32(name)
#define JIT_A64_NEW_REG_Int8(cc, name)     (cc).new_gp32(name)

/*
 * AArch64 ldr/str variants by size of the member.
 */
static inline void a64_load_mem(asmjit::a64::Compiler &cc,
                                asmjit::a64::Gp &dst,
                                asmjit::a64::Mem &mem,
                                size_t sz) {
  switch (sz) {
  case 1: cc.ldrb(dst.w(), mem); break;
  case 2: cc.ldrh(dst.w(), mem); break;
  case 4: cc.ldr(dst.w(), mem); break;
  default: cc.ldr(dst.x(), mem); break;
  }
}

static inline void a64_store_gp(asmjit::a64::Compiler &cc,
                                asmjit::a64::Mem &mem,
                                asmjit::a64::Gp &src,
                                size_t sz) {
  switch (sz) {
  case 1: cc.strb(src.w(), mem); break;
  case 2: cc.strh(src.w(), mem); break;
  case 4: cc.str(src.w(), mem); break;
  default: cc.str(src.x(), mem); break;
  }
}

static inline void a64_store_imm(asmjit::a64::Compiler &cc,
                                 asmjit::a64::Mem &mem,
                                 int64_t val,
                                 size_t sz) {
  asmjit::a64::Gp tmp = cc.new_gp64("tmp.imm");
  cc.mov(tmp, asmjit::imm(val));
  a64_store_gp(cc, mem, tmp, sz);
}

#define TYPES_INFO(struct_type, member_type, member_name, reg_type)               \
  static inline asmjit::a64::Gp                                                   \
  emit_load_##member_name##_from_##struct_type(asmjit::a64::Compiler &cc,         \
                                               asmjit::a64::Gp &object_addr) {    \
    asmjit::a64::Mem member_ptr =                                                  \
        asmjit::a64::ptr(object_addr.x(),                                         \
                         (int32_t)offsetof(struct_type, member_name));             \
    asmjit::a64::Gp member =                                                      \
        JIT_A64_NEW_REG_##reg_type(cc, #struct_type "_" #member_name);            \
    a64_load_mem(cc, member, member_ptr, sizeof(member_type));                    \
    return member;                                                                 \
  }                                                                                \
  static inline void emit_store_##member_name##_to_##struct_type(                 \
      asmjit::a64::Compiler &cc, asmjit::a64::Gp &object_addr,                   \
      asmjit::a64::Gp val) {                                                      \
    asmjit::a64::Mem member_ptr =                                                  \
        asmjit::a64::ptr(object_addr.x(),                                         \
                         (int32_t)offsetof(struct_type, member_name));             \
    a64_store_gp(cc, member_ptr, val, sizeof(member_type));                       \
  }                                                                                \
  static inline void emit_store_##member_name##_to_##struct_type(                 \
      asmjit::a64::Compiler &cc, asmjit::a64::Gp &object_addr, asmjit::Imm val) {\
    asmjit::a64::Mem member_ptr =                                                  \
        asmjit::a64::ptr(object_addr.x(),                                         \
                         (int32_t)offsetof(struct_type, member_name));             \
    a64_store_imm(cc, member_ptr, val.value(), sizeof(member_type));              \
  }
#include "jit_types_info.inc"
#undef TYPES_INFO

/*
 * Emit constant loads.
 *
 * All "EmitLoadConst*" functions return a 64-bit (pointer-sized) register
 * holding the immediate value. On AArch64 the compiler handles splitting
 * large immediates across multiple movz/movk instructions.
 */
static inline asmjit::a64::Gp EmitLoadConstUIntPtr(asmjit::a64::Compiler &cc,
                                                    const char *name,
                                                    void *c) {
  asmjit::a64::Gp reg = cc.new_gp_ptr(name);
  cc.mov(reg, asmjit::imm((uintptr_t)c));
  return reg;
}

static inline asmjit::a64::Gp EmitLoadConstIntPtr(asmjit::a64::Compiler &cc,
                                                   const char *name,
                                                   intptr_t c) {
  asmjit::a64::Gp reg = cc.new_gp_ptr(name);
  cc.mov(reg, asmjit::imm(c));
  return reg;
}

static inline asmjit::a64::Gp EmitLoadConstUInt8(asmjit::a64::Compiler &cc,
                                                  const char *name, uint8_t c) {
  asmjit::a64::Gp reg = cc.new_gp32(name);
  cc.mov(reg, asmjit::imm(c));
  return reg;
}

static inline asmjit::a64::Gp EmitLoadConstInt8(asmjit::a64::Compiler &cc,
                                                 const char *name, int8_t c) {
  asmjit::a64::Gp reg = cc.new_gp32(name);
  cc.mov(reg, asmjit::imm(c));
  return reg;
}

static inline asmjit::a64::Gp EmitLoadConstUInt32(asmjit::a64::Compiler &cc,
                                                   const char *name,
                                                   uint32_t c) {
  asmjit::a64::Gp reg = cc.new_gp32(name);
  cc.mov(reg, asmjit::imm(c));
  return reg;
}

static inline asmjit::a64::Gp EmitLoadConstInt32(asmjit::a64::Compiler &cc,
                                                  const char *name, int32_t c) {
  asmjit::a64::Gp reg = cc.new_gp32(name);
  cc.mov(reg, asmjit::imm(c));
  return reg;
}

static inline asmjit::a64::Gp EmitLoadConstUInt64(asmjit::a64::Compiler &cc,
                                                   const char *name,
                                                   uint64_t c) {
  asmjit::a64::Gp reg = cc.new_gp64(name);
  cc.mov(reg, asmjit::imm(c));
  return reg;
}

static inline asmjit::a64::Gp EmitLoadConstInt64(asmjit::a64::Compiler &cc,
                                                  const char *name, int64_t c) {
  asmjit::a64::Gp reg = cc.new_gp64(name);
  cc.mov(reg, asmjit::imm(c));
  return reg;
}

/* Array load/store helpers */
static inline void EmitLoadFromArray(asmjit::a64::Compiler &cc,
                                     asmjit::a64::Gp &Array, size_t Index,
                                     asmjit::a64::Gp &Elem, size_t ElemSize) {
  asmjit::a64::Mem ElemPtr =
      asmjit::a64::ptr(Array.x(), (int32_t)(Index * ElemSize));
  a64_load_mem(cc, Elem, ElemPtr, ElemSize);
}

static inline void EmitStoreToArray(asmjit::a64::Compiler &cc,
                                    asmjit::a64::Gp &Array, size_t Index,
                                    asmjit::a64::Gp &Elem, size_t ElemSize) {
  asmjit::a64::Mem ElemPtr =
      asmjit::a64::ptr(Array.x(), (int32_t)(Index * ElemSize));
  a64_store_gp(cc, ElemPtr, Elem, ElemSize);
}

static inline void EmitStoreToArray(asmjit::a64::Compiler &cc,
                                    asmjit::a64::Gp &Array, size_t Index,
                                    asmjit::Imm Elem, size_t ElemSize) {
  asmjit::a64::Mem ElemPtr =
      asmjit::a64::ptr(Array.x(), (int32_t)(Index * ElemSize));
  a64_store_imm(cc, ElemPtr, Elem.value(), ElemSize);
}

static inline void EmitLoadFromFlexibleArray(asmjit::a64::Compiler &cc,
                                             asmjit::a64::Gp &ObjectAddr,
                                             size_t ArrayOff, size_t Index,
                                             asmjit::a64::Gp &Elem,
                                             size_t ElemSize) {
  asmjit::a64::Mem ElemPtr = asmjit::a64::ptr(
      ObjectAddr.x(), (int32_t)(ArrayOff + Index * ElemSize));
  a64_load_mem(cc, Elem, ElemPtr, ElemSize);
}

static inline void EmitStoreToFlexibleArray(asmjit::a64::Compiler &cc,
                                            asmjit::a64::Gp &ObjectAddr,
                                            size_t ArrayOff, size_t Index,
                                            asmjit::a64::Gp &Elem,
                                            size_t ElemSize) {
  asmjit::a64::Mem ElemPtr = asmjit::a64::ptr(
      ObjectAddr.x(), (int32_t)(ArrayOff + Index * ElemSize));
  a64_store_gp(cc, ElemPtr, Elem, ElemSize);
}

static inline void EmitStoreToFlexibleArray(asmjit::a64::Compiler &cc,
                                            asmjit::a64::Gp &ObjectAddr,
                                            size_t ArrayOff, size_t Index,
                                            asmjit::Imm Elem,
                                            size_t ElemSize) {
  asmjit::a64::Mem ElemPtr = asmjit::a64::ptr(
      ObjectAddr.x(), (int32_t)(ArrayOff + Index * ElemSize));
  a64_store_imm(cc, ElemPtr, Elem.value(), ElemSize);
}

static inline asmjit::a64::Gp LoadFuncArgNull(asmjit::a64::Compiler &cc,
                                              asmjit::a64::Gp &v_fcinfo,
                                              size_t argno) {
  asmjit::a64::Gp v_argnull = cc.new_gp32("v_argnull");
  EmitLoadFromFlexibleArray(cc, v_fcinfo,
                            offsetof(FunctionCallInfoBaseData, args) +
                                argno * sizeof(NullableDatum) +
                                offsetof(NullableDatum, isnull),
                            0, v_argnull, sizeof(bool));
  return v_argnull;
}

static inline asmjit::a64::Gp LoadFuncArgValue(asmjit::a64::Compiler &cc,
                                               asmjit::a64::Gp &v_fcinfo,
                                               size_t argno) {
  asmjit::a64::Gp v_argvalue = cc.new_gp_ptr("v_argvalue");
  EmitLoadFromFlexibleArray(cc, v_fcinfo,
                            offsetof(FunctionCallInfoBaseData, args) +
                                argno * sizeof(NullableDatum) +
                                offsetof(NullableDatum, value),
                            0, v_argvalue, sizeof(Datum));
  return v_argvalue;
}

static inline void StoreFuncArgNull(asmjit::a64::Compiler &cc,
                                    asmjit::a64::Gp &v_fcinfo, size_t argno,
                                    asmjit::a64::Gp v_val) {
  EmitStoreToFlexibleArray(cc, v_fcinfo,
                           offsetof(FunctionCallInfoBaseData, args) +
                               argno * sizeof(NullableDatum) +
                               offsetof(NullableDatum, isnull),
                           0, v_val, sizeof(bool));
}

static inline void StoreFuncArgNull(asmjit::a64::Compiler &cc,
                                    asmjit::a64::Gp &v_fcinfo, size_t argno,
                                    asmjit::Imm v_val) {
  EmitStoreToFlexibleArray(cc, v_fcinfo,
                           offsetof(FunctionCallInfoBaseData, args) +
                               argno * sizeof(NullableDatum) +
                               offsetof(NullableDatum, isnull),
                           0, v_val, sizeof(bool));
}

static inline void StoreFuncArgValue(asmjit::a64::Compiler &cc,
                                     asmjit::a64::Gp &v_fcinfo, size_t argno,
                                     asmjit::a64::Gp v_val) {
  EmitStoreToFlexibleArray(cc, v_fcinfo,
                           offsetof(FunctionCallInfoBaseData, args) +
                               argno * sizeof(NullableDatum) +
                               offsetof(NullableDatum, value),
                           0, v_val, sizeof(Datum));
}

static inline void StoreFuncArgValue(asmjit::a64::Compiler &cc,
                                     asmjit::a64::Gp &v_fcinfo, size_t argno,
                                     asmjit::Imm v_val) {
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

/*
 * Helper: load a possibly-large immediate into a temp register and compare.
 * AArch64 CMP with immediate is limited to 12-bit values (optionally shifted
 * by 12).  For general use, we always load into a temp register.
 */
static inline asmjit::a64::Gp a64_load_cmp_imm(asmjit::a64::Compiler &cc,
                                                 asmjit::a64::Gp &reg,
                                                 int64_t cmp_imm) {
  /* Use matching width (W for 32-bit regs, X for 64-bit) */
  asmjit::a64::Gp tmp = reg.is_gp32() ? cc.new_gp32("tmp.cmp") : cc.new_gp64("tmp.cmp");
  cc.mov(tmp, asmjit::imm(cmp_imm));
  cc.cmp(reg, tmp);
  return tmp;
}

static inline void EmitCondJumpEQ(asmjit::a64::Compiler &cc,
                                  asmjit::a64::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  a64_load_cmp_imm(cc, reg, cmp_imm);
  cc.b_eq(label);
}

static inline void EmitCondJumpNE(asmjit::a64::Compiler &cc,
                                  asmjit::a64::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  a64_load_cmp_imm(cc, reg, cmp_imm);
  cc.b_ne(label);
}

static inline void EmitCondJumpGE(asmjit::a64::Compiler &cc,
                                  asmjit::a64::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  a64_load_cmp_imm(cc, reg, cmp_imm);
  cc.b_ge(label);
}

static inline void EmitCondJumpLE(asmjit::a64::Compiler &cc,
                                  asmjit::a64::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  a64_load_cmp_imm(cc, reg, cmp_imm);
  cc.b_le(label);
}

static inline void EmitCondJumpGT(asmjit::a64::Compiler &cc,
                                  asmjit::a64::Gp &reg, int64_t cmp_imm,
                                  asmjit::Label &label) {
  a64_load_cmp_imm(cc, reg, cmp_imm);
  cc.b_gt(label);
}

static inline void EmitCondJumpRegNE(asmjit::a64::Compiler &cc,
                                     asmjit::a64::Gp &lhs,
                                     asmjit::a64::Gp &rhs,
                                     asmjit::Label &label) {
  cc.cmp(lhs, rhs);
  cc.b_ne(label);
}

static inline void EmitCondJumpRegEQ(asmjit::a64::Compiler &cc,
                                     asmjit::a64::Gp &lhs,
                                     asmjit::a64::Gp &rhs,
                                     asmjit::Label &label) {
  cc.cmp(lhs, rhs);
  cc.b_eq(label);
}

static inline void EmitJump(asmjit::a64::Compiler &cc,
                            asmjit::Label &label) {
  cc.b(label);
}

/* dst = (reg == imm) ? 1 : 0 */
static inline void EmitSetEQ(asmjit::a64::Compiler &cc,
                             asmjit::a64::Gp &dst, asmjit::a64::Gp &reg,
                             int64_t cmp_imm) {
  a64_load_cmp_imm(cc, reg, cmp_imm);
  cc.cset(dst, asmjit::arm::CondCode::kEQ);
}

static inline void EmitSetLT(asmjit::a64::Compiler &cc,
                             asmjit::a64::Gp &dst, asmjit::a64::Gp &reg) {
  a64_load_cmp_imm(cc, reg, 0);
  cc.cset(dst, asmjit::arm::CondCode::kLT);
}

static inline void EmitSetLE(asmjit::a64::Compiler &cc,
                             asmjit::a64::Gp &dst, asmjit::a64::Gp &reg) {
  a64_load_cmp_imm(cc, reg, 0);
  cc.cset(dst, asmjit::arm::CondCode::kLE);
}

static inline void EmitSetGT(asmjit::a64::Compiler &cc,
                             asmjit::a64::Gp &dst, asmjit::a64::Gp &reg) {
  a64_load_cmp_imm(cc, reg, 0);
  cc.cset(dst, asmjit::arm::CondCode::kGT);
}

static inline void EmitSetGE(asmjit::a64::Compiler &cc,
                             asmjit::a64::Gp &dst, asmjit::a64::Gp &reg) {
  a64_load_cmp_imm(cc, reg, 0);
  cc.cset(dst, asmjit::arm::CondCode::kGE);
}

/* Zero a register */
static inline void EmitZero(asmjit::a64::Compiler &cc,
                            asmjit::a64::Gp &reg) {
  cc.mov(reg, asmjit::imm(0));
}

/* dst = dst | src  (3-operand on AArch64) */
static inline void EmitBitwiseOr(asmjit::a64::Compiler &cc,
                                 asmjit::a64::Gp &dst,
                                 asmjit::a64::Gp &src) {
  cc.orr(dst, dst, src);
}

/* dst = dst & src  (3-operand on AArch64) */
static inline void EmitBitwiseAnd(asmjit::a64::Compiler &cc,
                                  asmjit::a64::Gp &dst,
                                  asmjit::a64::Gp &src) {
  cc.and_(dst, dst, src);
}

/* dst = dst & imm
 * AArch64 AND only accepts "logical immediates" (a subset of bit patterns).
 * For general-purpose use, load the mask into a temp register to avoid
 * encoding errors with arbitrary bit patterns.
 * The temp register must have the same width as dst (W for gp32, X for gp64). */
static inline void EmitBitwiseAndImm(asmjit::a64::Compiler &cc,
                                     asmjit::a64::Gp &dst, int64_t imm_val) {
  asmjit::a64::Gp tmp = dst.is_gp32() ? cc.new_gp32("tmp.and") : cc.new_gp64("tmp.and");
  cc.mov(tmp, asmjit::imm(imm_val));
  cc.and_(dst, dst, tmp);
}

/* dst = dst | imm (same concern as AND) */
static inline void EmitBitwiseOrImm(asmjit::a64::Compiler &cc,
                                    asmjit::a64::Gp &dst, int64_t imm_val) {
  asmjit::a64::Gp tmp = dst.is_gp32() ? cc.new_gp32("tmp.or") : cc.new_gp64("tmp.or");
  cc.mov(tmp, asmjit::imm(imm_val));
  cc.orr(dst, dst, tmp);
}

/* dst = dst ^ src */
static inline void EmitBitwiseXor(asmjit::a64::Compiler &cc,
                                  asmjit::a64::Gp &dst,
                                  asmjit::a64::Gp &src) {
  cc.eor(dst, dst, src);
}

/* dst = dst + imm (3-operand on AArch64) */
static inline void EmitAddImm(asmjit::a64::Compiler &cc,
                              asmjit::a64::Gp &dst, int64_t imm_val) {
  cc.add(dst, dst, asmjit::imm(imm_val));
}

/* dst = dst + src */
static inline void EmitAddReg(asmjit::a64::Compiler &cc,
                              asmjit::a64::Gp &dst,
                              asmjit::a64::Gp &src) {
  cc.add(dst, dst, src);
}

/* dst++ → add dst, dst, 1 */
static inline void EmitInc(asmjit::a64::Compiler &cc,
                           asmjit::a64::Gp &dst) {
  cc.add(dst, dst, asmjit::imm(1));
}

/* Logical shift left */
static inline void EmitShlImm(asmjit::a64::Compiler &cc,
                              asmjit::a64::Gp &dst, uint32_t shift) {
  cc.lsl(dst, dst, asmjit::imm(shift));
}

/* Logical shift right */
static inline void EmitShrImm(asmjit::a64::Compiler &cc,
                              asmjit::a64::Gp &dst, uint32_t shift) {
  cc.lsr(dst, dst, asmjit::imm(shift));
}

/* Zero-extend 8-bit (byte) to native width */
static inline void EmitZeroExtend8(asmjit::a64::Compiler &cc,
                                   asmjit::a64::Gp &dst,
                                   asmjit::a64::Gp &src) {
  cc.uxtb(dst, src);
}

/* Zero-extend 16-bit to 32-bit */
static inline void EmitZeroExtend16(asmjit::a64::Compiler &cc,
                                    asmjit::a64::Gp &dst,
                                    asmjit::a64::Gp &src) {
  cc.uxth(dst, src);
}

/* Sign-extend 8-bit to 64-bit */
static inline void EmitSignExtend8to64(asmjit::a64::Compiler &cc,
                                       asmjit::a64::Gp &dst,
                                       asmjit::a64::Gp &src) {
  cc.sxtb(dst, src);
}

/* Sign-extend 16-bit to 32-bit */
static inline void EmitSignExtend16to32(asmjit::a64::Compiler &cc,
                                        asmjit::a64::Gp &dst,
                                        asmjit::a64::Gp &src) {
  cc.sxth(dst, src);
}

/* Sign-extend 32-bit to 64-bit */
static inline void EmitSignExtend32to64(asmjit::a64::Compiler &cc,
                                        asmjit::a64::Gp &dst,
                                        asmjit::a64::Gp &src) {
  cc.sxtw(dst, src);
}

/* Zero-extend 32-bit to 64-bit (writing a 32-bit reg zeros upper 32 bits) */
static inline void EmitZeroExtend32to64(asmjit::a64::Compiler &cc,
                                        asmjit::a64::Gp &dst,
                                        asmjit::a64::Gp &src) {
  /* On AArch64, moving into a 32-bit register automatically zero-extends */
  cc.mov(dst.w(), src.w());
}

#endif /* ASMJIT_A64_COMMON_H */
