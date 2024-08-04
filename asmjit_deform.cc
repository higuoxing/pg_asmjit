#include "asmjit/core/compiler.h"
#include "asmjit/x86/x86operand.h"
#include "asmjit_common.h"

namespace x86 = asmjit::x86;
namespace jit = asmjit;

extern "C" {

TupleDeformingFunc CompileTupleDeformingFunc(AsmJitContext *Context,
                                             jit::JitRuntime &Runtime,
                                             TupleDesc Desc,
                                             const TupleTableSlotOps *TtsOps,
                                             int NAtts) {
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
   * Check which columns have to exist, so we don't have to check the row's
   * natts unnecessarily.
   */
  int GuaranteedColumnNumber = -1;
  for (int attnum = 0; attnum < Desc->natts; ++attnum) {
    Form_pg_attribute att = TupleDescAttr(Desc, attnum);

    /*
     * If the column is declared NOT NULL then it must be present in every
     * tuple, unless there's a "missing" entry that could provide a
     * non-NULL value for it. That in turn guarantees that the NULL bitmap
     * - if there are any NULLable columns - is at least long enough to
     * cover columns up to attnum.
     *
     * Be paranoid and also check !attisdropped, even though the
     * combination of attisdropped && attnotnull combination shouldn't
     * exist.
     */
    if (att->attnotnull && !att->atthasmissing && !att->attisdropped)
      GuaranteedColumnNumber = attnum;
  }

  /*
   * void (*TupleDeformingFunc) (TupleTableSlot *);
   */
  jit::FuncNode *JittedDeformingFunc =
      Jitcc.addFunc(jit::FuncSignature::build<void, TupleTableSlot *>());
  x86::Gp Slot = Jitcc.newUIntPtr();
  JittedDeformingFunc->setArg(0, Slot);

  jit::Label L_Entry = Jitcc.newLabel(), L_AdjustUnavailCols = Jitcc.newLabel(),
             L_FindStart = Jitcc.newLabel(), L_Out = Jitcc.newLabel(),
             L_Dead = Jitcc.newLabel();

  jit::Label *L_AttCheckAttnoBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts),
             *L_AttStartBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts),
             *L_AttIsNullBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts),
             *L_AttCheckAlignBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts),
             *L_AttAlignBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts),
             *L_AttStoreBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts);
  for (int Attnum = 0; Attnum < NAtts; ++Attnum) {
    L_AttCheckAttnoBlocks[Attnum] = Jitcc.newLabel();
    L_AttStartBlocks[Attnum] = Jitcc.newLabel();
    L_AttIsNullBlocks[Attnum] = Jitcc.newLabel();
    L_AttCheckAlignBlocks[Attnum] = Jitcc.newLabel();
    L_AttAlignBlocks[Attnum] = Jitcc.newLabel();
    L_AttStoreBlocks[Attnum] = Jitcc.newLabel();
  }

  x86::Gp TtsValues = emit_load_tts_values_from_TupleTableSlot(Jitcc, Slot),
          TtsNulls = emit_load_tts_isnull_from_TupleTableSlot(Jitcc, Slot),
          TtsFlags = emit_load_tts_flags_from_TupleTableSlot(Jitcc, Slot),
          TtsNValid = emit_load_tts_nvalid_from_TupleTableSlot(Jitcc, Slot);

  x86::Gp SlotOffset, HeapTuple;

  if (TtsOps == &TTSOpsHeapTuple || TtsOps == &TTSOpsBufferHeapTuple) {
    SlotOffset = emit_load_off_from_HeapTupleTableSlot(Jitcc, Slot);
    HeapTuple = emit_load_tuple_from_HeapTupleTableSlot(Jitcc, Slot);
  } else if (TtsOps == &TTSOpsMinimalTuple) {
    SlotOffset = emit_load_off_from_MinimalTupleTableSlot(Jitcc, Slot);
    HeapTuple = emit_load_tuple_from_MinimalTupleTableSlot(Jitcc, Slot);
  } else {
    /* Should've returned at the start of the function. */
    pg_unreachable();
  }

  x86::Gp TupleHeader = emit_load_t_data_from_HeapTupleData(Jitcc, HeapTuple),
          Bits = emit_load_t_bits_from_HeapTupleHeaderData(Jitcc, TupleHeader),
          InfoMask1 =
              emit_load_t_infomask_from_HeapTupleHeaderData(Jitcc, TupleHeader),
          InfoMask2 = emit_load_t_infomask2_from_HeapTupleHeaderData(
              Jitcc, TupleHeader),
          Hoff = emit_load_t_hoff_from_HeapTupleHeaderData(Jitcc, TupleHeader);

  /* t_infomask & HEAP_HASNULL */
  x86::Gp HasNulls = Jitcc.newUInt16();
  Jitcc.mov(HasNulls, InfoMask1);
  Jitcc.and_(HasNulls, jit::imm(HEAP_HASNULL));

  /* t_infomask2 & HEAP_NATTS_MASK */
  x86::Gp MaxAtt = Jitcc.newUInt16();
  Jitcc.mov(MaxAtt, InfoMask2);
  Jitcc.and_(MaxAtt, jit::imm(HEAP_NATTS_MASK));

  x86::Mem TupleDataBase = x86::ptr(TupleHeader, Hoff);

  /*
   * Load tuple start offset from slot. Will be reset below in case there's
   * no existing deformed columns in slot.
   */
  x86::Gp Offset = Jitcc.newUIntPtr();
  Jitcc.movzx(Offset, SlotOffset);

  /*
   * Check if it is guaranteed that all the desired attributes are available
   * in the tuple (but still possibly NULL), by dint of either the last
   * to-be-deformed column being NOT NULL, or subsequent ones not accessed
   * here being NOT NULL.  If that's not guaranteed the tuple headers natt's
   * has to be checked, and missing attributes potentially have to be
   * fetched (using slot_getmissingattrs().
   */
  if (NAtts - 1 <= GuaranteedColumnNumber) {
    /*
     * Currently, it seems that we don't need emit codes for this branch.
     * Jitcc.jmp(AdjustUnavailCols);
     * Jitcc.bind(AdjustUnavailCols);
     * Jitcc.jmp(FindStart);
     */
  } else {
    Jitcc.cmp(MaxAtt, jit::imm(NAtts));
    Jitcc.jge(L_FindStart);
    /* Jitcc.bind(AdjustUnavailCols); */
    x86::Gp MaxAttAsI32 = Jitcc.newInt32();
    Jitcc.movzx(MaxAttAsI32, MaxAtt);
    jit::InvokeNode *SlotGetMissingAttrs;
    Jitcc.invoke(&SlotGetMissingAttrs, jit::imm(slot_getmissingattrs),
                 jit::FuncSignature::build<void, TupleTableSlot *, int, int>());
    SlotGetMissingAttrs->setArg(0, Slot);
    SlotGetMissingAttrs->setArg(1, MaxAtt);
    SlotGetMissingAttrs->setArg(2, jit::imm(NAtts));
    /* Jitcc.jmp(FindStart); */
  }

  Jitcc.bind(L_FindStart);

  /*
   * Build switch to go from nvalid to the right startblock.  Callers
   * currently don't have the knowledge, but it'd be good for performance to
   * avoid this check when it's known that the slot is empty (e.g. in scan
   * nodes).
   */

#if 0
  x86::Gp TtsOpsAddr = load_tts_ops_from_TupleTableSlot(Jitcc, Slot);
  x86::Gp GetSomeAttrs =
      load_getsomeattrs_from_TupleTableSlotOps(Jitcc, TtsOpsAddr);

  jit::InvokeNode *GetSomeAttrsInvoke;
  Jitcc.invoke(&GetSomeAttrsInvoke, GetSomeAttrs,
               jit::FuncSignature::build<void, TupleTableSlot *, int>());
  GetSomeAttrsInvoke->setArg(0, Slot);
  GetSomeAttrsInvoke->setArg(1, jit::imm(NAttrs));

  jit::Label Out = Jitcc.newLabel();

  x86::Mem v_TtsNValid =
      x86::ptr(Slot, offsetof(TupleTableSlot, tts_nvalid), sizeof(AttrNumber));
  x86::Gp TtsNValid = Jitcc.newInt32();
  Jitcc.movzx(TtsNValid, v_TtsNValid);

  Jitcc.cmp(TtsNValid, jit::imm(NAttrs));
  Jitcc.jge(Out);

  jit::InvokeNode *GetMissingAttrsInvoke;
  Jitcc.invoke(&GetMissingAttrsInvoke, jit::imm(slot_getmissingattrs),
               jit::FuncSignature::build<void, TupleTableSlot *, int, int>());
  GetMissingAttrsInvoke->setArg(0, Slot);
  GetMissingAttrsInvoke->setArg(1, TtsNValid);
  GetMissingAttrsInvoke->setArg(2, jit::imm(NAttrs));

  Jitcc.mov(v_TtsNValid, jit::imm(NAttrs));

  Jitcc.bind(Out);

#endif
  Jitcc.ret();
  Jitcc.endFunc();
  Jitcc.finalize();

  return (TupleDeformingFunc)EmitJittedFunction(Context, Code);
}
}
