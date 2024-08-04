#include "asmjit/core/compiler.h"
#include "asmjit/core/func.h"
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
  Assert(TtsOps != &TTSOpsVirtual);

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
  x86::Gp Slot = Jitcc.newUIntPtr();
  JittedDeformingFunc->setArg(0, Slot);

  /*
   * Check which columns have to exist, so we don't have to check the row's
   * natts unnecessarily.
   */
  int GuaranteedColumnNumber = -1;
  for (int AttNum = 0; AttNum < Desc->natts; ++AttNum) {
    Form_pg_attribute Att = TupleDescAttr(Desc, AttNum);

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
    if (Att->attnotnull && !Att->atthasmissing && !Att->attisdropped)
      GuaranteedColumnNumber = AttNum;
  }

  x86::Gp SlotOffset, SlotTuple;
  if (TtsOps == &TTSOpsHeapTuple || TtsOps == &TTSOpsBufferHeapTuple) {
    SlotOffset = emit_load_off_from_HeapTupleTableSlot(Jitcc, Slot);
    SlotTuple = emit_load_tuple_from_HeapTupleTableSlot(Jitcc, Slot);
  } else if (TtsOps == &TTSOpsMinimalTuple) {
    SlotOffset = emit_load_off_from_MinimalTupleTableSlot(Jitcc, Slot);
    SlotTuple = emit_load_tuple_from_MinimalTupleTableSlot(Jitcc, Slot);
  } else {
    /* Should've returned at the start of the function. */
    pg_unreachable();
  }

  x86::Gp SlotTupleData = emit_load_t_data_from_HeapTupleData(Jitcc, SlotTuple);
  x86::Gp SlotTupleInfoMask = emit_load_t_infomask_from_HeapTupleHeaderData(
      Jitcc, SlotTupleData); /* uint16 */
  x86::Gp SlotTupleInfoMask2 = emit_load_t_infomask2_from_HeapTupleHeaderData(
      Jitcc, SlotTupleData); /* uint16 */

  /* t_infomask & HEAP_HASNULL */
  x86::Gp HasNulls = Jitcc.newUInt16("hasnulls.u16"),
          HasNullsBit = Jitcc.newUInt16("hasnullsbit.u16");
  Jitcc.mov(HasNulls, SlotTupleInfoMask);
  Jitcc.and_(HasNulls, jit::imm(HEAP_HASNULL));
  Jitcc.xor_(HasNullsBit, HasNullsBit);
  Jitcc.cmp(HasNulls, jit::imm(0));
  Jitcc.setne(HasNullsBit);

  x86::Gp MaxAtt = Jitcc.newUInt16("maxatt.u16");
  Jitcc.mov(MaxAtt, SlotTupleInfoMask2);
  Jitcc.and_(MaxAtt, jit::imm(HEAP_NATTS_MASK));
  x86::Gp MaxAttI32 = Jitcc.newInt32("maxatt.i32");
  Jitcc.movsx(MaxAttI32, MaxAtt);

  jit::Label L_SkipAdjustUnavailCols = Jitcc.newLabel();
  if (GuaranteedColumnNumber < NAtts - 1) {
    Jitcc.cmp(MaxAttI32, jit::imm(NAtts));
    Jitcc.jge(L_SkipAdjustUnavailCols);

    jit::InvokeNode *SlotGetMissingAttrs;
    Jitcc.invoke(&SlotGetMissingAttrs, jit::imm(slot_getmissingattrs),
                 jit::FuncSignature::build<void, TupleTableSlot *, int, int>());
    SlotGetMissingAttrs->setArg(0, Slot);
    SlotGetMissingAttrs->setArg(1, MaxAttI32);
    SlotGetMissingAttrs->setArg(2, jit::imm(NAtts));
  }

  Jitcc.bind(L_SkipAdjustUnavailCols);

  x86::Gp NValid =
      emit_load_tts_nvalid_from_TupleTableSlot(Jitcc, Slot); /* uint16 */

  /*
   * switch (NValid) {
   * case 0:
   *   ...; break;
   * case 1:
   *   ...; break;
   * ...
   * case Attnum - 1:
   *   ...; break;
   * }
   */
  x86::Gp JmpTableOff = Jitcc.newIntPtr("JmpTableOffset"),
          JmpTarget = Jitcc.newIntPtr("JmpTarget");
  jit::Label L_JmpTable = Jitcc.newLabel();
  jit::Label *L_CheckAttnoBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts),
             *L_CheckAlignBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts),
             *L_AttAlignBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts),
             *L_AttStoreBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * NAtts);

  jit::JumpAnnotation *JA = Jitcc.newJumpAnnotation();
  for (int I = 0; I < NAtts; ++I) {
    L_CheckAttnoBlocks[I] = Jitcc.newLabel();
    L_CheckAlignBlocks[I] = Jitcc.newLabel();
    L_AttAlignBlocks[I] = Jitcc.newLabel();
    L_AttStoreBlocks[I] = Jitcc.newLabel();
  }
  for (int I = 0; I < NAtts; ++I)
    JA->addLabel(L_CheckAttnoBlocks[I]);

  /* Calculate the correct jmp address. */
  Jitcc.lea(JmpTableOff, x86::ptr(L_JmpTable));

  x86::Gp NValidU32 = Jitcc.newUInt32("nvalid.u32");
  Jitcc.movzx(NValidU32, NValid);
  if (Jitcc.is64Bit()) {
    Jitcc.movsxd(JmpTarget, x86::dword_ptr(JmpTableOff,
                                           NValidU32.cloneAs(JmpTableOff), 2));

  } else {
    Jitcc.mov(JmpTarget,
              x86::dword_ptr(JmpTableOff, NValidU32.cloneAs(JmpTableOff), 2));
  }

  Jitcc.add(JmpTarget, JmpTableOff);
  Jitcc.jmp(JmpTarget, JA);

  jit::Label L_Out = Jitcc.newLabel();

  /* if true, known_alignment describes definite offset of column */
  bool AttGuaranteedAlign = true;
  /* current known alignment */
  int KnownAlignment = 0;

  /*
   * Iterate over each attribute that needs to be deformed, build code to
   * deform it.
   */
  for (int AttNum = 0; AttNum < NAtts; ++AttNum) {
    Form_pg_attribute Att = TupleDescAttr(Desc, AttNum);
    int AlignTo;

    /* attcheckattnoblock */
    Jitcc.bind(L_CheckAttnoBlocks[AttNum]);
    /*
     * If this is the first attribute, slot->tts_nvalid was 0. Therefore
     * also reset offset to 0, it may be from a previous execution.
     */
    if (AttNum == 0) {
      Jitcc.mov(SlotOffset, jit::imm(0));
    }

    if (AttNum > GuaranteedColumnNumber) {
      Jitcc.cmp(MaxAttI32, jit::imm(AttNum));
      Jitcc.jle(L_Out);
    }

    /* attstartblock */
    /*
     * Check for nulls if necessary. No need to take missing attributes
     * into account, because if they're present the heaptuple's natts
     * would have indicated that a slot_getmissingattrs() is needed.
     */
    if (!Att->attnotnull) {
      x86::Gp NullByteMask = EmitLoadConstUInt8(Jitcc, "nullbytemask.u8",
                                                (1 << (AttNum & 0x07))),
              NullByte = Jitcc.newUInt8("nullbyte.u8"),
              NullBit = Jitcc.newUInt8("nullbit.u8"),
              AttIsNull = Jitcc.newUInt16("attisnull.u16");

      EmitLoadFromFlexibleArray(Jitcc, SlotTupleData,
                                offsetof(HeapTupleHeaderData, t_bits),
                                (AttNum >> 3), NullByte, sizeof(uint8));

      Jitcc.and_(NullByte, NullByteMask);
      Jitcc.xor_(NullBit, NullBit);
      Jitcc.cmp(NullByte, jit::imm(0));
      Jitcc.sete(NullBit);
      Jitcc.movzx(AttIsNull, NullBit);
      Jitcc.and_(AttIsNull, HasNullsBit);

      Jitcc.cmp(AttIsNull, jit::imm(0));
      Jitcc.je(L_CheckAlignBlocks[AttNum]);

      /* store null-byte */
      x86::Gp TtsNulls = emit_load_tts_isnull_from_TupleTableSlot(Jitcc, Slot);
      EmitStoreToArray(Jitcc, TtsNulls, AttNum, jit::imm(1), sizeof(bool));

      /* store zero datum */
      x86::Gp TtsValues = emit_load_tts_values_from_TupleTableSlot(Jitcc, Slot);
      EmitStoreToArray(Jitcc, TtsValues, AttNum, jit::imm(0), sizeof(Datum));

      if (AttNum + 1 == NAtts) {
        Jitcc.jmp(L_Out);
      } else {
        Jitcc.jmp(L_CheckAttnoBlocks[AttNum + 1]);
      }
      AttGuaranteedAlign = false;
    }

    /* attcheckalignblock */
    Jitcc.bind(L_CheckAlignBlocks[AttNum]);

    /* Determine required alignment */
    if (Att->attalign == TYPALIGN_INT)
      AlignTo = ALIGNOF_INT;
    else if (Att->attalign == TYPALIGN_CHAR)
      AlignTo = 1;
    else if (Att->attalign == TYPALIGN_DOUBLE)
      AlignTo = ALIGNOF_DOUBLE;
    else if (Att->attalign == TYPALIGN_SHORT)
      AlignTo = ALIGNOF_SHORT;
    else {
      elog(ERROR, "unknown alignment");
      AlignTo = 0;
    }

    /* ------
     * Even if alignment is required, we can skip doing it if provably
     * unnecessary:
     * - first column is guaranteed to be aligned
     * - columns following a NOT NULL fixed width datum have known
     *   alignment, can skip alignment computation if that known alignment
     *   is compatible with current column.
     * ------
     */
    if (AlignTo > 1 && (KnownAlignment < 0 ||
                        KnownAlignment != TYPEALIGN(AlignTo, KnownAlignment))) {
      /*
       * When accessing a varlena field, we have to "peek" to see if we
       * are looking at a pad byte or the first byte of a 1-byte-header
       * datum.  A zero byte must be either a pad byte, or the first
       * byte of a correctly aligned 4-byte length word; in either case,
       * we can align safely.  A non-zero byte must be either a 1-byte
       * length word, or the first byte of a correctly aligned 4-byte
       * length word; in either case, we need not align.
       */
      if (Att->attlen == -1) {
        /* don't know if short varlena or not */
        AttGuaranteedAlign = false;
        x86::Gp IsPaded = Jitcc.newInt8("ispadded");

        {
          x86::Gp AttData = Jitcc.newUIntPtr("attdata.uintptr");
          x86::Gp HoffU8 = emit_load_t_hoff_from_HeapTupleHeaderData(
                      Jitcc, SlotTupleData),
                  HoffU32 = Jitcc.newUInt32("t_hoff.u32"),
                  HoffU64 = Jitcc.newUInt64("t_hoff.u64");
          Jitcc.movzx(HoffU32, HoffU8);
          Jitcc.add(HoffU32, SlotOffset);
          Jitcc.movzx(HoffU64, HoffU32);
          Jitcc.mov(AttData, SlotTupleData);
          Jitcc.add(AttData, HoffU64);
          x86::Gp PossiblePadByte = Jitcc.newInt8("possible_pad_byte.i8");
          x86::Mem AttDataPtr = x86::ptr(AttData, 0, sizeof(int8));
          Jitcc.mov(PossiblePadByte, AttDataPtr);
          Jitcc.xor_(IsPaded, IsPaded);
          Jitcc.cmp(PossiblePadByte, jit::imm(0));
          Jitcc.sete(IsPaded);
        }

        Jitcc.cmp(IsPaded, jit::imm(0));
        Jitcc.je(L_AttStoreBlocks[AttNum]);
      }

      /* attalignblock */
      Jitcc.bind(L_AttAlignBlocks[AttNum]);

      /* translation of alignment code (cf TYPEALIGN()) */
      {
        /*
         * uint32 alignval = alignto - 1;
         * uint32 lh = offset + alignval;
         * uint32 rh = ~(alignto - 1);
         * offset = lh & rh;
         */
        x86::Gp LH = Jitcc.newUInt32("lh.u32"), RH = Jitcc.newUInt32("rh.u32");
        uint32 AlignVal = (uint32)AlignTo - 1;
        Jitcc.mov(LH, jit::imm(AlignVal));
        Jitcc.add(LH, SlotOffset);

        Jitcc.mov(RH, jit::imm(AlignVal));
        Jitcc.not_(RH);

        Jitcc.and_(LH, RH);
        Jitcc.mov(SlotOffset, LH);
      }

      /*
       * As alignment either was unnecessary or has been performed, we
       * now know the current alignment. This is only safe because this
       * value isn't used for varlena and nullable columns.
       */
      if (KnownAlignment >= 0) {
        Assert(KnownAlignment != 0);
        KnownAlignment = TYPEALIGN(AlignTo, KnownAlignment);
      }
    }

    /* attstoreblock */
    Jitcc.bind(L_AttStoreBlocks[AttNum]);

    if (AttGuaranteedAlign) {
      Assert(KnownAlignment >= 0);
      Jitcc.mov(SlotOffset, jit::imm(KnownAlignment));
    }

    /* compute what following columns are aligned to */
    if (Att->attlen < 0) {
      /* can't guarantee any alignment after variable length field */
      KnownAlignment = -1;
      AttGuaranteedAlign = false;
    } else if (Att->attnotnull && AttGuaranteedAlign && KnownAlignment >= 0) {
      /*
       * If the offset to the column was previously known, a NOT NULL &
       * fixed-width column guarantees that alignment is just the
       * previous alignment plus column width.
       */
      Assert(Att->attlen > 0);
      KnownAlignment += Att->attlen;
    } else if (Att->attnotnull && (Att->attlen % AlignTo) == 0) {
      /*
       * After a NOT NULL fixed-width column with a length that is a
       * multiple of its alignment requirement, we know the following
       * column is aligned to at least the current column's alignment.
       */
      Assert(Att->attlen > 0);
      KnownAlignment = AlignTo;
      Assert(KnownAlignment > 0);
      AttGuaranteedAlign = false;
    } else {
      KnownAlignment = -1;
      AttGuaranteedAlign = false;
    }

    /* compute address to load data from */
    x86::Gp AttData = Jitcc.newUIntPtr("attdata.uintptr");
    {
      /*
       * int8 *tupdata_base = (int8 *)(tuplep);
       * attdata = &tupdata_base[tuplep->t_hoff + offset];
       */
      x86::Gp HoffU8 = emit_load_t_hoff_from_HeapTupleHeaderData(Jitcc,
                                                                 SlotTupleData),
              HoffU32 = Jitcc.newUInt32("t_hoff.u32"),
              HoffU64 = Jitcc.newUInt64("t_hoff.u64");
      Jitcc.movzx(HoffU32, HoffU8);
      Jitcc.add(HoffU32, SlotOffset);
      Jitcc.movzx(HoffU64, HoffU32);
      Jitcc.mov(AttData, SlotTupleData);
      Jitcc.add(AttData, HoffU64);
    }

    /* store null-byte (false) */
    x86::Gp TtsNulls = emit_load_tts_isnull_from_TupleTableSlot(Jitcc, Slot);
    EmitStoreToArray(Jitcc, TtsNulls, AttNum, jit::imm(0), sizeof(bool));

    /*
     * Store datum. For byval: datums copy the value, extend to Datum's
     * width, and store. For byref types: store pointer to data.
     */
    if (Att->attbyval) {
      x86::Gp TmpDatumI64 = Jitcc.newInt64("tmpdatum.i64");
      switch (Att->attlen) {
      case 1: {
        x86::Gp TmpDatum = Jitcc.newInt8("tmpdatum");
        EmitLoadFromArray(Jitcc, AttData, 0, TmpDatum, sizeof(int8));
        Jitcc.movsx(TmpDatumI64, TmpDatum);
        break;
      }
      case 2: {
        x86::Gp TmpDatum = Jitcc.newInt16("tmpdatum");
        EmitLoadFromArray(Jitcc, AttData, 0, TmpDatum, sizeof(int16));
        Jitcc.movsxd(TmpDatumI64, TmpDatum);
        break;
      }
      case 4: {
        x86::Gp TmpDatum = Jitcc.newInt32("tmpdatum");
        EmitLoadFromArray(Jitcc, AttData, 0, TmpDatum, sizeof(int32));
        Jitcc.movsxd(TmpDatumI64, TmpDatum);
        break;
      }
      case 8: {
        EmitLoadFromArray(Jitcc, AttData, 0, TmpDatumI64, sizeof(int64));
        break;
      }
      default:
        elog(ERROR, "unknown attlen: %d", Att->attlen);
      }
      /* Store value */
      x86::Gp TtsValues = emit_load_tts_values_from_TupleTableSlot(Jitcc, Slot);
      EmitStoreToArray(Jitcc, TtsValues, AttNum, TmpDatumI64, sizeof(Datum));
    } else {
      /* Store pointer */
      x86::Gp TtsValues = emit_load_tts_values_from_TupleTableSlot(Jitcc, Slot);
      EmitStoreToArray(Jitcc, TtsValues, AttNum, AttData, sizeof(Datum));
    }

    /* Increment data pointer. */
    x86::Gp IncrementBy = Jitcc.newUInt32("incrementby");
    if (Att->attlen > 0) {
      Jitcc.mov(IncrementBy, jit::imm(Att->attlen));
    } else if (Att->attlen == -1) {
      jit::InvokeNode *VarSizeAny;
      Jitcc.invoke(&VarSizeAny, jit::imm(varsize_any),
                   jit::FuncSignature::build<uint32, void *>());
      VarSizeAny->setArg(0, AttData);
      VarSizeAny->setRet(0, IncrementBy);
    } else if (Att->attlen == -2) {
      jit::InvokeNode *Strlen;
      Jitcc.invoke(&Strlen, jit::imm(strlen),
                   jit::FuncSignature::build<uint32, void *>());
      Strlen->setArg(0, AttData);
      Strlen->setRet(0, IncrementBy);
      /* Count the trailing '\0' in */
      Jitcc.inc(IncrementBy);
    } else {
      Assert(false);
      Jitcc.mov(IncrementBy, jit::imm(0));
    }

    if (AttGuaranteedAlign) {
      Assert(KnownAlignment >= 0);
      Jitcc.mov(SlotOffset, jit::imm(KnownAlignment));
    } else {
      Jitcc.add(SlotOffset, IncrementBy);
    }
  }

  /* Out block */
  Jitcc.bind(L_Out);

  {
    /* slot->tts_nvalid = natts; */
    emit_store_tts_nvalid_to_TupleTableSlot(Jitcc, Slot, jit::imm(NAtts));

    /* slot->off = off; */
    if (TtsOps == &TTSOpsHeapTuple || TtsOps == &TTSOpsBufferHeapTuple) {
      emit_store_off_to_HeapTupleTableSlot(Jitcc, Slot, SlotOffset);
    } else if (TtsOps == &TTSOpsMinimalTuple) {
      emit_store_off_to_MinimalTupleTableSlot(Jitcc, Slot, SlotOffset);
    } else {
      /* Should've returned at the start of the function. */
      pg_unreachable();
    }

    /* slot->tts_flags |= TTS_FLAG_SLOW; */
    x86::Gp TtsFlags = emit_load_tts_flags_from_TupleTableSlot(Jitcc, Slot);
    Jitcc.or_(TtsFlags, jit::imm(TTS_FLAG_SLOW));
    emit_store_tts_flags_to_TupleTableSlot(Jitcc, Slot, TtsFlags);
  }

  Jitcc.ret();
  Jitcc.endFunc();

  /* Embed the jump table for CheckAttnoBlocks. */
  Jitcc.bind(L_JmpTable);
  for (int I = 0; I < NAtts; ++I) {
    Jitcc.embedLabelDelta(L_CheckAttnoBlocks[I], L_JmpTable, 4);
  }

  Jitcc.finalize();

  return (TupleDeformingFunc)EmitJittedFunction(Context, Code);
}
}
