#include "asmjit/core/compiler.h"
#include "asmjit/core/func.h"
#include "asmjit/x86/x86operand.h"
#include "asmjit_common.h"

namespace x86 = asmjit::x86;
namespace jit = asmjit;

extern "C" {

TupleDeformingFunc CompileTupleDeformingFunc(AsmJitContext *Context,
                                             jit::JitRuntime &Runtime,
                                             TupleDesc desc,
                                             const TupleTableSlotOps *tts_ops,
                                             int natts) {
  /* virtual tuples never need deforming, so don't generate code */
  Assert(tts_ops != &TTSOpsVirtual);

  /* decline to JIT for slot types we don't know to handle */
  if (tts_ops != &TTSOpsHeapTuple && tts_ops != &TTSOpsBufferHeapTuple &&
      tts_ops != &TTSOpsMinimalTuple)
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
  x86::Gp v_slot = Jitcc.newUIntPtr();
  JittedDeformingFunc->setArg(0, v_slot);

  /*
   * Check which columns have to exist, so we don't have to check the row's
   * natts unnecessarily.
   */
  int guaranteed_column_number = -1;
  for (int attnum = 0; attnum < desc->natts; ++attnum) {
    Form_pg_attribute att = TupleDescAttr(desc, attnum);

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
      guaranteed_column_number = attnum;
  }

  x86::Gp v_offset, v_tuple;
  if (tts_ops == &TTSOpsHeapTuple || tts_ops == &TTSOpsBufferHeapTuple) {
    v_offset = emit_load_off_from_HeapTupleTableSlot(Jitcc, v_slot);
    v_tuple = emit_load_tuple_from_HeapTupleTableSlot(Jitcc, v_slot);
  } else if (tts_ops == &TTSOpsMinimalTuple) {
    v_offset = emit_load_off_from_MinimalTupleTableSlot(Jitcc, v_slot);
    v_tuple = emit_load_tuple_from_MinimalTupleTableSlot(Jitcc, v_slot);
  } else {
    /* Should've returned at the start of the function. */
    pg_unreachable();
  }

  x86::Gp v_tuple_datap = emit_load_t_data_from_HeapTupleData(Jitcc, v_tuple);
  x86::Gp v_infomask1 = emit_load_t_infomask_from_HeapTupleHeaderData(
      Jitcc, v_tuple_datap); /* uint16 */
  x86::Gp v_infomask2 = emit_load_t_infomask2_from_HeapTupleHeaderData(
      Jitcc, v_tuple_datap); /* uint16 */

  /* t_infomask & HEAP_HASNULL */
  x86::Gp v_hasnulls = Jitcc.newUInt16("v_hasnulls.u16"),
          v_hasnullsbit = Jitcc.newUInt16("v_hasnullsbit.u16");
  Jitcc.mov(v_hasnulls, v_infomask1);
  Jitcc.and_(v_hasnulls, jit::imm(HEAP_HASNULL));
  Jitcc.xor_(v_hasnullsbit, v_hasnullsbit);
  Jitcc.cmp(v_hasnulls, jit::imm(0));
  Jitcc.setne(v_hasnullsbit);

  x86::Gp v_maxatt = Jitcc.newUInt16("v_maxatt.u16");
  Jitcc.mov(v_maxatt, v_infomask2);
  Jitcc.and_(v_maxatt, jit::imm(HEAP_NATTS_MASK));
  x86::Gp v_maxatt_i32 = Jitcc.newInt32("v_maxatt.i32");
  Jitcc.movsx(v_maxatt_i32, v_maxatt);

  jit::Label L_SkipAdjustUnavailCols = Jitcc.newLabel();
  if (guaranteed_column_number < natts - 1) {
    Jitcc.cmp(v_maxatt_i32, jit::imm(natts));
    Jitcc.jge(L_SkipAdjustUnavailCols);

    jit::InvokeNode *SlotGetMissingAttrs;
    Jitcc.invoke(&SlotGetMissingAttrs, jit::imm(slot_getmissingattrs),
                 jit::FuncSignature::build<void, TupleTableSlot *, int, int>());
    SlotGetMissingAttrs->setArg(0, v_slot);
    SlotGetMissingAttrs->setArg(1, v_maxatt_i32);
    SlotGetMissingAttrs->setArg(2, jit::imm(natts));
  }

  Jitcc.bind(L_SkipAdjustUnavailCols);

  x86::Gp v_nvalid =
      emit_load_tts_nvalid_from_TupleTableSlot(Jitcc, v_slot); /* uint16 */

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
  x86::Gp v_jump_table_off = Jitcc.newIntPtr("v_jump_table_off.intptr"),
          v_jump_target = Jitcc.newIntPtr("v_jump_target.intptr");
  jit::Label L_JmpTable = Jitcc.newLabel();
  jit::Label *L_CheckAttnoBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * natts),
             *L_CheckAlignBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * natts),
             *L_AttAlignBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * natts),
             *L_AttStoreBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * natts);

  jit::JumpAnnotation *JA = Jitcc.newJumpAnnotation();
  for (int attnum = 0; attnum < natts; ++attnum) {
    L_CheckAttnoBlocks[attnum] = Jitcc.newLabel();
    L_CheckAlignBlocks[attnum] = Jitcc.newLabel();
    L_AttAlignBlocks[attnum] = Jitcc.newLabel();
    L_AttStoreBlocks[attnum] = Jitcc.newLabel();
  }
  for (int I = 0; I < natts; ++I)
    JA->addLabel(L_CheckAttnoBlocks[I]);

  /* Calculate the correct jmp address. */
  Jitcc.lea(v_jump_table_off, x86::ptr(L_JmpTable));

  x86::Gp v_nvalid_u32 = Jitcc.newUInt32("v_nvalid.u32");
  Jitcc.movzx(v_nvalid_u32, v_nvalid);
  if (Jitcc.is64Bit()) {
    Jitcc.movsxd(v_jump_target,
                 x86::dword_ptr(v_jump_table_off,
                                v_nvalid_u32.cloneAs(v_jump_table_off), 2));

  } else {
    Jitcc.mov(v_jump_target,
              x86::dword_ptr(v_jump_table_off,
                             v_nvalid_u32.cloneAs(v_jump_table_off), 2));
  }

  Jitcc.add(v_jump_target, v_jump_table_off);
  Jitcc.jmp(v_jump_target, JA);

  jit::Label L_Out = Jitcc.newLabel();

  /* if true, known_alignment describes definite offset of column */
  bool att_guaranteed_align = true;
  /* current known alignment */
  int known_alignment = 0;

  /*
   * Iterate over each attribute that needs to be deformed, build code to
   * deform it.
   */
  for (int attnum = 0; attnum < natts; ++attnum) {
    Form_pg_attribute att = TupleDescAttr(desc, attnum);
    int alignto;

    /* attcheckattnoblock */
    Jitcc.bind(L_CheckAttnoBlocks[attnum]);
    /*
     * If this is the first attribute, slot->tts_nvalid was 0. Therefore
     * also reset offset to 0, it may be from a previous execution.
     */
    if (attnum == 0) {
      Jitcc.mov(v_offset, jit::imm(0));
    }

    if (attnum > guaranteed_column_number) {
      Jitcc.cmp(v_maxatt_i32, jit::imm(attnum));
      Jitcc.jle(L_Out);
    }

    /* attstartblock */
    /*
     * Check for nulls if necessary. No need to take missing attributes
     * into account, because if they're present the heaptuple's natts
     * would have indicated that a slot_getmissingattrs() is needed.
     */
    if (!att->attnotnull) {
      x86::Gp v_nullbytemask = EmitLoadConstUInt8(Jitcc, "v_nullbytemask.u8",
                                                  (1 << (attnum & 0x07))),
              v_nullbyte = Jitcc.newUInt8("v_nullbyte.u8"),
              v_nullbit = Jitcc.newUInt8("v_nullbit.u8"),
              v_attisnull = Jitcc.newUInt16("v_attisnull.u16");

      EmitLoadFromFlexibleArray(Jitcc, v_tuple_datap,
                                offsetof(HeapTupleHeaderData, t_bits),
                                (attnum >> 3), v_nullbyte, sizeof(uint8));

      Jitcc.and_(v_nullbyte, v_nullbytemask);
      Jitcc.xor_(v_nullbit, v_nullbit);
      Jitcc.cmp(v_nullbyte, jit::imm(0));
      Jitcc.sete(v_nullbit);
      Jitcc.movzx(v_attisnull, v_nullbit);
      Jitcc.and_(v_attisnull, v_hasnullsbit);

      Jitcc.cmp(v_attisnull, jit::imm(0));
      Jitcc.je(L_CheckAlignBlocks[attnum]);

      /* store null-byte */
      x86::Gp v_tts_nulls =
          emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_slot);
      EmitStoreToArray(Jitcc, v_tts_nulls, attnum, jit::imm(1), sizeof(bool));

      /* store zero datum */
      x86::Gp v_tts_values =
          emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot);
      EmitStoreToArray(Jitcc, v_tts_values, attnum, jit::imm(0), sizeof(Datum));

      if (attnum + 1 == natts) {
        Jitcc.jmp(L_Out);
      } else {
        Jitcc.jmp(L_CheckAttnoBlocks[attnum + 1]);
      }
      att_guaranteed_align = false;
    }

    /* attcheckalignblock */
    Jitcc.bind(L_CheckAlignBlocks[attnum]);

    /* Determine required alignment */
    if (att->attalign == TYPALIGN_INT)
      alignto = ALIGNOF_INT;
    else if (att->attalign == TYPALIGN_CHAR)
      alignto = 1;
    else if (att->attalign == TYPALIGN_DOUBLE)
      alignto = ALIGNOF_DOUBLE;
    else if (att->attalign == TYPALIGN_SHORT)
      alignto = ALIGNOF_SHORT;
    else {
      elog(ERROR, "unknown alignment");
      alignto = 0;
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
    if (alignto > 1 &&
        (known_alignment < 0 ||
         known_alignment != TYPEALIGN(alignto, known_alignment))) {
      /*
       * When accessing a varlena field, we have to "peek" to see if we
       * are looking at a pad byte or the first byte of a 1-byte-header
       * datum.  A zero byte must be either a pad byte, or the first
       * byte of a correctly aligned 4-byte length word; in either case,
       * we can align safely.  A non-zero byte must be either a 1-byte
       * length word, or the first byte of a correctly aligned 4-byte
       * length word; in either case, we need not align.
       */
      if (att->attlen == -1) {
        /* don't know if short varlena or not */
        att_guaranteed_align = false;
        x86::Gp v_ispaded = Jitcc.newInt8("ispadded");

        {
          x86::Gp attdata = Jitcc.newUIntPtr("attdata.uintptr");
          x86::Gp v_hoff_u8 = emit_load_t_hoff_from_HeapTupleHeaderData(
                      Jitcc, v_tuple_datap),
                  v_hoff_u32 = Jitcc.newUInt32("t_hoff.u32"),
                  v_hoff_u64 = Jitcc.newUInt64("t_hoff.u64");
          Jitcc.movzx(v_hoff_u32, v_hoff_u8);
          Jitcc.add(v_hoff_u32, v_offset);
          Jitcc.movzx(v_hoff_u64, v_hoff_u32);
          Jitcc.mov(attdata, v_tuple_datap);
          Jitcc.add(attdata, v_hoff_u64);
          x86::Gp v_possible_pad_byte = Jitcc.newInt8("v_possible_pad_byte.i8");
          x86::Mem m_attdatap = x86::ptr(attdata, 0, sizeof(int8));
          Jitcc.mov(v_possible_pad_byte, m_attdatap);
          Jitcc.xor_(v_ispaded, v_ispaded);
          Jitcc.cmp(v_possible_pad_byte, jit::imm(0));
          Jitcc.sete(v_ispaded);
        }

        Jitcc.cmp(v_ispaded, jit::imm(0));
        Jitcc.je(L_AttStoreBlocks[attnum]);
      }

      /* attalignblock */
      Jitcc.bind(L_AttAlignBlocks[attnum]);

      /* translation of alignment code (cf TYPEALIGN()) */
      {
        /*
         * uint32 alignval = alignto - 1;
         * uint32 lh = offset + alignval;
         * uint32 rh = ~(alignto - 1);
         * offset = lh & rh;
         */
        x86::Gp v_lh = Jitcc.newUInt32("lh.u32"),
                v_rh = Jitcc.newUInt32("rh.u32");
        uint32 alignval = (uint32)alignto - 1;
        Jitcc.mov(v_lh, jit::imm(alignval));
        Jitcc.add(v_lh, v_offset);

        Jitcc.mov(v_rh, jit::imm(alignval));
        Jitcc.not_(v_rh);

        Jitcc.and_(v_lh, v_rh);
        Jitcc.mov(v_offset, v_lh);
      }

      /*
       * As alignment either was unnecessary or has been performed, we
       * now know the current alignment. This is only safe because this
       * value isn't used for varlena and nullable columns.
       */
      if (known_alignment >= 0) {
        Assert(known_alignment != 0);
        known_alignment = TYPEALIGN(alignto, known_alignment);
      }
    }

    /* attstoreblock */
    Jitcc.bind(L_AttStoreBlocks[attnum]);

    if (att_guaranteed_align) {
      Assert(known_alignment >= 0);
      Jitcc.mov(v_offset, jit::imm(known_alignment));
    }

    /* compute what following columns are aligned to */
    if (att->attlen < 0) {
      /* can't guarantee any alignment after variable length field */
      known_alignment = -1;
      att_guaranteed_align = false;
    } else if (att->attnotnull && att_guaranteed_align &&
               known_alignment >= 0) {
      /*
       * If the offset to the column was previously known, a NOT NULL &
       * fixed-width column guarantees that alignment is just the
       * previous alignment plus column width.
       */
      Assert(att->attlen > 0);
      known_alignment += att->attlen;
    } else if (att->attnotnull && (att->attlen % alignto) == 0) {
      /*
       * After a NOT NULL fixed-width column with a length that is a
       * multiple of its alignment requirement, we know the following
       * column is aligned to at least the current column's alignment.
       */
      Assert(att->attlen > 0);
      known_alignment = alignto;
      Assert(known_alignment > 0);
      att_guaranteed_align = false;
    } else {
      known_alignment = -1;
      att_guaranteed_align = false;
    }

    /* compute address to load data from */
    x86::Gp v_attdatap = Jitcc.newUIntPtr("v_attdatap.uintptr");
    {
      /*
       * int8 *tupdata_base = (int8 *)(tuplep);
       * attdata = &tupdata_base[tuplep->t_hoff + offset];
       */
      x86::Gp v_hoff_u8 = emit_load_t_hoff_from_HeapTupleHeaderData(
                  Jitcc, v_tuple_datap),
              v_hoff_u32 = Jitcc.newUInt32("t_hoff.u32"),
              v_hoff_u64 = Jitcc.newUInt64("t_hoff.u64");
      Jitcc.movzx(v_hoff_u32, v_hoff_u8);
      Jitcc.add(v_hoff_u32, v_offset);
      Jitcc.movzx(v_hoff_u64, v_hoff_u32);
      Jitcc.mov(v_attdatap, v_tuple_datap);
      Jitcc.add(v_attdatap, v_hoff_u64);
    }

    /* store null-byte (false) */
    x86::Gp v_tts_nulls =
        emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_slot);
    EmitStoreToArray(Jitcc, v_tts_nulls, attnum, jit::imm(0), sizeof(bool));

    /*
     * Store datum. For byval: datums copy the value, extend to Datum's
     * width, and store. For byref types: store pointer to data.
     */
    if (att->attbyval) {
      x86::Gp v_tmp_datum_i64 = Jitcc.newInt64("v_tmpdatum.i64");
      switch (att->attlen) {
      case 1: {
        x86::Gp v_tmp_datum = Jitcc.newInt8("v_tmpdatum");
        EmitLoadFromArray(Jitcc, v_attdatap, 0, v_tmp_datum, sizeof(int8));
        Jitcc.movsx(v_tmp_datum_i64, v_tmp_datum);
        break;
      }
      case 2: {
        x86::Gp v_tmp_datum = Jitcc.newInt16("tmpdatum");
        EmitLoadFromArray(Jitcc, v_attdatap, 0, v_tmp_datum, sizeof(int16));
        Jitcc.movsxd(v_tmp_datum_i64, v_tmp_datum);
        break;
      }
      case 4: {
        x86::Gp v_tmp_datum = Jitcc.newInt32("tmpdatum");
        EmitLoadFromArray(Jitcc, v_attdatap, 0, v_tmp_datum, sizeof(int32));
        Jitcc.movsxd(v_tmp_datum_i64, v_tmp_datum);
        break;
      }
      case 8: {
        EmitLoadFromArray(Jitcc, v_attdatap, 0, v_tmp_datum_i64, sizeof(int64));
        break;
      }
      default:
        elog(ERROR, "unknown attlen: %d", att->attlen);
      }
      /* Store value */
      x86::Gp v_tts_values =
          emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot);
      EmitStoreToArray(Jitcc, v_tts_values, attnum, v_tmp_datum_i64,
                       sizeof(Datum));
    } else {
      /* Store pointer */
      x86::Gp v_tts_values =
          emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot);
      EmitStoreToArray(Jitcc, v_tts_values, attnum, v_attdatap, sizeof(Datum));
    }

    /* Increment data pointer. */
    x86::Gp v_incrby = Jitcc.newUInt32("incrementby");
    if (att->attlen > 0) {
      Jitcc.mov(v_incrby, jit::imm(att->attlen));
    } else if (att->attlen == -1) {
      jit::InvokeNode *InvokeVarSizeAny;
      Jitcc.invoke(&InvokeVarSizeAny, jit::imm(varsize_any),
                   jit::FuncSignature::build<uint32, void *>());
      InvokeVarSizeAny->setArg(0, v_attdatap);
      InvokeVarSizeAny->setRet(0, v_incrby);
    } else if (att->attlen == -2) {
      jit::InvokeNode *InvokeStrLen;
      Jitcc.invoke(&InvokeStrLen, jit::imm(strlen),
                   jit::FuncSignature::build<uint32, void *>());
      InvokeStrLen->setArg(0, v_attdatap);
      InvokeStrLen->setRet(0, v_incrby);
      /* Count the trailing '\0' in */
      Jitcc.inc(v_incrby);
    } else {
      Assert(false);
      Jitcc.mov(v_incrby, jit::imm(0));
    }

    if (att_guaranteed_align) {
      Assert(known_alignment >= 0);
      Jitcc.mov(v_offset, jit::imm(known_alignment));
    } else {
      Jitcc.add(v_offset, v_incrby);
    }
  }

  /* Out block */
  Jitcc.bind(L_Out);

  {
    /* slot->tts_nvalid = natts; */
    emit_store_tts_nvalid_to_TupleTableSlot(Jitcc, v_slot, jit::imm(natts));

    /* slot->off = off; */
    if (tts_ops == &TTSOpsHeapTuple || tts_ops == &TTSOpsBufferHeapTuple) {
      emit_store_off_to_HeapTupleTableSlot(Jitcc, v_slot, v_offset);
    } else if (tts_ops == &TTSOpsMinimalTuple) {
      emit_store_off_to_MinimalTupleTableSlot(Jitcc, v_slot, v_offset);
    } else {
      /* Should've returned at the start of the function. */
      pg_unreachable();
    }

    /* slot->tts_flags |= TTS_FLAG_SLOW; */
    x86::Gp v_tts_flags =
        emit_load_tts_flags_from_TupleTableSlot(Jitcc, v_slot);
    Jitcc.or_(v_tts_flags, jit::imm(TTS_FLAG_SLOW));
    emit_store_tts_flags_to_TupleTableSlot(Jitcc, v_slot, v_tts_flags);
  }

  Jitcc.ret();
  Jitcc.endFunc();

  /* Embed the jump table for CheckAttnoBlocks. */
  Jitcc.bind(L_JmpTable);
  for (int I = 0; I < natts; ++I) {
    Jitcc.embedLabelDelta(L_CheckAttnoBlocks[I], L_JmpTable, 4);
  }

  Jitcc.finalize();

  return (TupleDeformingFunc)EmitJittedFunction(Context, Code);
}
}
