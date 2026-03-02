#include "asmjit_common.h"

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

  jit::CodeHolder Code;
  Code.init(Runtime.environment(), Runtime.cpu_features());
  arch::Compiler Jitcc(&Code);

  /*
   * void (*TupleDeformingFunc) (TupleTableSlot *);
   */
  jit::FuncNode *JittedDeformingFunc =
      Jitcc.add_func(jit::FuncSignature::build<void, TupleTableSlot *>());
  arch::Gp v_slot = Jitcc.new_gp_ptr("v_slot");
  JittedDeformingFunc->set_arg(0, v_slot);

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

  arch::Gp v_offset, v_tuple;
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

  arch::Gp v_tuple_datap =
      emit_load_t_data_from_HeapTupleData(Jitcc, v_tuple);
  arch::Gp v_infomask1 =
      emit_load_t_infomask_from_HeapTupleHeaderData(Jitcc, v_tuple_datap);
  arch::Gp v_infomask2 =
      emit_load_t_infomask2_from_HeapTupleHeaderData(Jitcc, v_tuple_datap);

  /* t_infomask & HEAP_HASNULL */
  arch::Gp v_hasnulls = Jitcc.new_gp32("v_hasnulls"),
           v_hasnullsbit = Jitcc.new_gp32("v_hasnullsbit");
  Jitcc.mov(v_hasnulls, v_infomask1);
  EmitBitwiseAndImm(Jitcc, v_hasnulls, HEAP_HASNULL);
  /* v_hasnullsbit = (v_hasnulls != 0), i.e. tuple has nulls */
  {
    arch::Gp v_tmp = Jitcc.new_gp32("v_tmp");
    EmitSetEQ(Jitcc, v_tmp, v_hasnulls, 0); /* tmp = (hasnulls == 0) */
    /* hasnullsbit = 1 ^ tmp = (hasnulls != 0) */
    arch::Gp v_one = EmitLoadConstUInt32(Jitcc, "one", 1);
    Jitcc.mov(v_hasnullsbit, v_one);
    EmitBitwiseXor(Jitcc, v_hasnullsbit, v_tmp);
  }

  arch::Gp v_maxatt = Jitcc.new_gp32("v_maxatt");
  Jitcc.mov(v_maxatt, v_infomask2);
  EmitBitwiseAndImm(Jitcc, v_maxatt, HEAP_NATTS_MASK);

  jit::Label L_SkipAdjustUnavailCols = Jitcc.new_label();
  if (guaranteed_column_number < natts - 1) {
    EmitCondJumpGE(Jitcc, v_maxatt, natts, L_SkipAdjustUnavailCols);

    jit::InvokeNode *SlotGetMissingAttrs;
    Jitcc.invoke(asmjit::Out(SlotGetMissingAttrs), JIT_FN_PTR(Jitcc, slot_getmissingattrs),
                 jit::FuncSignature::build<void, TupleTableSlot *, int, int>());
    SlotGetMissingAttrs->set_arg(0, v_slot);
    SlotGetMissingAttrs->set_arg(1, v_maxatt);
    SlotGetMissingAttrs->set_arg(2, jit::imm(natts));
  }

  Jitcc.bind(L_SkipAdjustUnavailCols);

  arch::Gp v_nvalid =
      emit_load_tts_nvalid_from_TupleTableSlot(Jitcc, v_slot);

  /*
   * Dispatch to the correct CheckAttnoBlock based on nvalid.
   *
   * We emit a linear chain of comparisons:
   *   if (nvalid >= natts) goto L_Out;
   *   if (nvalid == natts-1) goto L_CheckAttnoBlocks[natts-1];
   *   ...
   *   if (nvalid == 0) goto L_CheckAttnoBlocks[0];
   *
   * This is architecture-neutral and correct, if less cache-friendly for
   * large natts than the x86 jump-table approach. The interpreter uses the
   * same linear-scan approach (slot_getmissingattrs).
   */
  jit::Label *L_CheckAttnoBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * natts),
             *L_CheckAlignBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * natts),
             *L_AttAlignBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * natts),
             *L_AttStoreBlocks =
                 (jit::Label *)palloc(sizeof(jit::Label) * natts);

  for (int attnum = 0; attnum < natts; ++attnum) {
    L_CheckAttnoBlocks[attnum] = Jitcc.new_label();
    L_CheckAlignBlocks[attnum] = Jitcc.new_label();
    L_AttAlignBlocks[attnum] = Jitcc.new_label();
    L_AttStoreBlocks[attnum] = Jitcc.new_label();
  }

  jit::Label L_Out = Jitcc.new_label();

  /* if nvalid >= natts, nothing to do */
  EmitCondJumpGE(Jitcc, v_nvalid, natts, L_Out);

  /* dispatch: jump to L_CheckAttnoBlocks[nvalid] */
  for (int I = natts - 1; I >= 1; --I) {
    EmitCondJumpEQ(Jitcc, v_nvalid, I, L_CheckAttnoBlocks[I]);
  }
  /* fall through to block 0 */
  EmitJump(Jitcc, L_CheckAttnoBlocks[0]);

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
      EmitCondJumpLE(Jitcc, v_maxatt, attnum, L_Out);
    }

    /* attstartblock */
    /*
     * Check for nulls if necessary. No need to take missing attributes
     * into account, because if they're present the heaptuple's natts
     * would have indicated that a slot_getmissingattrs() is needed.
     */
    if (!att->attnotnull) {
      arch::Gp v_nullbytemask =
                   EmitLoadConstUInt8(Jitcc, "v_nullbytemask",
                                      (uint8_t)(1 << (attnum & 0x07))),
               v_nullbyte = Jitcc.new_gp32("v_nullbyte"),
               v_nullbit = Jitcc.new_gp32("v_nullbit"),
               v_attisnull = Jitcc.new_gp32("v_attisnull");

      EmitLoadFromFlexibleArray(Jitcc, v_tuple_datap,
                                offsetof(HeapTupleHeaderData, t_bits),
                                (attnum >> 3), v_nullbyte, sizeof(uint8));

      EmitBitwiseAnd(Jitcc, v_nullbyte, v_nullbytemask);
      EmitSetEQ(Jitcc, v_nullbit, v_nullbyte, 0);
      Jitcc.mov(v_attisnull, v_nullbit);
      EmitBitwiseAnd(Jitcc, v_attisnull, v_hasnullsbit);

      EmitCondJumpEQ(Jitcc, v_attisnull, 0, L_CheckAlignBlocks[attnum]);

      /* store null-byte */
      arch::Gp v_tts_nulls =
          emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_slot);
      EmitStoreToArray(Jitcc, v_tts_nulls, attnum, jit::imm(1), sizeof(bool));

      /* store zero datum */
      arch::Gp v_tts_values =
          emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot);
      EmitStoreToArray(Jitcc, v_tts_values, attnum, jit::imm(0), sizeof(Datum));

      if (attnum + 1 == natts) {
        EmitJump(Jitcc, L_Out);
      } else {
        EmitJump(Jitcc, L_CheckAttnoBlocks[attnum + 1]);
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
        arch::Gp v_ispaded = Jitcc.new_gp32("ispadded");

        {
          arch::Gp attdata = Jitcc.new_gp_ptr("attdata");
          arch::Gp v_hoff_u8 = emit_load_t_hoff_from_HeapTupleHeaderData(
                      Jitcc, v_tuple_datap),
                   v_hoff_u32 = Jitcc.new_gp32("t_hoff.u32"),
                   v_hoff_u64 = Jitcc.new_gp64("t_hoff.u64");
          EmitZeroExtend8(Jitcc, v_hoff_u32, v_hoff_u8);
          EmitAddReg(Jitcc, v_hoff_u32, v_offset);
          EmitZeroExtend32to64(Jitcc, v_hoff_u64, v_hoff_u32);
          Jitcc.mov(attdata, v_tuple_datap);
          EmitAddReg(Jitcc, attdata, v_hoff_u64);
          arch::Gp v_possible_pad_byte = Jitcc.new_gp32("v_possible_pad_byte");
          EmitLoadFromArray(Jitcc, attdata, 0, v_possible_pad_byte,
                            sizeof(int8));
          EmitSetEQ(Jitcc, v_ispaded, v_possible_pad_byte, 0);
        }

        EmitCondJumpEQ(Jitcc, v_ispaded, 0, L_AttStoreBlocks[attnum]);
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
        uint32 alignval = (uint32)alignto - 1;
        arch::Gp v_lh = Jitcc.new_gp32("lh");
        Jitcc.mov(v_lh, jit::imm(alignval));
        EmitAddReg(Jitcc, v_lh, v_offset);
        EmitBitwiseAndImm(Jitcc, v_lh, ~(int64_t)alignval);
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
      Assert(att->attlen > 0);
      known_alignment += att->attlen;
    } else if (att->attnotnull && (att->attlen % alignto) == 0) {
      Assert(att->attlen > 0);
      known_alignment = alignto;
      Assert(known_alignment > 0);
      att_guaranteed_align = false;
    } else {
      known_alignment = -1;
      att_guaranteed_align = false;
    }

    /* compute address to load data from */
    arch::Gp v_attdatap = Jitcc.new_gp_ptr("v_attdatap");
    {
      /*
       * int8 *tupdata_base = (int8 *)(tuplep);
       * attdata = &tupdata_base[tuplep->t_hoff + offset];
       */
      arch::Gp v_hoff_u8 = emit_load_t_hoff_from_HeapTupleHeaderData(
                  Jitcc, v_tuple_datap),
               v_hoff_u32 = Jitcc.new_gp32("t_hoff.u32"),
               v_hoff_u64 = Jitcc.new_gp64("t_hoff.u64");
      EmitZeroExtend8(Jitcc, v_hoff_u32, v_hoff_u8);
      EmitAddReg(Jitcc, v_hoff_u32, v_offset);
      EmitZeroExtend32to64(Jitcc, v_hoff_u64, v_hoff_u32);
      Jitcc.mov(v_attdatap, v_tuple_datap);
      EmitAddReg(Jitcc, v_attdatap, v_hoff_u64);
    }

    /* store null-byte (false) */
    arch::Gp v_tts_nulls =
        emit_load_tts_isnull_from_TupleTableSlot(Jitcc, v_slot);
    EmitStoreToArray(Jitcc, v_tts_nulls, attnum, jit::imm(0), sizeof(bool));

    /*
     * Store datum. For byval: datums copy the value, extend to Datum's
     * width, and store. For byref types: store pointer to data.
     */
    if (att->attbyval) {
      arch::Gp v_tmp_datum = Jitcc.new_gp64("v_tmpdatum");
      switch (att->attlen) {
      case 1: {
        arch::Gp v_raw = Jitcc.new_gp32("v_raw");
        EmitLoadFromArray(Jitcc, v_attdatap, 0, v_raw, sizeof(int8));
        EmitSignExtend8to64(Jitcc, v_tmp_datum, v_raw);
        break;
      }
      case 2: {
        arch::Gp v_raw = Jitcc.new_gp32("v_raw");
        EmitLoadFromArray(Jitcc, v_attdatap, 0, v_raw, sizeof(int16));
        EmitSignExtend16to32(Jitcc, v_tmp_datum, v_raw);
        break;
      }
      case 4: {
        arch::Gp v_raw = Jitcc.new_gp32("v_raw");
        EmitLoadFromArray(Jitcc, v_attdatap, 0, v_raw, sizeof(int32));
        EmitSignExtend32to64(Jitcc, v_tmp_datum, v_raw);
        break;
      }
      case 8: {
        EmitLoadFromArray(Jitcc, v_attdatap, 0, v_tmp_datum, sizeof(int64));
        break;
      }
      default:
        elog(ERROR, "unknown attlen: %d", att->attlen);
      }
      /* Store value */
      arch::Gp v_tts_values =
          emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot);
      EmitStoreToArray(Jitcc, v_tts_values, attnum, v_tmp_datum, sizeof(Datum));
    } else {
      /* Store pointer */
      arch::Gp v_tts_values =
          emit_load_tts_values_from_TupleTableSlot(Jitcc, v_slot);
      EmitStoreToArray(Jitcc, v_tts_values, attnum, v_attdatap, sizeof(Datum));
    }

    /* Increment data pointer. */
    arch::Gp v_incrby = Jitcc.new_gp32("incrementby");
    if (att->attlen > 0) {
      Jitcc.mov(v_incrby, jit::imm(att->attlen));
    } else if (att->attlen == -1) {
      jit::InvokeNode *InvokeVarSizeAny;
      Jitcc.invoke(asmjit::Out(InvokeVarSizeAny), JIT_FN_PTR(Jitcc, varsize_any),
                   jit::FuncSignature::build<uint32, void *>());
      InvokeVarSizeAny->set_arg(0, v_attdatap);
      InvokeVarSizeAny->set_ret(0, v_incrby);
    } else if (att->attlen == -2) {
      jit::InvokeNode *InvokeStrLen;
      Jitcc.invoke(asmjit::Out(InvokeStrLen), JIT_FN_PTR(Jitcc, strlen),
                   jit::FuncSignature::build<uint32, void *>());
      InvokeStrLen->set_arg(0, v_attdatap);
      InvokeStrLen->set_ret(0, v_incrby);
      /* Count the trailing '\0' in */
      EmitInc(Jitcc, v_incrby);
    } else {
      Assert(false);
      Jitcc.mov(v_incrby, jit::imm(0));
    }

    if (att_guaranteed_align) {
      Assert(known_alignment >= 0);
      Jitcc.mov(v_offset, jit::imm(known_alignment));
    } else {
      EmitAddReg(Jitcc, v_offset, v_incrby);
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
    arch::Gp v_tts_flags =
        emit_load_tts_flags_from_TupleTableSlot(Jitcc, v_slot);
    EmitBitwiseOrImm(Jitcc, v_tts_flags, TTS_FLAG_SLOW);
    emit_store_tts_flags_to_TupleTableSlot(Jitcc, v_slot, v_tts_flags);
  }

  Jitcc.ret();
  Jitcc.end_func();

  Jitcc.finalize();

  return (TupleDeformingFunc)EmitJittedFunction(Context, Code);
}
}
