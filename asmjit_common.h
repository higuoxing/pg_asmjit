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

#endif
