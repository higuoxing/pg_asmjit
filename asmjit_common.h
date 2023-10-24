#ifndef _PG_ASMJIT_H_
#define _PG_ASMJIT_H_

#include <asmjit/asmjit.h>

extern "C" {
#if __cplusplus > 199711L
#define register
#endif

#include "postgres.h"

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
#include "utils/resowner_private.h"

extern bool AsmJitCompileExpr(ExprState *State);
extern void AsmJitReleaseContext(JitContext *Ctx);
extern void AsmJitResetAfterError(void);
}

#endif
