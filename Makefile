MODULE_big = asmjit
PROJ_ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ASMJIT_SRCS = $(wildcard $(PROJ_ROOT_DIR)/deps/asmjit/asmjit/*/*.cpp)
OBJS = asmjit.o asmjit_deform.o asmjit_expr.o
OBJS += $(ASMJIT_SRCS:.cpp=.o)

# Headers live under deps/asmjit/asmjit/, so include deps/asmjit so that
# <asmjit/...> resolves correctly.  Keep deps/asmjit/src for backward
# compatibility with any remaining out-of-tree object files.
PG_CPPFLAGS += -Ideps/asmjit -Ideps/asmjit/src

# libasmjit is built in the `build` dir.
EXTRA_CLEAN = build

# We should use C++ compiler to link object files.
override COMPILER = $(CXX) $(CFLAGS)

# Disable the bitcode generation.
override with_llvm = no

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/pg_asmjit
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif
