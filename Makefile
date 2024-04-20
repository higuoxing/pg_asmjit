MODULE_big = asmjit
EXTENSION = asmjit
PROJ_ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ASMJIT_SRCS = $(wildcard $(PROJ_ROOT_DIR)/deps/asmjit/src/*/*/*.cpp)
OBJS = asmjit.o asmjit_deform.o asmjit_expr.o
OBJS += $(ASMJIT_SRCS:.cpp=.o)

PG_CPPFLAGS += -Ideps/asmjit/src

# libasmjit is built in the `build` dir.
EXTRA_CLEAN = build

# We should use C++ compiler to link object files.
override COMPILER = $(CXX) $(CFLAGS)

# Disable the bitcode generation.
override with_llvm = no

PG_CONFIG := pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
