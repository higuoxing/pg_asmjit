MODULE_big = asmjit
EXTENSION = asmjit
PROJ_ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

OBJS = asmjit.o

# Disable the bitcode generation.
override with_llvm = no

# Build the libasmjit static library first.
SHLIB_PREREQS += libasmjit
PG_CPPFLAGS += -Ideps/asmjit/src
SHLIB_LINK += -L$(PROJ_ROOT_DIR)/build/deps/libasmjit -lasmjit

# libasmjit is built in the `build` dir.
EXTRA_CLEAN = build

# We should use C++ compiler to link object files.
override COMPILER = $(CXX) $(CFLAGS)

PG_CONFIG := pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# We link against the asmjit library statically.
libasmjit:
	@mkdir -p $(PROJ_ROOT_DIR)/build/deps/libasmjit
	cd $(PROJ_ROOT_DIR)/build/deps/libasmjit && cmake -DASMJIT_STATIC=on -DCMAKE_POSITION_INDEPENDENT_CODE=on $(PROJ_ROOT_DIR)/deps/asmjit
	cd $(PROJ_ROOT_DIR)/build/deps/libasmjit && make -j`nproc`
