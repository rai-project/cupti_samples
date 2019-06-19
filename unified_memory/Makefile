#
# Copyright 2013-2018 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I../../include

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

ifeq ($(OS),Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    LIB_PATH ?= ../../lib64
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS= -lcuda -L $(LIB_PATH) -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS= -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
        LIBS= -lcuda -L $(LIB_PATH) -lcupti
    endif
    OBJ = o
endif

ifeq ($(OS), Darwin)
unified_memory:
	@echo "Unified memory is not supported on MacOSX. Build/clean/run aborted."
else
unified_memory: unified_memory.$(OBJ)
	nvcc -o $@ unified_memory.$(OBJ) $(LIBS)

unified_memory.$(OBJ): unified_memory.cu
	nvcc -c $(INCLUDES) $<

run: unified_memory
	./$<

clean:
	rm -f unified_memory unified_memory.$(OBJ)
endif

