#
# Copyright 2014-2018 NVIDIA Corporation. All rights reserved.
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

event_multi_gpu: event_multi_gpu.$(OBJ)
	nvcc -o $@ event_multi_gpu.$(OBJ) $(LIBS)

event_multi_gpu.$(OBJ): event_multi_gpu.cu
	nvcc -c $(INCLUDES) $<

run: event_multi_gpu
	./$<

clean:
	rm -f event_multi_gpu event_multi_gpu.$(OBJ)

