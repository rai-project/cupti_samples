#
# Copyright 2015-2018 NVIDIA Corporation. All rights reserved.
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

nvlink_bandwidth: nvlink_bandwidth.$(OBJ)
	nvcc -o $@ nvlink_bandwidth.$(OBJ) $(LIBS)

nvlink_bandwidth.$(OBJ): nvlink_bandwidth.cu
	nvcc -c $(INCLUDES) $<

run: nvlink_bandwidth
	./$< --cpu-to-gpu

clean:
	rm -f nvlink_bandwidth nvlink_bandwidth.$(OBJ)

