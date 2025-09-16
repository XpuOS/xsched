WORK_PATH = $(CURDIR)
XSCHED_PATH = $(WORK_PATH)/../../../..
XSCHED_INCLUDE_PATH = $(XSCHED_PATH)/output/include
XSCHED_LIB_PATH = $(XSCHED_PATH)/output/lib

CXX = g++
CC = gcc
DEBUG_FLAGS = -g

ifeq ($(strip $(ONEAPI_ROOT)),)
  $(error ERROR: ONEAPI_ROOT is not set. Please set it to your oneAPI installation path, e.g., C:/Program Files (x86)/Intel/oneAPI)
endif
ifeq ($(strip $(LEVELZERO_ROOT)),)
  $(error ERROR: LEVELZERO_ROOT is not set. Please set it to point to the level-zero installation)
endif

CLANG = $(ONEAPI_ROOT)/2025.2/bin/compiler/clang.exe
LLVM_SPIRV = $(ONEAPI_ROOT)/2025.2/bin/compiler/llvm-spirv.exe

SRC = app_nontrans_sched.cpp
BIN = app_nontrans_sched.exe
KERNEL = vector_add

CXXFLAGS = -I"$(LEVELZERO_ROOT)/include" -I$(XSCHED_INCLUDE_PATH) -std=c++17
LDFLAGS = -L"$(XSCHED_LIB_PATH)" -lze_loader

.PHONY: all clean rebuild

all: $(BIN) $(KERNEL).spv

$(KERNEL).spv: $(KERNEL).cl
	$(CLANG) -c -target spir -O0 -emit-llvm -o $(KERNEL).bc $(KERNEL).cl
	$(LLVM_SPIRV) $(KERNEL).bc -o $@

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(DEBUG_FLAGS)

clean:
	@if exist "$(KERNEL).bc" del "$(KERNEL).bc"
	@if exist "$(KERNEL).spv" del "$(KERNEL).spv"
	@if exist "$(BIN)" del "$(BIN)"

rebuild: clean all