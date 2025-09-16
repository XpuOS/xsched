WORK_PATH = $(CURDIR)
XSCHED_PATH = $(WORK_PATH)/../../..
XSCHED_INCLUDE_PATH = $(XSCHED_PATH)/output/include
XSCHED_LIB_PATH = $(XSCHED_PATH)/output/lib
CUDA_LIB_PATH = "$(CUDA_PATH)/lib/x64"

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -I$(XSCHED_INCLUDE_PATH)

DRIVER_LIB = nvcuda
NVCC_LDFLAGS = -L$(XSCHED_LIB_PATH) -l$(DRIVER_LIB) -L$(CUDA_LIB_PATH) -lcudart

# Target executable
TARGET = app_with_hints.exe

# Source files
SRC = app_with_hints.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $<

clean:
	del *.exe

.PHONY: all clean
