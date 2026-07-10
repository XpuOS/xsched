# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17

HIPCC = hipcc
HIPCC_FLAGS = -O3 -std=c++17

# HIP library path and name — override for your platform:
#   Standard AMD ROCm:      make hip HIP_LIB_PATH=/opt/rocm/lib HIP_LIB=amdhip64
#   Hygon DTK 26.04:        make hip HIP_LIB_PATH=/opt/dtk-26.04/lib HIP_LIB=amdhip64
HIP_LIB_PATH ?= /opt/dtk-26.04/lib
HIP_LIB ?= amdhip64

# Target executable
TARGET = app

# Source files
SRC_CUDA = app.cu
SRC_HIP = app.hip

cuda: $(SRC_CUDA)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $<

hip: $(SRC_HIP)
	$(HIPCC) $(HIPCC_FLAGS) -L$(HIP_LIB_PATH) -l$(HIP_LIB) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

.PHONY: cuda hip clean
