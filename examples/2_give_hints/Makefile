WORK_PATH = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
XSCHED_PATH = $(abspath $(WORK_PATH)/../..)
XSCHED_INCLUDE_PATH = $(XSCHED_PATH)/output/include
XSCHED_LIB_PATH = $(XSCHED_PATH)/output/lib

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++11 -I$(XSCHED_INCLUDE_PATH)
# link XSched libraries and set rpath so that the app can find the libraries
NVCC_LDFLAGS = -Xlinker -rpath -Xlinker $(XSCHED_LIB_PATH) -L$(XSCHED_LIB_PATH) -lpreempt -lhalcuda

HIPCC = hipcc
HIPCC_FLAGS = -O3 -std=c++11 -I$(XSCHED_INCLUDE_PATH)
# link XSched libraries and set rpath so that the app can find the libraries
HIPCC_LDFLAGS = -Xlinker -rpath -Xlinker $(XSCHED_LIB_PATH) -L/opt/rocm/lib -lamdhip64 -L$(XSCHED_LIB_PATH) -lpreempt -lhalhip 

# Target executable
TARGET = app_with_hints

# Source files
SRC_CUDA = app_with_hints.cu
SRC_HIP = app_with_hints.hip


cuda: $(SRC_CUDA)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $(TARGET) $<

hip: $(SRC_HIP)
	$(HIPCC) $(HIPCC_FLAGS) $(HIPCC_LDFLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

.PHONY: cuda hip clean
