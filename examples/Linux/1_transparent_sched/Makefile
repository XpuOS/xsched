# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++11

HIPCC = hipcc
HIPCC_FLAGS = -O3 -std=c++11

# Target executable
TARGET = app

# Source files
SRC_CUDA = app.cu
SRC_HIP = app.hip

cuda: $(SRC_CUDA)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $<

hip: $(SRC_HIP)
	$(HIPCC) $(HIPCC_FLAGS) -L/opt/rocm/lib -lamdhip64 -o $(TARGET) $<

clean:
	rm -f $(TARGET)

.PHONY: cuda hip clean
