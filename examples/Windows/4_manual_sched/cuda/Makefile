WORK_PATH = $(CURDIR)
XSCHED_PATH = $(WORK_PATH)/../../../..
XSCHED_INCLUDE_PATH = $(XSCHED_PATH)/output/include
XSCHED_LIB_PATH = $(XSCHED_PATH)/output/lib
XSCHED_BIN_PATH = $(XSCHED_PATH)/output/bin
CUDA_LIB_PATH = "$(CUDA_PATH)/lib/x64"

# Compiler
NVCC = nvcc
# NVCC flags
NVCC_FLAGS = -c -I$(XSCHED_INCLUDE_PATH) -o $@

DRIVER_LIB = nvcuda
NVCC_LDFLAGS = -L$(XSCHED_LIB_PATH) -l$(DRIVER_LIB) -L$(CUDA_LIB_PATH) -lcudart

# Target executable
TARGET = app_sched.exe

# Source files
SRC = app_sched.cu
OBJ = app_sched.o

all: $(TARGET)

$(TARGET): $(OBJ) $(LIB)
	$(NVCC) -o $(TARGET) $(OBJ) $(NVCC_LDFLAGS)
	del *.o

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $<

clean:
	del *.dll *.exe

.PHONY: all clean
