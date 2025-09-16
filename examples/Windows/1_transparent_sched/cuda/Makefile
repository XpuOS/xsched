CUDA_LIB_PATH = "$(CUDA_PATH)/lib/x64"

# Compiler
NVCC = nvcc
# NVCC Linker
NVCC_LDFLAGS = -L$(CUDA_LIB_PATH) -lcudart

# Target executable
TARGET = app.exe

# Source files
SRC = app.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC) $(NVCC_LDFLAGS)

clean:
	del *.exe

.PHONY: all clean
