# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++11

# Target executable
TARGET = app

# Source files
SRC = app.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
