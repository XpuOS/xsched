# XSched inject toolchain Makefile (multi-arch generation).
#   make ARCH=120 bin dump cc     # explicit arch (86 / 120 / ...)
#   make bin dump cc              # auto-detect from the local GPU
#   Artifacts: inject_$(ARCH).cubin / .asm / _cc.asm
#   Requires CUDA >= 12.8 for sm_120 (>= 12.9 recommended).

ARCH ?= $(strip $(subst .,,$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader)))

ifeq ($(ARCH),)
$(error ARCH not specified and GPU auto-detect failed; use: make ARCH=120)
endif

CUBIN  = inject_$(ARCH).cubin
ASM    = inject_$(ARCH).asm
CCASM  = inject_$(ARCH)_cc.asm

NVCC      ?= nvcc
CUOBJDUMP ?= cuobjdump
NVDISASM  ?= nvdisasm

.PHONY: all bin dump cc clean

all: bin dump cc

bin: $(CUBIN)

$(CUBIN): inject.cu
	$(NVCC) -cubin $< -o $@ --keep-device-functions -Xptxas -astoolspatch -arch=sm_$(ARCH)

dump: $(ASM)

$(ASM): $(CUBIN)
	$(CUOBJDUMP) -sass $(CUBIN) > $@

cc: $(CCASM)

$(CCASM): $(CUBIN)
	$(NVDISASM) -c $(CUBIN) > $@

ifeq ($(OS), Windows_NT)
clean:
	-cmd /c del /q $(CUBIN) $(ASM) $(CCASM)
else
clean:
	-rm -f $(CUBIN) $(ASM) $(CCASM)
endif
