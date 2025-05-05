GPU_GEN_CODE = $(subst .,,${shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader})

bin: dump.cu
	nvcc -cubin $< -o dump.cubin -arch=sm_${GPU_GEN_CODE}

dump: dump.cubin
	cuobjdump -sass $< > dump.asm

.PHONY: clean
clean:
	@rm dump.cubin
