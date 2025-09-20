GPU_GEN_CODE = $(subst .,,${shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader})

bin: inject.cu
	nvcc -cubin $< -o inject.cubin --keep-device-functions -Xptxas -astoolspatch -arch=sm_${GPU_GEN_CODE}

dump: inject.cubin
	cuobjdump -sass $< > inject.asm

.PHONY: clean
clean:
	rm inject.cubin
