#include <cstdint>

#define INJECT_FUNC EXPORT_C_FUNC __device__ __noinline__
#define EXPORT_C_FUNC extern "C" __attribute__((visibility("default")))

inline __device__ void nop();
inline __device__ void exit();
inline __device__ uint32_t get_blockid();

INJECT_FUNC void nop_func()
{
    nop();
}

INJECT_FUNC void dummy(uint64_t val)
{
    printf("dummy, val: %lx\n", val);
}

INJECT_FUNC void jmp_to_target(uint64_t target)
{
    if (target == 0) { /* jmp(0x7f1200000000); */ }
    else             { /* jmp(target); */ }
}

// the sass of jmp_to_target() can be modifed from this function.
INJECT_FUNC void jmp_to_target_helper(uint64_t target)
{
    if (target != 0) exit();
}

INJECT_FUNC void relocate_func_call(uint64_t func_addr)
{
    const uint32_t ret_offset = 0x12345678;
    uint64_t ret_addr = func_addr + ret_offset;
    dummy(ret_addr);
}

/* preempt buffer layout: (see also tools/instrument/inject.cu)
 * |<------- uint32 ------->|<--- 32 bits -->|<---- uint64 ----->|<--------- uint32 --------->|<----------- uint32 ---------->|<------ ... ------>|
 * |<-- global_exit_flag -->|<-- reserved -->|<-- preempt_idx -->|<-- exit_flag_of_block_0 -->|<-- restore_flag_of_block_0 -->|<-- block_1 ... -->|
 */
INJECT_FUNC void check_preempt(uint32_t *preempt_buffer_addr,   // R4 @ c[0x0][0x1880], R5 @ c[0x0][0x1884]
                               uint64_t kernel_idx)             // R6 @ c[0x0][0x1890], R7 @ c[0x0][0x1894]
{
    nop(); // LDC R4, c[0x0][0x1880];
    nop(); // LDC R5, c[0x0][0x1884];

    uint64_t preempt_idx;
    uint32_t block_idx = get_blockid();
    uint32_t *global_exit_flag = preempt_buffer_addr;
    uint64_t *preempt_idx_ptr = (uint64_t *)&preempt_buffer_addr[2];
    uint32_t *block_exit_flag = &preempt_buffer_addr[2 * block_idx + 4];
    uint32_t *block_restore_flag = block_exit_flag + 1;

    if ((threadIdx.x | threadIdx.y | threadIdx.z) != 0) goto sync;

    if (!*global_exit_flag) {
        *block_exit_flag = 0;
        goto fence;
    }

    nop(); // LDC R6, c[0x0][0x1890];
    nop(); // LDC R7, c[0x0][0x1894];
    *block_exit_flag = 1;
    preempt_idx = *preempt_idx_ptr;
    if (preempt_idx == 0) goto save_preempt_idx;
    if (preempt_idx == kernel_idx) goto save_restore_flag;
    goto fence;

save_preempt_idx:
    *preempt_idx_ptr = kernel_idx;

save_restore_flag:
    *block_restore_flag = 1;

fence:
    __threadfence_block();

sync:
    __syncthreads();
    if (*block_exit_flag != 0) exit();
}

INJECT_FUNC void check_preempt_trap(uint32_t *preempt_buffer_addr,   // R4 @ c[0x0][0x1880], R5 @ c[0x0][0x1884]
                                    uint64_t kernel_idx,             // R6 @ c[0x0][0x1890], R7 @ c[0x0][0x1894]
                                    uint32_t idempotent)             // R8 @ c[0x0][0x1898]
{
    nop(); // LDC R4, c[0x0][0x1880];
    nop(); // LDC R5, c[0x0][0x1884];
    nop(); // LDC R6, c[0x0][0x1890];
    nop(); // LDC R7, c[0x0][0x1894];
    nop(); // LDC R8, c[0x0][0x1898];

    __threadfence();

    uint64_t preempt_idx;
    uint32_t block_idx = get_blockid();
    uint32_t *global_exit_flag = preempt_buffer_addr;
    uint64_t *preempt_idx_ptr = (uint64_t *)&preempt_buffer_addr[2];
    uint32_t *block_exit_flag = &preempt_buffer_addr[2 * block_idx + 4];
    uint32_t *block_restore_flag = block_exit_flag + 1;

    if (!idempotent || *global_exit_flag == 0) goto out;

    preempt_idx = *preempt_idx_ptr;
    if (preempt_idx == 0) goto save_preempt_idx;
    if (preempt_idx == kernel_idx) goto save_restore_flag;
    goto exit;

save_preempt_idx:
    *preempt_idx_ptr = kernel_idx;

save_restore_flag:
    *block_restore_flag = 1;

exit:
    exit();

out:
    return;
}

INJECT_FUNC void restore_exec(uint32_t *preempt_buffer_addr)    // R4 @ c[0x0][0x1880], R5 @ c[0x0][0x1884]
{
    nop(); // LDC R4, c[0x0][0x1880];
    nop(); // LDC R5, c[0x0][0x1884];

    uint32_t block_idx = get_blockid();
    uint32_t *block_restore_flag = &preempt_buffer_addr[2 * block_idx + 5];

    if (*block_restore_flag == 0) exit();

    __syncthreads();
    *block_restore_flag = 0;

    // return to func_entry_point @ (c[0x0][0x1888], c[0x0][0x188c])
    nop(); // LDC R20, c[0x0][0x1888];
    nop(); // LDC R21, c[0x0][0x188c];
}

INJECT_FUNC void exit_if_idempotent(uint32_t *preempt_buffer_addr,  // R4 @ c[0x0][0x1880], R5 @ c[0x0][0x1884]
                                    uint64_t kernel_idx)            // R6 @ c[0x0][0x1890], R7 @ c[0x0][0x1894]
{
    uint32_t block_idx;
    uint32_t idempotent; // @ R0
    uint64_t *preempt_idx_ptr;
    uint32_t *global_exit_flag;
    uint32_t *block_restore_flag;
    
    nop(); // STL [0xfffe00], R0;
    nop(); // IMAD.MOV.U32 R0, RZ, RZ, RZ ;
    nop(); // @P0 MOV R0, 0x1 ;
    nop(); // STL [0xfffe04], R0;
    idempotent = *preempt_buffer_addr; // LDC R0, c[0x0][0x1898];
    if (!idempotent) goto ret;

    nop(); // LDC R4, c[0x0][0x1880];
    nop(); // LDC R5, c[0x0][0x1884];
    nop(); // LDC R6, c[0x0][0x1890]; 
    nop(); // LDC R7, c[0x0][0x1894];
    block_idx = get_blockid();
    global_exit_flag = preempt_buffer_addr;
    preempt_idx_ptr = (uint64_t *)&preempt_buffer_addr[2];
    block_restore_flag = &preempt_buffer_addr[2 * block_idx + 5];

    *global_exit_flag = 1;
    *preempt_idx_ptr = kernel_idx;
    *block_restore_flag = 1;
    exit();

ret:
    nop(); // LDL R0, [0xfffe04];
    nop(); // ISETP.NE.AND P0, PT, R0, RZ, PT ;
    nop(); // LDL R0, [0xfffe00];
}

inline __device__ uint32_t get_blockid()
{
    return blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
}

inline __device__ void nop()
{
    asm volatile("brkpt;");
}

inline __device__ void exit()
{
    asm("exit;");
}
