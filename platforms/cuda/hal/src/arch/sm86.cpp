#include <cstring>

#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/level2/instrument.h"
#include "xsched/cuda/hal/level3/trap.h"

SET_CUDA_QUEUE_LEVEL(3);
using namespace xsched::cuda;

#define INSTR_LEN 16

static const uint64_t check_preempt_instrs_trap[] =
{
    // 0x000000000000794d,
    // 0x000fea0003800000,
    // 0xfffffff000007947,
    // 0x000fc0000383ffff,
    
    // 0x0000027000007945,
    // 0x000fe40003800000,
    0x00062000ff047b82,
    0x000fc00000000800,
    0x00062100ff057b82,
    0x000fc00000000800,
    0x00062400ff067b82,
    0x000fc00000000800,
    0x00062500ff077b82,
    0x000fc00000000800,
    0x00062600ff087b82,
    0x000fc00000000800,
    0x0000000000007919,
    0x000e220000002500,
    0x000000ff0800720c,
    0x000fc60003f05270,
    0x0000000000037919,
    0x000e680000002600,
    0x00000000000b7919,
    0x000ea20000002700,
    0x00000000fffff984,
    0x000fe20000000800,
    0x00000000fffff984,
    0x000fe20000000800,
    0x00000000fffff984,
    0x000fe20000000800,
    0x00000000fffff984,
    0x000fe20000000800,
    0x0000000000007992,
    0x000fec0000002000,
    0x00000000000079ab,
    0x000fc00000000000,
    0x00000000ff00798f,
    0x000fca0002000000,
    0x0000015000008947,
    0x000fea0003800000,
    0x0000000004087980,
    0x000ee40000100900,
    0x000000ff0800720c,
    0x008fda0003f05270,
    0x0000012000008947,
    0x000fea0003800000,
    0x0000000804087980,
    0x000ee20000100b00,
    0x0000040000007a24,
    0x003fe200078e0203,
    0x000000c000007945,
    0x000fe60003800000,
    0x0000050000007a24,
    0x004fca00078e020b,
    0x00000004000b7811,
    0x000fca00078e08ff,
    0x000000040b0a7825,
    0x000fe200078e0004,
    0x000000ff0800720c,
    0x008fc80003f05070,
    0x000000ff0900720c,
    0x000fda0003f05300,
    0x0000004000008947,
    0x000fea0003800000,
    0x000000060800720c,
    0x000fc80003f05070,
    0x000000070900720c,
    0x000fda0003f05300,
    0x0000002000008947,
    0x000fea0003800000,
    0x000000000000794d,
    0x000fea0003800000,
    0x0000000804007385,
    0x0001e40000100b06,
    0x0000000000007941,
    0x000fea0003800000,
    0x00000001ff037424,
    0x000fca00078e00ff,
    0x000000040a007385,
    0x000fe20000100903,
    0x000000000000794d,
    0x000fea0003800000,
    // 0x0000000000007941,
    // 0x000fea0003800000,
};

static void SetJmpAddr(char *instr, const uint64_t jmp_addr)
{
    // for example, jmp 0x7f1234567890
    *(uint64_t *)(instr + 0) = 0x345678900000794a;
    *(uint64_t *)(instr + 8) = 0x000fea0003807f12;
    instr[4] = jmp_addr & 0xff;
    instr[5] = (jmp_addr >> 8) & 0xff;
    instr[6] = (jmp_addr >> 16) & 0xff;
    instr[7] = (jmp_addr >> 24) & 0xff;
    instr[8] = (jmp_addr >> 32) & 0xff;
    instr[9] = (jmp_addr >> 40) & 0xff;
}

void TrapManager::InstrumentTrapHandler(void *trap_handler_host,
                                        CUdeviceptr trap_handler_dev,
                                        size_t trap_handler_size,
                                        void *extra_instrs_host,
                                        CUdeviceptr extra_instrs_device,
                                        size_t extra_instrs_size)
{
    XASSERT(trap_handler_size % INSTR_LEN == 0,
            "trap_handler_size must be an integer multiple of INSTR_LEN");
    XASSERT(extra_instrs_size >= 
            sizeof(check_preempt_instrs_trap) + 2 * INSTR_LEN,
            "extra_instrs_device too small (%ld)", extra_instrs_size);
    uint64_t *trap_handler_instrs_host = (uint64_t *)trap_handler_host;

    // instrument trap handler
    uint64_t coveredInstr[2] = {
        // should be IADD3 R1, R1, -0x10, RZ ;
        trap_handler_instrs_host[0x3e0 / 8 + 0],
        trap_handler_instrs_host[0x3e0 / 8 + 1],
    };

    char jmp_back_instr[INSTR_LEN];
    SetJmpAddr(jmp_back_instr, trap_handler_dev + 0x3e0 + INSTR_LEN);

    memcpy(extra_instrs_host, check_preempt_instrs_trap,
           sizeof(check_preempt_instrs_trap));
    memcpy((char *)extra_instrs_host + sizeof(check_preempt_instrs_trap),
           coveredInstr, INSTR_LEN);
    memcpy((char *)extra_instrs_host + sizeof(check_preempt_instrs_trap)
           + INSTR_LEN,
           jmp_back_instr, INSTR_LEN);

    SetJmpAddr((char *)trap_handler_host + 0x3e0, extra_instrs_device);
}

void InstrumentContext::GetResumeInstructions(const void **instructions,
                                              size_t *size)
{
    static const uint64_t resume_instructions[] =
    {
        0x0000013000007945,
        0x000fe40003800000,
        0x0000000000007919,
        0x000e220000002500,
        0x00062000ff047b82,
        0x000fc00000000800,
        0x00062100ff057b82,
        0x000fc00000000800,
        0x0000000000037919,
        0x000e280000002600,
        0x0000000000077919,
        0x000e620000002700,
        0x0000040000007a24,
        0x001fc800078e0203,
        0x0000050000007a24,
        0x002fca00078e0207,
        0x0000000500037811,
        0x000fca00078e08ff,
        0x0000000403047825,
        0x000fca00078e0004,
        0x0000000004007980,
        0x000ea40000100900,
        0x000000ff0000720c,
        0x004fda0003f05270,
        0x000000000000894d,
        0x000fea0003800000,
        0xffffffff00007948,
        0x000fe20003800000,
        0x0000000000007b1d,
        0x000fec0000000000,
        0x0000000004007385,
        0x0001e200001009ff,
        0x00062200ff147b82,
        0x000fc00000000800,
        0x00062300ff157b82,
        0x000fc00000000800,
        0x0000000000007947,
        0x000fea0003800000,
        0x0000000000007941,
        0x000fea0003800000,
        0x0000000014007950,
        0x001fea0003e00000,
        0xfffffff000007947,
        0x000fc0000383ffff,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
        0x0000000000007918,
        0x000fc00000000000,
    };
    *instructions = resume_instructions;
    *size = sizeof(resume_instructions);
}

void InstrumentContext::GetCheckInstructions(const void **instructions,
                                             size_t *size)
{
    static const uint64_t check_preempt_instrunctions[] = 
    {
        0x0000034000007945,
        0x000fe40003800000,
        0x0000000000087919,
        0x000e220000002200,
        0x00062000ff047b82,
        0x000fc00000000800,
        0x00062100ff057b82,
        0x000fc00000000800,
        0x0000029000017945,
        0x000fe20003800000,
        0x00000000000b7919,
        0x000e280000002100,
        0x00000000000a7919,
        0x000e280000002300,
        0x0000000000007919,
        0x000e680000002500,
        0x0000000000037919,
        0x000e680000002600,
        0x0000000000097919,
        0x000ea20000002700,
        0x0000000b0aff7212,
        0x001fe2000780fe08,
        0x0000040000007a24,
        0x002fc800078e0203,
        0x0000050000007a24,
        0x004fca00078e0209,
        0x0000000400097811,
        0x000fca00078e08ff,
        0x0000000409087825,
        0x000fe200078e0004,
        0x000001d000000947,
        0x000fea0003800000,
        0x0000000004007980,
        0x000ea20000100900,
        0x0000016000027945,
        0x000fe20003800000,
        0x000000ff0000720c,
        0x004fda0003f05270,
        0x0000012000008947,
        0x000fea0003800000,
        0x00000001ff037424,
        0x000fe200078e00ff,
        0x00062400ff067b82,
        0x000fc00000000800,
        0x00062500ff077b82,
        0x000fc00000000800,
        0x0000000008007385,
        0x0001e80000100903,
        0x00000008040a7980,
        0x000ea20000100b00,
        0x000000a000037945,
        0x000fe20003800000,
        0x000000ff0a00720c,
        0x004fc80003f05070,
        0x000000ff0b00720c,
        0x000fda0003f05300,
        0x0000005000008947,
        0x000fea0003800000,
        0x000000060a00720c,
        0x001fc80003f05070,
        0x000000070b00720c,
        0x000fda0003f05300,
        0x0000000000030942,
        0x000fe20003800000,
        0x0000002000008947,
        0x000fea0003800000,
        0x0000005000007947,
        0x000fea0003800000,
        0x0000000804007385,
        0x0011e40000100b06,
        0x0000000000037941,
        0x000fea0003800000,
        0x0000000408007385,
        0x0003e20000100903,
        0x0000001000007947,
        0x000fea0003800000,
        0x0000000008007385,
        0x0001e400001009ff,
        0x0000000000027941,
        0x000fea0003800000,
        0x00000000fffff984,
        0x000fe20000000800,
        0x00000000fffff984,
        0x000fe20000000800,
        0x00000000fffff984,
        0x000fe20000000800,
        0x00000000fffff984,
        0x000fe20000000800,
        0x0000000000007992,
        0x000fec0000000000,
        0x0000000000017941,
        0x000fea0003800000,
        0xffffffff00007948,
        0x000fe20003800000,
        0x0000000000007b1d,
        0x000fec0000000000,
        0x0000000008087980,
        0x003ea40000100900,
        0x000000ff0800720c,
        0x004fda0003f05270,
        0x0000001000008947,
        0x000fea0003800000,
        0x000000000000794d,
        0x000fea0003800000,
        0x0000000000007941,
        0x000fea0003800000,
    };
    *instructions = check_preempt_instrunctions;
    *size = sizeof(check_preempt_instrunctions);
}
