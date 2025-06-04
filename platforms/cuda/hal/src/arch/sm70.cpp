#include <cstring>

#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/arch/sm70.h"

using namespace xsched::cuda;

#define INSTR_LEN 16

void GuardianSM70::GetGuardianInstructions(const void **guardian_instr, size_t *size)
{
    static const uint64_t guardian_instructions[] = 
    {
        0x000000fffffff389,
        0x000fe200000e00ff,
        0x0000000000ff7355,
        0x000fe20000100000,
        0x0000034000007945,
        0x000fe40003800000,
        0x00062000ff047b82,
        0x000fc00000000800,
        0x00062100ff057b82,
        0x000fc00000000800,
        0x0000000000007919,
        0x000e220000002500,
        0x0000000001ff7355,
        0x000fe20000100000,
        0x0000027000017945,
        0x000fe40003800000,
        0x0000000000037919,
        0x000e280000002600,
        0x0000000000087919,
        0x000e680000002200,
        0x00000000000b7919,
        0x000e680000002100,
        0x00000000000a7919,
        0x000e680000002300,
        0x0000000000097919,
        0x000ea20000002700,
        0x0000040000007a24,
        0x001fe200078e0203,
        0x0000000b0aff7212,
        0x002fc6000780fe08,
        0x0000050000007a24,
        0x004fca00078e0209,
        0x0000000400097811,
        0x000fca00078e08ff,
        0x0000000409087825,
        0x000fe200078e0004,
        0x000001b000000947,
        0x000fea0003800000,
        0x0000000004007980,
        0x000ea2000010e900,
        0x0000000002ff7355,
        0x000fe20000100000,
        0x0000017000027945,
        0x000fe20003800000,
        0x000000ff0000720c,
        0x004fd80003f05270,
        0x0000013000008947,
        0x000fea0003800000,
        0x00062400ff067b82,
        0x000fc00000000800,
        0x00062500ff077b82,
        0x000fc00000000800,
        0x00000001ff037424,
        0x000fd000078e00ff,
        0x0000000008007385,
        0x0001e8000010e903,
        0x00000008040a7980,
        0x000ea2000010eb00,
        0x0000000003ff7355,
        0x000fe20000100000,
        0x000000a000037945,
        0x000fe20003800000,
        0x000000ff0a00720c,
        0x004fc80003f05070,
        0x000000ff0b00720c,
        0x000fd80003f05300,
        0x0000005000008947,
        0x000fea0003800000,
        0x000000060a00720c,
        0x001fc80003f05070,
        0x000000070b00720c,
        0x000fd80003f05300,
        0x0000000000030942,
        0x000fe20003800000,
        0x0000002000008947,
        0x000fea0003800000,
        0x0000005000007947,
        0x000fea0003800000,
        0x0000000804007385,
        0x0011e4000010eb06,
        0x0000000000037941,
        0x000fea0003800000,
        0x0000000408007385,
        0x0003e2000010e903,
        0x0000001000007947,
        0x000fea0003800000,
        0x0000000008007385,
        0x0001e4000010e9ff,
        0x0000000000027941,
        0x000fea0003800000,
        0x0000000000007992,
        0x000fea0000000000,
        0x0000000000017941,
        0x000fea0003800000,
        0xffffffff00007948,
        0x000fe80003800000,
        0x0000000000007918,
        0x000fe20000000000,
        0x0000000000007b1d,
        0x000fea0000000000,
        0x0000000008087980,
        0x003ea4000010e900,
        0x000000ff0800720c,
        0x004fd80003f05270,
        0x0000001000008947,
        0x000fea0003800000,
        0x000000000000794d,
        0x000fea0003800000,
        0x0000000000007941,
        0x000fea0003800000,
    };
    *guardian_instr = guardian_instructions;
    *size = sizeof(guardian_instructions);
}

void GuardianSM70::GetResumeInstructions(const void **resume_instr, size_t *size)
{
    static const uint64_t resume_instructions[] =
    {
        0x000000fffffff389,
        0x000fe200000e00ff,
        0x0000000000ff7355,
        0x000fe20000100000,
        0x0000013000007945,
        0x000fe40003800000,
        0x00062000ff047b82,
        0x000fc00000000800,
        0x00062100ff057b82,
        0x000fc00000000800,
        0x0000000000007919,
        0x000e280000002500,
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
        0x000fd000078e0004,
        0x0000000004007980,
        0x000ea4000010e900,
        0x000000ff0000720c,
        0x004fd80003f05270,
        0x000000000000894d,
        0x000fea0003800000,
        0xffffffff00007948,
        0x000fe80003800000,
        0x0000000000007918,
        0x000fe20000000000,
        0x0000000000007b1d,
        0x000fea0000000000,
        0x0000000004007385,
        0x0001e2000010e9ff,
        0x00062200ff147b82,
        0x000fc00000000800,
        0x00062300ff157b82,
        0x000fc00000000800,
        0x0000000000007941,
        0x000fea0003800000,
        0x0000000014007950,
        0x001fea0003e00000,
        0xfffffff000007947,
        0x000fc0000383ffff,
    };
    *resume_instr = resume_instructions;
    *size = sizeof(resume_instructions);
}

static const uint64_t trap_inject_instrs[] =
{
    0x000000fffffff389,
    0x000fe200000e00ff,
    0x0000000000ff7355,
    0x000fe20000100000,
    0x0000024000007945,
    0x000fe40003800000,
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
    0x0000000000007992,
    0x000fea0000002000,
    0x00000000000079ab,
    0x000fc00000000000,
    0x00000000ff00798f,
    0x000fca0002000000,
    0x0000016000008947,
    0x000fea0003800000,
    0x0000000004087980,
    0x000ee4000010e900,
    0x000000ff0800720c,
    0x008fd80003f05270,
    0x0000013000008947,
    0x000fea0003800000,
    0x0000000804087980,
    0x000ee2000010eb00,
    0x0000040000007a24,
    0x003fe200078e0203,
    0x0000000000ff7355,
    0x000fe20000100000,
    0x000000c000007945,
    0x000fe40003800000,
    0x0000050000007a24,
    0x004fca00078e020b,
    0x00000004000b7811,
    0x000fca00078e08ff,
    0x000000040b0a7825,
    0x000fe200078e0004,
    0x000000ff0800720c,
    0x008fc80003f05070,
    0x000000ff0900720c,
    0x000fd80003f05300,
    0x0000004000008947,
    0x000fea0003800000,
    0x000000060800720c,
    0x000fc80003f05070,
    0x000000070900720c,
    0x000fd80003f05300,
    0x0000002000008947,
    0x000fea0003800000,
    0x000000000000794d,
    0x000fea0003800000,
    0x0000000804007385,
    0x0001e4000010eb06,
    0x0000000000007941,
    0x000fea0003800000,
    0x00000001ff037424,
    0x000fd000078e00ff,
    0x000000040a007385,
    0x000fe2000010e903,
    0x000000000000794d,
    0x000fea0003800000,
    0x0000000000007941,
    0x000fea0003800000,
};

void TarpHandlerSM70::SetJumpInstruction(char *instr, const uint64_t jmp_addr)
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

size_t TarpHandlerSM70::GetInjectSize()
{
    return sizeof(trap_inject_instrs) + 2 * INSTR_LEN;
}

void TarpHandlerSM70::Instrument(void *handler_host, CUdeviceptr handler_dev, size_t size,
                                 void *inject_host , CUdeviceptr inject_dev)
{
    XASSERT(size % INSTR_LEN == 0,
            "trap handler size must be an integer multiple of INSTR_LEN (%d)", INSTR_LEN);
    uint64_t *trap_handler_instrs = (uint64_t *)handler_host;

    // replace the instruction with a jump instruction to the inject_dev
    uint64_t replaced_instr[2] = {
        // should be IADD3 R1, R1, -0x10, RZ ;
        trap_handler_instrs[0x260 / 8 + 0],
        trap_handler_instrs[0x260 / 8 + 1],
    };

    // jump back to the next instruction of the replaced instruction
    char jmp_back_instr[INSTR_LEN];
    SetJumpInstruction(jmp_back_instr, handler_dev + 0x260 + INSTR_LEN);

    memcpy(inject_host, trap_inject_instrs, sizeof(trap_inject_instrs));
    // followed by the replaced instruction
    memcpy((char *)inject_host + sizeof(trap_inject_instrs), replaced_instr, INSTR_LEN);
    // followed by the jump back instruction
    memcpy((char *)inject_host + sizeof(trap_inject_instrs) + INSTR_LEN, jmp_back_instr, INSTR_LEN);

    // set the jump instruction to the inject_dev
    SetJumpInstruction((char *)handler_host + 0x260, inject_dev);
}
