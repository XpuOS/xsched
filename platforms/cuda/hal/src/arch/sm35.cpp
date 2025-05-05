#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/level2/cuda_queue.h"
#include "xsched/cuda/hal/level2/instrument.h"
#include "xsched/cuda/hal/level3/trap.h"

SET_CUDA_QUEUE_LEVEL(2);
using namespace xsched::cuda;

void TrapManager::InstrumentTrapHandler(void *,
                                        CUdeviceptr ,
                                        size_t ,
                                        void *,
                                        CUdeviceptr ,
                                        size_t)
{
    XERRO("Level-3 not implemented for sm35");
}

void InstrumentContext::GetResumeInstructions(const void **instructions,
                                              size_t *size)
{
    static const uint64_t resume_instructions[] =
    {
        0x08a0a0908c8cbcbc,
        0x7ca0000c401ffc12,
        0x7ca0000c421ffc16,
        0x86400000129c0002,
        0x86400000131c000e,
        0x86400000139c001a,
        0x51080c00071c0002,
        0x51081800079c0002,
        0x08b8bcb0fca0b0a0,
        0xc0c00400029c0001,
        0xa0041000021c0011,
        0xa2101400021c0015,
        0xc4800000001c1000,
        0xdb581c007f9c001e,
        0x180000000020003c,
        0x8540dc00001c0002,
        0x08000000bcbcbc10,
        0xe4800000001c13fc,
        0x7ca0000c441ffc02,
        0x7ca0000c461ffc06,
        0x10000000001c003c,
        0x12007ffffc1c003c,
        0x85800000001c3c02,
        0x85800000001c3c02,
    };
    *instructions = resume_instructions;
    *size = sizeof(resume_instructions);
}

void InstrumentContext::GetCheckInstructions(const void **instructions,
                                             size_t *size)
{
    static const uint64_t check_preempt_instrunctions[] = 
    {
        0x77000000001c0002,
        0x7ca0000c401ffc12,
        0x7ca0000c421ffc16,
        0x7ca0000c481ffc1a,
        0x7ca0000c4a1ffc1e,
        0x08a088808c8c8c10,
        0x86400000129c0002,
        0x14800000a4000000,
        0x86400000131c000e,
        0x86400000139c0026,
        0x86400000109c0022,
        0x51080c00071c0002,
        0x86400000111c000e,
        0x08b0a010a09c8010,
        0x86400000119c002a,
        0xe2001000019c200e,
        0x51082400079c0002,
        0xe2001000051c0c0e,
        0xc0c00400021c0001,
        0xdb581c007f9c0c1e,
        0xa0041000021c0021,
        0x08a0bcb0fc10bc10,
        0xa2101400021c0025,
        0x8580000000403c02,
        0xcc800000001c1000,
        0x1500000054000000,
        0xdb581c007f9c001e,
        0x120000004020003c,
        0x74000000009fc00e,
        0x08b0bcb0b0fcb810,
        0xe4800000001c200c,
        0x1480000028000000,
        0xcd800000041c1028,
        0xe0940000051ffffe,
        0xdb585c007f9c2c1e,
        0x120000001420003c,
        0xe0940000051c1bfe,
        0x0880b810bcb8bcb0,
        0xdb585c00039c2c1e,
        0x8580000000603c02,
        0x1a000000001c003c,
        0xe5800000045c1018,
        0xe4800000021c200c,
        0x1a000000001c003c,
        0xe4800000001c23fc,
        0x08bcb0fc00bcbcbc,
        0x1a000000001c003c,
        0x85800000001c3c02,
        0x85800000005c3c02,
        0x8540dc00001c0002,
        0xcc800000001c2020,
        0xdb581c007f9c201e,
        0x120000000c20003c,
        0x0800000000bcbc10,
        0xe4c03c007f9c03fe,
        0x18000000001c003c,
    };
    *instructions = check_preempt_instrunctions;
    *size = sizeof(check_preempt_instrunctions);
}
