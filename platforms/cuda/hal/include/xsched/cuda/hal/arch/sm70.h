#pragma once

#include "xsched/cuda/hal/level2/guardian.h"
#include "xsched/cuda/hal/level3/trap.h"

namespace xsched::cuda
{

class GuardianSM70 : public Guardian
{
public:
    GuardianSM70() = default;
    virtual ~GuardianSM70() = default;

    virtual void GetGuardianInstructions(const void **guardian_instr, size_t *size) override;
    virtual void GetResumeInstructions(const void **resume_instr, size_t *size) override;
};

class TarpHandlerSM70 : public TarpHandler
{
public:
    TarpHandlerSM70() = default;
    virtual ~TarpHandlerSM70() = default;

    void SetJumpInstruction(char *instr, const uint64_t jmp_addr);
    virtual size_t GetInjectSize() override;
    virtual void Instrument(void *handler_host, CUdeviceptr handler_dev, size_t size,
                            void *inject_host , CUdeviceptr inject_dev) override;
};

} // namespace xsched::cuda
