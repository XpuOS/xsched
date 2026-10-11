#pragma once

#include "xsched/cuda/hal/level2/guardian.h"

namespace xsched::cuda
{

/// @brief Level-2 (guardian) support for Blackwell sm_120.
/// Instruction arrays are generated offline and hardware-reviewed; sm_120
/// has no trap handler (Level-3) support yet: TarpHandlerSM120 is stage three.
class GuardianSM120 : public Guardian
{
public:
    GuardianSM120() = default;
    virtual ~GuardianSM120() = default;

    virtual void GetGuardianInstructions(const void **guardian_instr, size_t *size) override;
    virtual void GetResumeInstructions(const void **resume_instr, size_t *size) override;

    // Hardware-verified register/barrier floors (guardian R11 / resume R21).
    virtual size_t RequiredRegs() override { return 32; }
    virtual size_t RequiredBarriers() override { return 1; }
};

} // namespace xsched::cuda
