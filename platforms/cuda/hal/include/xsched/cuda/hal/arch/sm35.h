#pragma once

#include "xsched/cuda/hal/level2/guardian.h"
#include "xsched/cuda/hal/level3/trap.h"

namespace xsched::cuda
{

class GuardianSM35 : public Guardian
{
public:
    GuardianSM35() = default;
    virtual ~GuardianSM35() = default;

    virtual void GetGuardianInstructions(const void **guardian_instr, size_t *size) override;
    virtual void GetResumeInstructions(const void **resume_instr, size_t *size) override;
};

} // namespace xsched::cuda
