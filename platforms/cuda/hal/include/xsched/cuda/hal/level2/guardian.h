#pragma once

#include <memory>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

class Guardian
{
public:
    Guardian() = default;
    virtual ~Guardian() = default;
    static std::shared_ptr<Guardian> Instance(CUdevice dev);

    virtual void GetGuardianInstructions(const void **guardian_instr, size_t *size) = 0;
    virtual void GetResumeInstructions(const void **resume_instr, size_t *size) = 0;
};

} // namespace xsched::cuda
