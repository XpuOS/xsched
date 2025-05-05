#pragma once

#include <memory>

#include "xsched/utils/function.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/vpi/hal/handle.h"
#include "xsched/vpi/hal/driver.h"
#include "xsched/vpi/hal/vpi_command.h"

namespace xsched::vpi
{

#define VPI_SHIM_FUNC(name, cmd, ...) \
inline VPIStatus X##name(VPIStream stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
{ \
    auto xq = xsched::preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(stream)); \
    if (xq == nullptr) return Driver::name(stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
    auto hw_cmd = std::make_shared<cmd>(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
    xq->Submit(hw_cmd); \
    return VPI_SUCCESS; \
}

VPI_SHIM_FUNC(SubmitConvertImageFormat, VpiConvertImageFormatCommand, uint64_t, backend, VPIImage, input, VPIImage, output, const VPIConvertImageFormatParams *, params);
VPI_SHIM_FUNC(SubmitGaussianFilter, VpiGaussianFilterCommand, uint64_t, backend, VPIImage, input, VPIImage, output, int32_t, kernelSizeX, int32_t, kernelSizeY, float, sigmaX, float, sigmaY, VPIBorderExtension, border);
VPI_SHIM_FUNC(SubmitRescale, VpiRescaleCommand, uint64_t, backend, VPIImage, input, VPIImage, output, VPIInterpolationType, interpolationType, VPIBorderExtension, border, uint64_t, flags);
VPI_SHIM_FUNC(SubmitStereoDisparityEstimator, VpiStereoDisparityEstimatorCommand, uint64_t, backend, VPIPayload, payload, VPIImage, left, VPIImage, right, VPIImage, disparity, VPIImage, confidenceMap, const VPIStereoDisparityEstimatorParams *, params);

VPIStatus XEventRecord(VPIEvent event, VPIStream stream);
VPIStatus XEventSync(VPIEvent event);
void XEventDestroy(VPIEvent event);

VPIStatus XStreamSync(VPIStream stream);
VPIStatus XStreamCreate(uint64_t flags, VPIStream *stream);

} // namespace xsched::vpi
