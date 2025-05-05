#pragma once

#include "xsched/vpi/hal/driver.h"
#include "xsched/vpi/hal/vpi_assert.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::vpi
{

#define VPI_COMMAND(name, base, func, ...) \
    class name : public base \
    { \
    public: \
        name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
            : base(), FOR_EACH_PAIR_COMMA(DECLARE_COPY_PRIVATE_ARG, __VA_ARGS__) {} \
        virtual ~name() = default; \
    private: \
        FOR_EACH_PAIR_SEMICOLON(DECLARE_PRIVATE_PARAM, __VA_ARGS__) \
        virtual VPIStatus Launch(VPIStream stream) override \
        { return func(stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_PRIVATE_ARG, __VA_ARGS__)); } \
    };

class VpiCommand : public preempt::HwCommand
{
public:
    VpiCommand(preempt::XCommandProperties props): HwCommand(props) {}
    virtual ~VpiCommand();
    virtual void Synchronize() override;
    virtual bool Synchronizable() override;
    virtual bool EnableSynchronization() override;
    VPIStatus LaunchWrapper(VPIStream stream);

private:
    virtual VPIStatus Launch(VPIStream stream) = 0;
    VPIEvent following_event_ = nullptr;
};

class VpiEventRecordCommand : public VpiCommand
{
public:
    VpiEventRecordCommand(VPIEvent event);
    virtual ~VpiEventRecordCommand() = default;
    virtual void Synchronize() override { VPI_ASSERT(Driver::EventSync(event_)); }
    virtual bool Synchronizable() override { return true; }
    virtual bool EnableSynchronization() override { return true; }

private:
    VPIEvent event_;
    virtual VPIStatus Launch(VPIStream stream) override
    { return Driver::EventRecord(event_, stream); }
};

class VpiAlgorithmCommand : public VpiCommand
{
public:
    VpiAlgorithmCommand(): VpiCommand(preempt::kCommandPropertyNone) {}
    virtual ~VpiAlgorithmCommand() = default;
};

VPI_COMMAND(VpiConvertImageFormatCommand, VpiAlgorithmCommand, Driver::SubmitConvertImageFormat, uint64_t, backend, VPIImage, input, VPIImage, output, const VPIConvertImageFormatParams *, params);
VPI_COMMAND(VpiGaussianFilterCommand, VpiAlgorithmCommand, Driver::SubmitGaussianFilter, uint64_t, backend, VPIImage, input, VPIImage, output, int32_t, kernelSizeX, int32_t, kernelSizeY, float, sigmaX, float, sigmaY, VPIBorderExtension, border);
VPI_COMMAND(VpiRescaleCommand, VpiAlgorithmCommand, Driver::SubmitRescale, uint64_t, backend, VPIImage, input, VPIImage, output, VPIInterpolationType, interpolationType, VPIBorderExtension, border, uint64_t, flags);
VPI_COMMAND(VpiStereoDisparityEstimatorCommand, VpiAlgorithmCommand, Driver::SubmitStereoDisparityEstimator, uint64_t, backend, VPIPayload, payload, VPIImage, left, VPIImage, right, VPIImage, disparity, VPIImage, confidenceMap, const VPIStereoDisparityEstimatorParams *, params);

} // namespace xsched::vpi
