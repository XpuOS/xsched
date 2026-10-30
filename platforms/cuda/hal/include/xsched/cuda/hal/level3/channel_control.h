#pragma once

namespace xsched::cuda::channel
{

// Internal ABI between the HAL and the explicitly preloaded RM tracker.
inline constexpr char ControlSymbol[] = "xsched_cuda_channel_control_v2";
inline constexpr char ValidatedDriverVersion[] = "595.71.05";

enum Operation : int
{
    kAcquire = 0, // Input: compute channel. Output: its owning group.
    kPause = 1,
    kResume = 2,
    kRelease = 3,
};

// Nonzero means failure. Pause must never report success after rollback.
// All operations after acquire use the returned group and the original binding.
using Control = int (*)(int fd, unsigned client, unsigned subdevice,
                        unsigned *handle, int operation);

} // namespace xsched::cuda::channel
