#include <atomic>
#include <cstdlib>
#include <unistd.h>

#include "xsched/utils/log.h"
#include "xsched/utils/pci.h"
#include "xsched/utils/xassert.h"
#include "xsched/protocol/device.h"
#include "xsched/corex/hal.h"
#include "xsched/corex/hal/common/driver.h"
#include "xsched/corex/hal/common/corex_assert.h"
#include "xsched/corex/hal/level1/corex_queue.h"

using namespace xsched::corex;
using namespace xsched::utils;
using namespace xsched::preempt;
using namespace xsched::protocol;

CorexQueueLv1::CorexQueueLv1(cudaStream_t stream): kStream(stream)
{
    // get cuda context
    CUcontext stream_context = nullptr;
    CUcontext current_context = nullptr;
    COREX_ASSERT(Driver::CtxGetCurrent(&current_context));
    COREX_ASSERT(Driver::StreamGetCtx(stream, &stream_context));
    XASSERT(current_context == stream_context,
        "create CorexQueueLv1 failed: current context (%p) does not match stream context (%p)",
        current_context, stream_context);
    context_ = stream_context;

    cudaDeviceProp prop{};
    COREXRT_ASSERT(RtDriver::GetDevice(&cudevice_));
    COREXRT_ASSERT(RtDriver::GetDeviceProperties(&prop, cudevice_));
    xdevice_ = MakeDevice(kDeviceTypeGPU, XDeviceId(MakePciId(prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0)));

    // get stream flags
    COREXRT_ASSERT(RtDriver::StreamGetFlags(kStream, &stream_flags_));

    // make sure no commands are running on stream_
    COREXRT_ASSERT(RtDriver::StreamSynchronize(kStream));
}

void CorexQueueLv1::Launch(std::shared_ptr<HwCommand> hw_cmd)
{
    auto corex_cmd = std::dynamic_pointer_cast<CorexCommand>(hw_cmd);
    XASSERT(corex_cmd != nullptr, "hw_cmd is not a CorexCommand");
    COREXRT_ASSERT(corex_cmd->LaunchWrapper(kStream));
}

void CorexQueueLv1::Synchronize()
{
    COREXRT_ASSERT(RtDriver::StreamSynchronize(kStream));
}

void CorexQueueLv1::OnXQueueCreate()
{
    COREXRT_ASSERT(RtDriver::SetDevice(cudevice_));
    COREX_ASSERT(Driver::CtxSetCurrent(context_));
}

EXPORT_C_FUNC XResult CorexQueueGet(HwQueueHandle *hwq, cudaStream_t stream)
{
    if (hwq == nullptr) {
        XWARN("CorexQueueGet failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (stream == nullptr) {
        XWARN("CorexQueueGet failed: does not support default stream");
        return kXSchedErrorNotSupported;
    }

    HwQueueHandle hwq_h = GetHwQueueHandle(stream);
    auto hwq_shptr = HwQueueManager::Get(hwq_h);
    if (hwq_shptr == nullptr) return kXSchedErrorNotFound;
    *hwq = hwq_h;
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult CorexQueueCreate(HwQueueHandle *hwq, cudaStream_t stream)
{
    if (hwq == nullptr) {
        XWARN("CorexQueueCreate failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (stream == nullptr) {
        XWARN("CorexQueueCreate failed: does not support default stream");
        return kXSchedErrorNotSupported;
    }

    HwQueueHandle hwq_h = GetHwQueueHandle(stream);
    auto res = HwQueueManager::Add(hwq_h, [&]() {
        return std::make_shared<CorexQueueLv1>(stream);
    });

    if (res != kXSchedSuccess) return res;
    *hwq = hwq_h;

    // Work around for CoreX exit-time hang: as long as the worker thread of xsched
    // calls a cuda api, the program will hang at exit. Possibly due to
    // the corex-implemented cuda library didn't handle the thread-local data correctly.
    // The workaround is to register an atexit handler to skip the broken teardown.
    // Since std::atexit is LIFO, the handler registered on first stream-create will run first.
    static std::atomic_bool created = false;
    if (created.exchange(true)) return res;
    std::atexit([]() {
        std::fflush(nullptr);
        _exit(0);
    });
    return res;
}
