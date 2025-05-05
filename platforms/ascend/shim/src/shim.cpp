#include "xsched/xqueue.h"
#include "xsched/utils/map.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/ascend/hal.h"
#include "xsched/ascend/shim/shim.h"
#include "xsched/ascend/hal/acl_queue.h"
#include "xsched/ascend/hal/acl_command.h"

using namespace xsched::preempt;

namespace xsched::ascend
{

static utils::ObjectMap<aclrtEvent, std::shared_ptr<AclEventRecordCommand>> g_events;

aclError XopCompileAndExecute(const char * opType, int numInputs, aclopCompileAndExecute_arg2_t inputDesc, aclopCompileAndExecute_arg3_t inputs, int numOutputs, aclopCompileAndExecute_arg5_t outputDesc, aclopCompileAndExecute_arg6_t outputs, const aclopAttr * attr, aclopEngineType engineType, aclopCompileType compileFlag, const char * opPath , aclrtStream stream)
{
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return OpCompiler::opCompileAndExecute(opType, numInputs, inputDesc, inputs, numOutputs, outputDesc, outputs, attr, engineType, compileFlag, opPath, stream);
    
    auto input_descs = std::make_shared<std::vector<std::shared_ptr<TensorDesc>>>();
    auto input_buffers = std::make_shared<std::vector<std::shared_ptr<DataBuffer>>>();
    auto output_descs = std::make_shared<std::vector<std::shared_ptr<TensorDesc>>>();
    auto output_buffers = std::make_shared<std::vector<std::shared_ptr<DataBuffer>>>();
    input_descs->reserve(numInputs);
    input_buffers->reserve(numInputs);
    output_descs->reserve(numOutputs);
    output_buffers->reserve(numOutputs);
    for (int i = 0; i < numInputs; ++i) {
        input_descs->push_back(TensorDesc::Create(inputDesc[i]));
        input_buffers->push_back(DataBuffer::Create(inputs[i]));
    }
    for (int i = 0; i < numOutputs; ++i) {
        output_descs->push_back(TensorDesc::Create(outputDesc[i]));
        output_buffers->push_back(DataBuffer::Create(outputs[i]));
    }
    
    auto hw_cmd = std::make_shared<AclOpCompileAndExecuteCommand>(opType,
                                                                  input_descs, input_buffers,
                                                                  output_descs, output_buffers,
                                                                  OpAttr::Create(attr),
                                                                  engineType, compileFlag, opPath);
    xq->Submit(hw_cmd);
    return ACL_SUCCESS;
}

aclError XrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    XDEBG("XrtRecordEvent(event: %p, stream: %p)", event, stream);
    if (event == nullptr) return Driver::rtRecordEvent(event, stream);

    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::rtRecordEvent(event, stream);

    auto xevent = std::make_shared<AclEventRecordCommand>(event);
    xq->Submit(xevent);
    g_events.Add(event, xevent);
    return ACL_SUCCESS;
}

aclError XrtSynchronizeEvent(aclrtEvent event)
{
    XDEBG("XrtSynchronizeEvent(event: %p)", event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::rtSynchronizeEvent(event);
    xevent->Wait();
    return ACL_SUCCESS;
}

aclError XrtSynchronizeDevice()
{
    XDEBG("XrtSynchronizeDevice()");
    auto res = XQueueManager::ForEachWaitAll();
    XASSERT(res == kXSchedSuccess, "failed to WaitAll() on all XQueues, err: %d", res);
    return Driver::rtSynchronizeDevice();
}

aclError XrtSynchronizeDeviceWithTimeout(int32_t timeout)
{
    XDEBG("XrtSynchronizeDeviceWithTimeout(timeout: %d)", timeout);
    auto res = XQueueManager::ForEachWaitAll();
    XASSERT(res == kXSchedSuccess, "failed to WaitAll() on all XQueues, err: %d", res);
    return Driver::rtSynchronizeDeviceWithTimeout(timeout);
}

aclError XrtSynchronizeStream(aclrtStream stream)
{
    XDEBG("XrtSynchronizeStream(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::rtSynchronizeStream(stream);
    xq->WaitAll();
    return ACL_SUCCESS;
}

aclError XrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)
{
    XDEBG("XrtSynchronizeStreamWithTimeout(stream: %p, timeout: %d)", stream, timeout);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::rtSynchronizeStreamWithTimeout(stream, timeout);
    xq->WaitAll();
    return ACL_SUCCESS;
}

aclError XrtCreateStream(aclrtStream *stream)
{
    aclError res = Driver::rtCreateStream(stream);
    if (res != ACL_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return AclQueueCreate(hwq, *stream); });
    XDEBG("XrtCreateStream(stream: %p)", *stream);
    return res;
}

aclError XrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag)
{
    aclError res = Driver::rtCreateStreamWithConfig(stream, priority, flag);
    if (res != ACL_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return AclQueueCreate(hwq, *stream); });
    XDEBG("XrtCreateStreamWithConfig(stream: %p, priority: %u, flag: 0x%x)", *stream, priority, flag);
    return res;
}

void XopDestroyAttr(const aclopAttr *attr)
{
    if (OpAttr::Destroy(attr)) return;
    Driver::opDestroyAttr(attr);
}

void XDestroyTensorDesc(const aclTensorDesc *desc)
{
    if (TensorDesc::Destroy(desc)) return;
    Driver::DestroyTensorDesc(desc);
}

aclError XDestroyDataBuffer(const aclDataBuffer *dataBuffer)
{
    if (DataBuffer::Destroy(dataBuffer)) return ACL_SUCCESS;
    return Driver::DestroyDataBuffer(dataBuffer);
}

} // namespace xsched::ascend
