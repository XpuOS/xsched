#include "xsched/utils/xassert.h"
#include "xsched/ascend/hal/event_pool.h"
#include "xsched/ascend/hal/acl_command.h"

using namespace xsched::ascend;

AclCommand::~AclCommand()
{
    if (following_event_ == nullptr) return;
    EventPool::Instance().Push(following_event_);
}

void AclCommand::Synchronize()
{
    XASSERT(following_event_ != nullptr,
            "following_event_ is nullptr, EnableSynchronization() should be called first");
    ACL_ASSERT(Driver::rtSynchronizeEvent(following_event_));
}

bool AclCommand::Synchronizable()
{
    return following_event_ != nullptr;
}

bool AclCommand::EnableSynchronization()
{
    following_event_ = (aclrtEvent)EventPool::Instance().Pop();
    return following_event_ != nullptr;
}

aclError AclCommand::LaunchWrapper(aclrtStream stream)
{
    aclError ret = Launch(stream);
    if (UNLIKELY(ret != ACL_SUCCESS)) return ret;
    if (following_event_ != nullptr) ret = Driver::rtRecordEvent(following_event_, stream);
    return ret;
}

AclEventRecordCommand::AclEventRecordCommand(aclrtEvent event): event_(event)
{
    XASSERT(event_ != nullptr, "aclrtEvent should not be nullptr");
    this->SetProps(preempt::kCommandPropertyIdempotent);
}

std::mutex TensorDesc::tensor_desc_mutex_;
std::mutex DataBuffer::data_buffer_mutex_;
std::mutex OpAttr::op_attr_mutex_;
std::unordered_map<const aclTensorDesc *, std::shared_ptr<TensorDesc>> TensorDesc::tensor_descs_;
std::unordered_map<const aclDataBuffer *, std::shared_ptr<DataBuffer>> DataBuffer::data_buffers_;
std::unordered_map<const aclopAttr *, std::shared_ptr<OpAttr>> OpAttr::op_attrs_;

std::shared_ptr<TensorDesc> TensorDesc::Create(const aclTensorDesc *desc)
{
    std::lock_guard<std::mutex> lock(tensor_desc_mutex_);
    auto it = tensor_descs_.find(desc);
    if (it != tensor_descs_.end()) return it->second;
    auto tensor_desc = std::make_shared<TensorDesc>();
    tensor_desc->desc_ = desc;
    tensor_descs_[desc] = tensor_desc;
    return tensor_desc;
}

bool TensorDesc::Destroy(const aclTensorDesc *desc)
{
    std::unique_lock<std::mutex> lock(tensor_desc_mutex_);
    auto it = tensor_descs_.find(desc);
    if (it == tensor_descs_.end()) return false;
    auto tensor_desc = it->second;
    tensor_descs_.erase(it);
    lock.unlock();
    tensor_desc = nullptr;
    return true;
}

std::shared_ptr<DataBuffer> DataBuffer::Create(const aclDataBuffer *buffer)
{
    std::lock_guard<std::mutex> lock(data_buffer_mutex_);
    auto it = data_buffers_.find(buffer);
    if (it != data_buffers_.end()) return it->second;
    auto data_buffer = std::make_shared<DataBuffer>();
    data_buffer->buffer_ = buffer;
    data_buffers_[buffer] = data_buffer;
    return data_buffer;
}

bool DataBuffer::Destroy(const aclDataBuffer *buffer)
{
    std::unique_lock<std::mutex> lock(data_buffer_mutex_);
    auto it = data_buffers_.find(buffer);
    if (it == data_buffers_.end()) return false;
    auto data_buffer = it->second;
    data_buffers_.erase(it);
    lock.unlock();
    data_buffer = nullptr;
    return true;
}

std::shared_ptr<OpAttr> OpAttr::Create(const aclopAttr *attr)
{
    std::lock_guard<std::mutex> lock(op_attr_mutex_);
    auto it = op_attrs_.find(attr);
    if (it != op_attrs_.end()) return it->second;
    auto op_attr = std::make_shared<OpAttr>();
    op_attr->attr_ = attr;
    op_attrs_[attr] = op_attr;
    return op_attr;
}

bool OpAttr::Destroy(const aclopAttr *attr)
{
    std::unique_lock<std::mutex> lock(op_attr_mutex_);
    auto it = op_attrs_.find(attr);
    if (it == op_attrs_.end()) return false;
    auto op_attr = it->second;
    op_attrs_.erase(it);
    lock.unlock();
    op_attr = nullptr;
    return true;
}

aclError AclOpCompileAndExecuteCommand::Launch(aclrtStream stream)
{
    int num_inputs = inputDesc_->size();
    int num_outputs = outputDesc_->size();
    const aclTensorDesc **input_desc  = (const aclTensorDesc**)malloc(num_inputs  * sizeof(aclTensorDesc*));
    const aclDataBuffer **inputs      = (const aclDataBuffer**)malloc(num_inputs  * sizeof(aclDataBuffer*));
    const aclTensorDesc **output_desc = (const aclTensorDesc**)malloc(num_outputs * sizeof(aclTensorDesc*));
          aclDataBuffer **outputs     = (      aclDataBuffer**)malloc(num_outputs * sizeof(aclDataBuffer*));
    for (int i = 0; i < num_inputs; ++i) {
        input_desc[i] = inputDesc_->at(i)->desc();
        inputs[i] = inputs_->at(i)->buffer();
    }
    for (int i = 0; i < num_outputs; ++i) {
        output_desc[i] = outputDesc_->at(i)->desc();
        outputs[i] = (aclDataBuffer *)outputs_->at(i)->buffer();
    }
    return OpCompiler::opCompileAndExecute(opType_,
                                           num_inputs, input_desc, inputs,
                                           num_outputs, output_desc, outputs,
                                           attr_->attr(), engineType_, compileFlag_, opPath_,
                                           stream);
}
