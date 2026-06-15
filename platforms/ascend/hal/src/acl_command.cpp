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
    if (desc == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(tensor_desc_mutex_);
    auto it = tensor_descs_.find(desc);
    if (it != tensor_descs_.end()) return it->second;
    auto tensor_desc = std::make_shared<TensorDesc>();
    tensor_desc->desc_ = desc;
    tensor_descs_[desc] = tensor_desc;
    return tensor_desc;
}

std::shared_ptr<TensorDesc> TensorDesc::Destroy(const aclTensorDesc *desc)
{
    if (desc == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(tensor_desc_mutex_);
    auto it = tensor_descs_.find(desc);
    if (it == tensor_descs_.end()) return nullptr;
    auto tensor_desc = it->second;
    tensor_descs_.erase(it);
    return tensor_desc;
}

DataBuffer::~DataBuffer()
{
    if (buffer_) ACL_ASSERT(Driver::DestroyDataBuffer(buffer_));
    if (snapshot_data_) free(snapshot_data_);
}

bool DataBuffer::IsHostAddr(const void *addr)
{
    aclrtPtrAttributes attributes;
    aclError ret = Driver::rtPointerGetAttributes(addr, &attributes);
    if (ret != ACL_SUCCESS) return true;
    return attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE;
}

std::shared_ptr<DataBuffer> DataBuffer::Create(const aclDataBuffer *buffer, bool deep_copy)
{
    if (buffer == nullptr) return nullptr;
    void *addr = Driver::GetDataBufferAddr(buffer);
    size_t size = Driver::GetDataBufferSizeV2(buffer);
    auto data_buffer = std::make_shared<DataBuffer>();

    if (deep_copy && addr != nullptr && size > 0 && IsHostAddr(addr)) {
        // create a deep copy snapshot, no need to add to data buffer maps,
        // since the copy will only be used inside XSched, and the
        // original buffer will only be used outside XSched (in applications).
        XDEBG("deep copy host input %zu B", size);
        void *snapshot_data = malloc(size);
        XASSERT(snapshot_data != nullptr, "failed to malloc host snapshot data");
        memcpy(snapshot_data, addr, size);
        aclDataBuffer *snapshot_buffer = Driver::CreateDataBuffer(snapshot_data, size);
        XASSERT(snapshot_buffer != nullptr, "failed to create host snapshot aclDataBuffer");
        data_buffer->buffer_ = snapshot_buffer;
        data_buffer->snapshot_data_ = snapshot_data;
        return data_buffer;
    }

    std::lock_guard<std::mutex> lock(data_buffer_mutex_);
    auto it = data_buffers_.find(buffer);
    if (it != data_buffers_.end()) return it->second;
    data_buffer->buffer_ = buffer;
    data_buffers_[buffer] = data_buffer;
    return data_buffer;
}

std::shared_ptr<DataBuffer> DataBuffer::Destroy(const aclDataBuffer *buffer)
{
    if (buffer == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(data_buffer_mutex_);
    auto it = data_buffers_.find(buffer);
    if (it == data_buffers_.end()) return nullptr;
    auto data_buffer = it->second;
    data_buffers_.erase(it);
    return data_buffer;
}

std::shared_ptr<OpAttr> OpAttr::Create(const aclopAttr *attr)
{
    if (attr == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(op_attr_mutex_);
    auto it = op_attrs_.find(attr);
    if (it != op_attrs_.end()) return it->second;
    auto op_attr = std::make_shared<OpAttr>();
    op_attr->attr_ = attr;
    op_attrs_[attr] = op_attr;
    return op_attr;
}

std::shared_ptr<OpAttr> OpAttr::Destroy(const aclopAttr *attr)
{
    if (attr == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(op_attr_mutex_);
    auto it = op_attrs_.find(attr);
    if (it == op_attrs_.end()) return nullptr;
    auto op_attr = it->second;
    op_attrs_.erase(it);
    return op_attr;
}

AclOpCompileAndExecuteCommand::AclOpCompileAndExecuteCommand(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, bool deep_copy_inputs)
: op_type_(opType ? opType : ""), op_path_(opPath ? opPath : "")
, engine_type_(engineType), compile_flag_(compileFlag)
{
    input_descs_.reserve(numInputs);
    inputs_.reserve(numInputs);
    output_descs_.reserve(numOutputs);
    outputs_.reserve(numOutputs);

    input_descs_ptrs_.reserve(numInputs);
    inputs_ptrs_.reserve(numInputs);
    output_descs_ptrs_.reserve(numOutputs);
    outputs_ptrs_.reserve(numOutputs);

    for (int i = 0; i < numInputs; ++i) {
        auto input_desc = TensorDesc::Create(inputDesc[i]);
        auto input = DataBuffer::Create(inputs[i], deep_copy_inputs);

        input_descs_.push_back(input_desc);
        inputs_.push_back(input);
        input_descs_ptrs_.push_back(input_desc ? input_desc->desc() : nullptr);
        inputs_ptrs_.push_back(input ? input->buffer() : nullptr);
    }
    for (int i = 0; i < numOutputs; ++i) {
        auto output_desc = TensorDesc::Create(outputDesc[i]);
        auto output = DataBuffer::Create(outputs[i], false);

        output_descs_.push_back(output_desc);
        outputs_.push_back(output);
        output_descs_ptrs_.push_back(output_desc ? output_desc->desc() : nullptr);
        outputs_ptrs_.push_back(output ? (aclDataBuffer *)output->buffer() : nullptr);
    }
    attr_ = OpAttr::Create(attr);
}

aclError AclOpCompileAndExecuteCommand::Launch(aclrtStream stream)
{
    return OpCompiler::opCompileAndExecute(op_type_.empty() ? nullptr : op_type_.c_str(),
        input_descs_ptrs_.size(), input_descs_ptrs_.data(), inputs_ptrs_.data(),
        output_descs_ptrs_.size(), output_descs_ptrs_.data(), outputs_ptrs_.data(),
        attr_ ? attr_->attr() : nullptr, engine_type_, compile_flag_,
        op_path_.empty() ? nullptr : op_path_.c_str(), stream);
}
