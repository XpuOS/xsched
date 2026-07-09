#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ipc/ipc.h"
#include "ipc/msgq/msgq.h"
#include "ipc/pipe/pipe.h"
#include "utils/assert.h"
#include "utils/log.h"

#ifndef _WIN32
#include <sys/ipc.h> // For IPC_PRIVATE
#endif

namespace ipc {

Node::Node(std::string name, NodeType ntype, ChannelType ctype)
    : name_(name)
    , node_type_(ntype)
{
#ifdef _WIN32
    switch (ctype) {
    case ChannelType::kMessageQueue:
        channel_ = std::make_shared<msgq::MessageQueue>(name, ntype);
        break;
    case ChannelType::kNamedPipe:
        channel_ = std::make_shared<pipe::NamedPipe>(name, ntype);
        break;
    default:
        channel_ = std::make_shared<msgq::MessageQueue>(name, ntype);
    }
#else
    std::hash<std::string> hasher;
    size_t hash = hasher(name);
    key_t key = static_cast<key_t>(hash & 0xFFFFFFFF);
    XASSERT_EXIT(key == IPC_PRIVATE, "Generated key is IPC_PRIVATE, which is invalid");

    switch (ctype) {
    case ChannelType::kMessageQueue:
        channel_ = std::make_shared<msgq::MessageQueue>(name, ntype, key);
        break;
    case ChannelType::kNamedPipe:
        XASSERT_EXIT(true, "Named pipe channel is not supported on Linux.");
    default:
        channel_ = std::make_shared<msgq::MessageQueue>(name, ntype, key);
    }
#endif
}

Node::~Node()
{
    Remove();
}

bool Node::Send(void const* data, size_t data_size)
{
    XASSERT_RETURN(!channel_, false, "Channel not initialized");
    XASSERT_RETURN(node_type_ != NodeType::kSender, false, "Cannot Send data from a Receiver Node");

    return channel_->Send(data, data_size);
}

std::shared_ptr<Buffer> Node::Receive()
{
    XASSERT_RETURN(!channel_, nullptr, "Channel not initialized");
    XASSERT_RETURN(node_type_ != NodeType::kReceiver, nullptr, "Cannot Receive data from a kSender Node");

    return channel_->Receive();
}

bool Node::Remove()
{
    if (channel_) {
        bool result = channel_->Remove();
        channel_.reset(); // Reset the shared pointer to release the channel
        return result;
    }
    return true; // No channel to disconnect
}

} // namespace ipc