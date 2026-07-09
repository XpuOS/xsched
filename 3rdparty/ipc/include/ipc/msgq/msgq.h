#pragma once

#include <cstddef>
#include <cstdint>

#include "ipc/ipc.h"

using namespace ipc;

#ifdef _WIN32

#include "boost/interprocess/ipc/message_queue.hpp"

namespace msgq {

class MessageQueue : public Channel {
public:
    MessageQueue(std::string name, NodeType ntype);
    ~MessageQueue();

    bool Send(const void* data, size_t data_size = 0) override;
    std::shared_ptr<Buffer> Receive() override;
    bool Remove() override;

private:
    const std::string msgq_name_;
    const NodeType node_type_;

    const int max_msg_size_ = 1024; // Default max message size
    const int max_msg_count_ = 100; // Default max message count

    std::unique_ptr<boost::interprocess::message_queue> message_queue_;
};

} // namespace msgq

#else

#include <sys/msg.h>

namespace msgq {

class MessageQueue : public Channel {
public:
    MessageQueue(std::string name, NodeType ntype, key_t key);
    ~MessageQueue();

    bool Send(const void* data, size_t data_size = 0) override;
    std::shared_ptr<Buffer> Receive() override;
    bool Remove() override;

private:
    const std::string msgq_name_;
    const NodeType node_type_;
    const key_t key_;

    int msgid_ = -1;
    msglen_t max_msg_size_ = 0;

    static constexpr long MESSAGE_TYPE = 1;
    struct Message {
        long mtype; // Message type, required by System V communication standards
        size_t size;
        char data[];
    };
};

} // namespace msgq

#endif // _WIN32
