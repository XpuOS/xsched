#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ipc/msgq/msgq.h"
#include "utils/assert.h"
#include "utils/log.h"

#ifdef _WIN32

#include "boost/interprocess/ipc/message_queue.hpp"

using namespace boost::interprocess;

namespace msgq {

MessageQueue::MessageQueue(std::string name, NodeType ntype)
    : msgq_name_(name)
    , node_type_(ntype)
{
    switch (ntype) {
    case NodeType::kReceiver:
        message_queue::remove(msgq_name_.c_str());
        try {
            message_queue_ = std::make_unique<message_queue>(
                create_only,
                msgq_name_.c_str(),
                max_msg_count_, // Max number of messages
                max_msg_size_   // Max message size
            );
        } catch (const interprocess_exception& e) {
            XASSERT_EXIT(true, "Receiver %s create failed: %s", msgq_name_.c_str(), e.what());
        }
        XDEBG("kReceiver (MessageQueue) '%s' created successfully", msgq_name_.c_str());
        break;
    case NodeType::kSender:
        // Move connection establishment to Send method
        // Prevent errors caused by not creating a receiver during initialization
        break;
    default:
        XASSERT_EXIT(true, "Unknown NodeType %d for node %s", static_cast<int>(ntype), msgq_name_.c_str());
        break;
    }
}

MessageQueue::~MessageQueue()
{
    MessageQueue::Remove();
}

bool MessageQueue::Send(const void* data, size_t data_size)
{
    if (!message_queue_)
        try {
            message_queue_ = std::make_unique<message_queue>(
                open_only,
                msgq_name_.c_str());
            XDEBG("kSender (MessageQueue) '%s' opened successfully", msgq_name_.c_str());
        } catch (const interprocess_exception& e) {
            XASSERT_EXIT(true, "Sender %s create failed: %s", msgq_name_.c_str(), e.what());
        }

    XASSERT_RETURN(node_type_ == NodeType::kReceiver, false, "kReceiver can't send data");
    XASSERT_RETURN(!data, false, "Data is null");
    XASSERT_RETURN(!message_queue_, false, "Message queue is not initialized");

    try {
        message_queue_->send(data, data_size, 0);
        XDEBG("kSender '%s' sent %zu bytes", msgq_name_.c_str(), data_size);
        return true;
    } catch (const interprocess_exception& e) {
        XERRO("Sender %s Send failed: %s", msgq_name_.c_str(), e.what());
        return false;
    }
}

std::shared_ptr<Buffer> MessageQueue::Receive()
{
    XASSERT_RETURN(!message_queue_, nullptr, "Message queue is not initialized");

    try {
        size_t received_size;
        unsigned int priority;
        auto buffer = std::make_shared<Buffer>(malloc(max_msg_size_), max_msg_size_);

        XDEBG("Receiver '%s' waiting for data...", msgq_name_.c_str());
        message_queue_->receive(buffer->Data(), buffer->Size(), received_size, priority);
        XDEBG("Receiver '%s' received %zu bytes", msgq_name_.c_str(), received_size);
        buffer->SetSize(received_size);

        return buffer;

    } catch (const interprocess_exception& e) {
        XERRO("Receiver %s Receive failed: %s", msgq_name_.c_str(), e.what());
        return nullptr;
    }
}

bool MessageQueue::Remove()
{
    try {
        if (node_type_ == NodeType::kReceiver) {
            XDEBG("Removing message queue '%s'", msgq_name_.c_str());
            return message_queue::remove(msgq_name_.c_str());
        } else
            return true;
    } catch (const interprocess_exception& e) {
        XERRO("Receiver %s Remove failed: %s", msgq_name_.c_str(), e.what())
        return false;
    }
}

} // namespace msgq

#else

#include <sys/ipc.h>
#include <sys/types.h>
#include <unistd.h>

namespace msgq {

MessageQueue::MessageQueue(std::string name, NodeType ntype, key_t key)
    : msgq_name_(name)
    , node_type_(ntype)
    , key_(key)
{
    switch (ntype) {
    case NodeType::kReceiver:
        // IPC_CREAT: create the message queue if it does not exist
        // IPC_EXCL: fail if key exists, must be used together with IPC_CREAT
        // Octal number determines the access permissions to the message queue
        // 6 (110 binary) represents read and write permissions (4 read + 2 write)
        // 0666 indicates that all users (owners, groups, and others) can read and write to this message queue
        msgid_ = msgget(key, IPC_EXCL | IPC_CREAT | 0666);
        if (msgid_ == -1) {
            // -1 indicates that msgq already exists
            // But only one receiver can exist for a message queue
            XINFO("kReceiver of Node '%s' (key: 0x%x) already exists, do you want to delete it? (y/n)", msgq_name_.c_str(), key);
            char response;
            std::cin >> response;
            XASSERT_EXIT(response == 'n' || response == 'N', "kReceiver already exists, exiting");

            // Delete the existing message queue
            XDEBG("Deleting existing message queue '%s' (key: 0x%x)", msgq_name_.c_str(), key);
            msgid_ = msgget(key, 0666);
            XASSERT_EXIT(msgid_ == -1, "Failed to get existing message queue, key: 0x%x", key);
            // IPC_RMID: Remove the message queue
            // If the message queue is not empty, it will be deleted after all messages are read
            // If the message queue is empty, it will be deleted immediately
            XASSERT_EXIT(msgctl(msgid_, IPC_RMID, nullptr) == -1, "Delete fail, msgid: %d", msgid_);
            msgid_ = msgget(key, IPC_EXCL | IPC_CREAT | 0666);
            XASSERT_EXIT(msgid_ == -1, "Failed to create new message queue, key: 0x%x", key);
        }

        struct msqid_ds queue_info;
        XASSERT_EXIT(msgctl(msgid_, IPC_STAT, &queue_info) == -1, "msgctl(IPC_STAT) fail");
        max_msg_size_ = queue_info.msg_qbytes;
        XDEBG("kReceiver (MessageQueue) '%s' (key: 0x%x) created with ID %d", msgq_name_.c_str(), key, msgid_);
        break;
    case NodeType::kSender:
        // Move connection establishment to Send method
        // Prevent errors caused by not creating a receiver during initialization
        break;
    default:
        XASSERT_EXIT(true, "Unknown NodeType %d for Node %s", static_cast<int>(ntype), msgq_name_.c_str());
        break;
    }
}

MessageQueue::~MessageQueue()
{
    MessageQueue::Remove();
}

bool MessageQueue::Send(const void* data, size_t data_size)
{
    if (msgid_ == -1) {
        msgid_ = msgget(key_, 0666);
        XASSERT_RETURN(msgid_ == -1, false, "kReceiver of Node '%s' (key: 0x%x) does not exist", msgq_name_.c_str(), key_);
        XDEBG("kSender (MessageQueue) '%s' (key: 0x%x) created with ID %d", msgq_name_.c_str(), key_, msgid_);

        // Get the maximum message size for this queue
        struct msqid_ds queue_info;
        XASSERT_RETURN(msgctl(msgid_, IPC_STAT, &queue_info) == -1, false, "msgctl(IPC_STAT) fail");
        max_msg_size_ = queue_info.msg_qbytes;
    }

    XASSERT_RETURN(!data, false, "Data is null");

    size_t total_size = sizeof(Message) + data_size;
    XASSERT_RETURN(total_size > max_msg_size_, false, "Data size %zu exceeds maximum message size %zu", data_size, max_msg_size_);

    Message* message = static_cast<Message*>(malloc(total_size));
    XASSERT_RETURN(!message, false, "malloc fail");

    message->mtype = MESSAGE_TYPE;
    message->size = data_size;
    memcpy(message->data, data, data_size);

    if (msgsnd(msgid_, message, total_size, 0) == -1) {
        // Fail reasons:
        // 1. kReceiver restart makes the msgid_ invalid
        XASSERT(true, "msgsnd fail");
        free(message);
        return false;
    }

    free(message);
    XDEBG("kSender '%s' sent %zu bytes", msgq_name_.c_str(), data_size);
    return true;
}

std::shared_ptr<Buffer> MessageQueue::Receive()
{
    std::unique_ptr<char[]> buffer(new char[max_msg_size_]);
    XASSERT_RETURN(!buffer, nullptr, "malloc fail");

    XDEBG("Receiver '%s' waiting for data...", msgq_name_.c_str());
    ssize_t received = msgrcv(msgid_, buffer.get(), max_msg_size_, 0, 0);
    XASSERT_RETURN(received == -1, nullptr, "msgrcv fail");
    XDEBG("Receiver '%s' received %ld bytes", msgq_name_.c_str(), received);

    Message* message = reinterpret_cast<Message*>(buffer.get());
    XASSERT_RETURN(static_cast<size_t>(received) != sizeof(Message) + message->size, nullptr,
        "Received size %ld does not match expected size %zu", received, sizeof(Message) + message->size);

    auto result = std::make_shared<Buffer>(malloc(message->size), message->size);
    XASSERT_RETURN(!result, nullptr, "malloc fail");
    memcpy(result->Data(), message->data, message->size);
    return result;
}

bool MessageQueue::Remove()
{
    if (node_type_ == NodeType::kReceiver) {
        XDEBG("Removing message queue '%s' (key: 0x%x) with ID %d", msgq_name_.c_str(), key_, msgid_);
        // The destructors of Node and msgq will call Remove() multiple times
        // msgctl will return -1 and set errno to EINVAL if Remove repeatedly
        XASSERT_RETURN(msgctl(msgid_, IPC_RMID, nullptr) == -1 && errno != EINVAL, false, "msgctl(IPC_RMID) fail, msgid: %d", msgid_);
    }
    return true;
}

} // namespace msgq

#endif // _WIN32