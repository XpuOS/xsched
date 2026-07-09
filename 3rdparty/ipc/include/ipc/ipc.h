#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ipc {

enum class NodeType {
    kSender,
    kReceiver
};

enum class ChannelType {
    kUnknown,
    kMessageQueue,
    kNamedPipe
};

class Buffer {
    void* data_;
    size_t data_size_;

public:
    Buffer(void* data, size_t size)
        : data_(data)
        , data_size_(size)
    {
    }
    ~Buffer() { free(data_); }

    size_t Size() { return data_size_; }
    void SetSize(size_t size) { data_size_ = size; }
    void* Data() { return data_; }
};

class Channel {
public:
    Channel() = default;
    virtual ~Channel() = default;

    // Disable copy constructor and assignment operator
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    virtual bool Send(const void* data, size_t data_size) = 0;
    virtual std::shared_ptr<Buffer> Receive() = 0;
    virtual bool Remove() = 0;
};

class Node {
public:
    Node(std::string name, NodeType ntype, ChannelType ctype = ChannelType::kUnknown);
    ~Node();

    // Disable copy constructor and assignment operator
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    const std::string& getName() const { return name_; }

    bool Send(const void* data, size_t data_size);
    std::shared_ptr<Buffer> Receive();
    bool Remove();

private:
    const std::string name_;           // Name of the IPC Node
    const NodeType node_type_;         // Type of the Node (kSender or kReceiver)
    std::shared_ptr<Channel> channel_; // Pointer to the underlying IPC channel
};

} // namespace ipc
