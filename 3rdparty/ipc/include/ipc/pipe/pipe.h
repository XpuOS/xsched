#pragma once

#ifdef _WIN32
#include <atomic>
#include <condition_variable>
#include <queue>
#include <thread>
#include <windows.h>

#include "ipc/ipc.h"

using namespace ipc;

namespace pipe {

class NamedPipe : public Channel {
public:
    NamedPipe(const std::string& name, NodeType ntype);
    ~NamedPipe();

    bool Send(const void* data, size_t data_size);
    std::shared_ptr<Buffer> Receive();
    bool Remove();

private:
    std::string pipe_name_;
    NodeType node_type_;

    HANDLE send_pipe_;
    bool send_connected_;

    std::thread recv_thread_; // The main thread of the server
    HANDLE recv_stop_event_; // Event to notify the Receive thread to stop
    std::atomic<bool> recv_stop_flag_; // Check whether the main thread has terminated
    std::vector<std::thread> recv_handle_threads_; // Threads to handle connections

    std::queue<std::shared_ptr<Buffer>> recv_queue_;  // Queue to store received data
    std::condition_variable queue_cv_; // Wake up the Receive() upon receiving data
    std::mutex queue_mutex_;

    static const DWORD BUFFER_SIZE = 4096; // Default buffer size for named pipe communication

    bool Connect();
    void RecvLoop();
    void RecvHandle(HANDLE pipe);
};
} // namespace pipe
#endif // _WIN32
