#ifdef _WIN32
#include <atomic>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "ipc/pipe/pipe.h"
#include "utils/assert.h"
#include "utils/log.h"

namespace pipe {

// The name of the named pipeline follows the following format:
// "\\<ServerName>\pipe\<PipeName>"
// - ServerName can be the name of a remote computer or a dot (.) representing the local computer.
// - PipeName is the name given to a pipeline, which can contain any character except for the back slash (\),
//   and the length of the entire name string should be limited to 256 characters.
//   Pipeline names are not sensitive to uppercase and lowercase letters.
// Due to the inability of the server to create pipelines on remote hosts, the <ServerName> section can only be a single point.
// The client can write <ServerName> as a dot or remote hostname.
#define LOCAL_PIPI R"(\\.\pipe\)"

NamedPipe::NamedPipe(const std::string& name, NodeType ntype)
    : pipe_name_(LOCAL_PIPI + name)
    , node_type_(ntype)
{
    switch (ntype) {
    case NodeType::kSender:
        send_pipe_ = INVALID_HANDLE_VALUE;
        send_connected_ = false;
        // Move connection establishment to send method
        // Prevent errors caused by not creating a receiver during initialization
        break;
    case NodeType::kReceiver:
        // Start the receiver thread
        recv_thread_ = std::thread(&NamedPipe::RecvLoop, this);
        recv_stop_event_ = CreateEvent(
            NULL, // default security attributes
            TRUE, // manual-reset event
            FALSE, // initial state is nonsignaled
            NULL // object name
        );
        XASSERT_EXIT(recv_stop_event_ == NULL, "CreateEvent failed");
        recv_stop_flag_.store(false);
        break;
    default:
        XASSERT_RETURN(true, , "Unknown link type in NamedPipe constructor");
        break;
    }
}

NamedPipe::~NamedPipe()
{
    NamedPipe::Remove();
}

bool NamedPipe::Send(const void* data, size_t data_size)
{
    XASSERT_RETURN(node_type_ == NodeType::kReceiver, false, "kReceiver can't send data");
    XASSERT_RETURN(!data, false, "Data is null");
    XASSERT_RETURN(!Connect(), false, "Connect failed in send");

    DWORD bytesWritten;
    while (true) {
        BOOL success = WriteFile(
            send_pipe_,
            data,
            static_cast<DWORD>(data_size),
            &bytesWritten,
            NULL);

        if (!success && GetLastError() == ERROR_PIPE_NOT_CONNECTED) {
            // The pipe has been closed by Receiver
            XINFO("The pipe has been ended, try to reconnect...");
            send_connected_ = false;
            // Reset the current pipe instance and reconnect
            DisconnectNamedPipe(send_pipe_);
            Connect();
            // Try sending data again after reconnecting
            continue;
        }
        XASSERT_RETURN(!success, false, "WriteFile failed");
        break;
    }

    XASSERT_RETURN(bytesWritten != data_size, false, "WriteFile write wrong size data, expected: %zu, written: %lu", data_size, bytesWritten);
    XDEBG("kSender '%s' write %lu byte", pipe_name_.c_str(), bytesWritten);
    return true;
}

std::shared_ptr<Buffer> NamedPipe::Receive()
{
    // The constructor will automatically lock queue_mutex_
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // First check if there is any readable data (to avoid loss notifications)
    // Prevent the function from not being called before receiving the RecvHandle notification
    if (recv_queue_.empty()) {
        // The wait method will release the mutex held by the lock
        // and block the current thread until other threads call notify_one() or notify_all() to wake it up.
        queue_cv_.wait(lock, [this] {
            // After waking up, the thread will reacquire the lock and check the predicate condition
            // If true, continue with the execution
            // If false, release the lock again and block
            XDEBG("Receiver '%s' queue is empty, waiting ...", pipe_name_.c_str());
            return !recv_queue_.empty() || recv_stop_flag_.load();
        });
    }

    XASSERT_RETURN(recv_queue_.empty(), nullptr, "recv_queue_ is empty");

    auto result = recv_queue_.front();
    recv_queue_.pop();
    XDEBG("Receiver '%s' pop data from queue", pipe_name_.c_str());

    return result;
}

bool NamedPipe::Remove()
{
    XDEBG("Removing named pipe %s '%s'", node_type_ == NodeType::kSender ? "sender" : "receiver", pipe_name_.c_str());

    if (node_type_ == NodeType::kSender && send_connected_) {
        DisconnectNamedPipe(send_pipe_);
        send_connected_ = false;
        return true;
    } else {
        recv_stop_flag_.store(true);
        SetEvent(recv_stop_event_);

        queue_mutex_.lock();
        queue_cv_.notify_all();
        XASSERT(!recv_queue_.empty(), "There is unread data in the queue");
        std::queue<std::shared_ptr<Buffer>>().swap(recv_queue_);
        queue_mutex_.unlock();

        if (recv_thread_.joinable()) {
            recv_thread_.join();
        }
        for (auto& thread : recv_handle_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        return true;
    }
}

void NamedPipe::RecvLoop()
{
    while (!recv_stop_flag_.load()) {
        HANDLE pipe = CreateNamedPipeA(
            pipe_name_.c_str(),
            // CreateNamedPipeA     | CreateFileA                 | Description
            // ---------------------|-----------------------------|------------------------------------------------------------
            // PIPE_ACCESS_INBOUND  | GENERIC_READ                | Read-only for server and write-only for client.
            // PIPE_ACCESS_OUTBOUND | GENERIC_WRITE               | Write-only for server and read-only for client.
            // PIPE_ACCESS_DUPLEX   | GENERIC_READ | GENEIC_WRITE | Pipeline can be read/written by both the server and client.
            PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED, // OpenMode
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT, // PipeMode
            PIPE_UNLIMITED_INSTANCES, // Maximum instances of pipes with the same name
            BUFFER_SIZE, // Output buffer size
            BUFFER_SIZE, // Input buffer size
            0, // Default timeout
            NULL);

        if (pipe == INVALID_HANDLE_VALUE) {
            if (GetLastError() == ERROR_PIPE_BUSY) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            XASSERT(true, "CreateNamedPipeA failed");
        }
        XDEBG("Receiver '%s' pipe created successfully", pipe_name_.c_str());

        // Set up overlapping structures and associate event
        OVERLAPPED overlapped = { 0 };
        overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        XASSERT(overlapped.hEvent == NULL, "CreateEvent failed");

        // Non blocking connection
        if (!ConnectNamedPipe(pipe, &overlapped)) {
            // ERROR_IO_PENDING indicates:
            // operation has been successfully started,
            // but it has not yet been completed
            if (GetLastError() != ERROR_IO_PENDING) {
                XASSERT(true, "ConnectNamedPipe failed");
                CloseHandle(pipe);
                CloseHandle(overlapped.hEvent);
                continue;
            }
        }

        // Waiting for connection or stop event
        HANDLE waitHandles[2] = { overlapped.hEvent, recv_stop_event_ };
        DWORD waitResult = WaitForMultipleObjects(2, waitHandles, FALSE, INFINITE);

        // Processing stop signal
        if (waitResult == WAIT_OBJECT_0 + 1) {
            CancelIo(pipe);
            CloseHandle(pipe);
            CloseHandle(overlapped.hEvent);
            break;
        }

        // Check the connection results
        DWORD bytesTransferred;
        if (!GetOverlappedResult(pipe, &overlapped, &bytesTransferred, FALSE)) {
            XASSERT(true, "GetOverlappedResult failed");
            CloseHandle(pipe);
            CloseHandle(overlapped.hEvent);
            continue;
        }

        // Connection successful, start working thread
        XDEBG("Receiver '%s' pipe connected successfully", pipe_name_.c_str());
        auto handle_thread = std::thread(&NamedPipe::RecvHandle, this, pipe);
        XASSERT_EXIT(!handle_thread.joinable(), "RecvHandle thread failed to start");
        recv_handle_threads_.push_back(std::move(handle_thread));
        CloseHandle(overlapped.hEvent);
    }
}

void NamedPipe::RecvHandle(HANDLE pipe)
{
    XDEBG("Receiver '%s' started a thread to handle connection", pipe_name_.c_str());
    char buffer[BUFFER_SIZE];
    OVERLAPPED overlapped = { 0 };
    overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    while (!recv_stop_flag_.load()) {
        XDEBG("Receiver '%s' read data", pipe_name_.c_str());
        DWORD bytesRead;
        BOOL success = ReadFile(
            pipe,
            buffer,
            BUFFER_SIZE,
            &bytesRead,
            &overlapped);

        // Handle asynchronous operations
        if (!success && GetLastError() != ERROR_IO_PENDING) {
            // Error or disconnection
            XASSERT(true, "ReadFile failed");
            break;
        }

        // Waiting for data or stop signal
        XDEBG("Receiver '%s' waiting for reading ...", pipe_name_.c_str());
        HANDLE waitHandles[2] = { overlapped.hEvent, recv_stop_event_ };
        DWORD waitResult = WaitForMultipleObjects(2, waitHandles, FALSE, INFINITE);

        // Stop Event
        if (waitResult == WAIT_OBJECT_0 + 1) {
            CancelIo(pipe);
            break;
        }

        // Obtain reading results
        if (!GetOverlappedResult(pipe, &overlapped, &bytesRead, FALSE)) {
            if (GetLastError() == ERROR_BROKEN_PIPE) {
                // The pipe has been closed by kSender
                XINFO("The pipe has been ended, close pipe instance.");
                break;
            } else
                XASSERT_RETURN(true, , "GetOverlappedResult failed");
        }

        // Processing valid data
        if (bytesRead > 0) {
            auto data = std::make_shared<Buffer>(malloc(bytesRead), bytesRead);
            memcpy(data->Data(), buffer, bytesRead);
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                recv_queue_.push(data);
                queue_cv_.notify_one();
            }
        }
        XDEBG("Receiver '%s' read %lu bytes data from pipe", pipe_name_.c_str(), bytesRead);
    }

    // Cleanup
    XDEBG("Receiver '%s' stop a handle thread", pipe_name_.c_str());
    CloseHandle(overlapped.hEvent);
    FlushFileBuffers(pipe);
    DisconnectNamedPipe(pipe);
    CloseHandle(pipe);
}

// kSender connects to Receiver
bool NamedPipe::Connect()
{
    if (send_connected_) {
        return true;
    }

    const int max_retries = 30;
    const int retry_interval_ms = 100;

    for (int i = 0; i < max_retries; ++i) {
        if (WaitNamedPipeA(pipe_name_.c_str(), NMPWAIT_USE_DEFAULT_WAIT)) {
            send_pipe_ = CreateFileA(
                pipe_name_.c_str(),
                GENERIC_WRITE, // Write only for sender
                FILE_SHARE_READ | FILE_SHARE_WRITE, // Allow sharing for reading and writing
                NULL, // Default security attributes
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                NULL);

            if (send_pipe_ != INVALID_HANDLE_VALUE) {
                send_connected_ = true;
                XDEBG("kSender '%s' connected successfully", pipe_name_.c_str());
                return true;
            }
            XASSERT(true, "CreateFile failed, retry %d times ...", i);
        }

        if (GetLastError() != ERROR_FILE_NOT_FOUND) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_ms));
    }

    XASSERT_RETURN(true, false, "Connect failed after retrying %d times", max_retries);
}

} // namespace pipe
#endif // _WIN32