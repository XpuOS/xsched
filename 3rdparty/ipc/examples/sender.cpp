#include <ipc/ipc.h>
#include <string>

int main()
{
    // Example of sending a message using IPC
    std::string data = "Hello, IPC!";

    // Create an IPC Node named "example"
    ipc::Node sender("example", ipc::NodeType::kSender);

    // Send the data
    if (!sender.Send(data.c_str(), data.size() + 1)) {
        fprintf(stderr, "Failed to send message\n");
        return 1;
    }
}