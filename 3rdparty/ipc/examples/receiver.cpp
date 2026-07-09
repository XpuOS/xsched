#include <ipc/ipc.h>

int main()
{
    // Create an IPC Node named "example"
    ipc::Node receiver("example", ipc::NodeType::kReceiver);

    // Receive the data
    auto rec = receiver.Receive();
    if (!rec) {
        fprintf(stderr, "Failed to Receive message\n");
        return 1;
    }
    const char* data = static_cast<const char*>(rec.get()->Data());
    // Print the received data
    printf("Received message: %s\n", data);
    return 0;
}