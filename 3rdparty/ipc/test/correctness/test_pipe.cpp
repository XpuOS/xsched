#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "ipc/ipc.h"

using namespace ipc;

void pipe_basic()
{
    const char* msg = "Hello, IPC!";

    std::thread server_thread([msg]() {
        ipc::Node server_node("basic", ipc::NodeType::kReceiver, ipc::ChannelType::kNamedPipe);
        auto rec = server_node.Receive();
        if (!rec) {
            fprintf(stderr, "Server failed to Receive message\n");
            exit(1);
        }
        const char* res = static_cast<const char*>(rec.get()->Data());
        EXPECT_STREQ(res, msg);
    });

    // Ensure that the server is started and waiting for connection
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ipc::Node client_node("basic", ipc::NodeType::kSender, ipc::ChannelType::kNamedPipe);
    // +1 for null terminator
    // In some versions of compilers and dynamic library dependencies, errors may occur if there is no "+ 1" for strlen
    // But sometimes even without "+ 1", there won't be any errors
    EXPECT_TRUE(client_node.Send(msg, strlen(msg) + 1));

    server_thread.join();
}

void pipe_loop()
{
    const char* msg = "Hello, IPC";

    std::thread server_thread([msg]() {
        ipc::Node server_node("loop", ipc::NodeType::kReceiver, ipc::ChannelType::kNamedPipe);
        for (int i = 0; i < 100; ++i) {
            auto rec = server_node.Receive();
            if (!rec) {
                fprintf(stderr, "Failed to Receive message\n");
                exit(1);
            }
            const char* res = static_cast<const char*>(rec.get()->Data());
            std::string expected_msg = std::string(msg) + " - Message #" + std::to_string(i + 1);
            EXPECT_STREQ(res, expected_msg.c_str());
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ipc::Node client_node("loop", ipc::NodeType::kSender, ipc::ChannelType::kNamedPipe);
    for (int i = 0; i < 100; ++i) {
        std::string full_msg = std::string(msg) + " - Message #" + std::to_string(i + 1);
        EXPECT_TRUE(client_node.Send(full_msg.c_str(), full_msg.size() + 1));
    }

    server_thread.join();
}

void pipe_struct()
{
    struct meta {
        int id;
        char name[50];
        double value;
    };
    struct message {
        meta meta_info;
        long num;
        char mtext[256];
    };

    message msg;
    msg.meta_info.id = 1;
    strcpy(msg.meta_info.name, "Test Message");
    msg.meta_info.value = 42.0;
    msg.num = 1;
    strcpy(msg.mtext, "Hello, IPC with struct!");

    std::thread server_thread([msg]() {
        ipc::Node server_node("struct", ipc::NodeType::kReceiver, ipc::ChannelType::kNamedPipe);
        auto rec = server_node.Receive();
        if (!rec) {
            fprintf(stderr, "Server failed to Receive message\n");
            exit(1);
        }
        auto received_msg = static_cast<message*>(rec.get()->Data());
        EXPECT_EQ(received_msg->meta_info.id, msg.meta_info.id);
        EXPECT_STREQ(received_msg->meta_info.name, msg.meta_info.name);
        EXPECT_DOUBLE_EQ(received_msg->meta_info.value, msg.meta_info.value);
        EXPECT_EQ(received_msg->num, msg.num);
        EXPECT_STREQ(received_msg->mtext, msg.mtext);
    });

    // Ensure that the server is started and waiting for connection
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ipc::Node client_node("struct", ipc::NodeType::kSender, ipc::ChannelType::kNamedPipe);
    EXPECT_TRUE(client_node.Send(&msg, sizeof(msg)));

    server_thread.join();
}

void pipe_class()
{
    class MyClass {
    public:
        int id;
        char name[50];
        double value;

        MyClass(int i, const char* n, double v)
            : id(i)
            , value(v)
        {
            strncpy(name, n, sizeof(name) - 1);
        }
    };

    MyClass obj(1, "Test Class", 3.14);

    std::thread server_thread([obj]() {
        ipc::Node server_node("class", ipc::NodeType::kReceiver, ipc::ChannelType::kNamedPipe);
        auto rec = server_node.Receive();
        if (!rec) {
            fprintf(stderr, "Server failed to Receive message\n");
            exit(1);
        }
        auto received_obj = static_cast<MyClass*>(rec.get()->Data());
        EXPECT_EQ(received_obj->id, obj.id);
        EXPECT_STREQ(received_obj->name, obj.name);
        EXPECT_DOUBLE_EQ(received_obj->value, obj.value);
    });

    // Ensure that the server is started and waiting for connection
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ipc::Node client_node("class", ipc::NodeType::kSender, ipc::ChannelType::kNamedPipe);
    EXPECT_TRUE(client_node.Send(&obj, sizeof(MyClass)));

    server_thread.join();
}

void pipe_subclass()
{
    class Base {
    public:
        int id;
        char name[50];

        Base(int i, const char* n)
            : id(i)
        {
            strncpy(name, n, sizeof(name) - 1);
        }
    };

    class Derived : public Base {
    public:
        double value;

        Derived(int i, const char* n, double v)
            : Base(i, n)
            , value(v)
        {
        }
    };

    Derived obj(1, "Test Subclass", 2.718);

    std::thread server_thread([obj]() {
        ipc::Node server_node("subclass", ipc::NodeType::kReceiver, ipc::ChannelType::kNamedPipe);
        auto rec = server_node.Receive();
        if (!rec) {
            fprintf(stderr, "Server failed to Receive message\n");
            exit(1);
        }
        auto received_obj = static_cast<Derived*>(rec.get()->Data());
        EXPECT_EQ(received_obj->id, obj.id);
        EXPECT_STREQ(received_obj->name, obj.name);
        EXPECT_DOUBLE_EQ(received_obj->value, obj.value);
    });

    // Ensure that the server is started and waiting for connection
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ipc::Node client_node("subclass", ipc::NodeType::kSender, ipc::ChannelType::kNamedPipe);
    EXPECT_TRUE(client_node.Send(&obj, sizeof(Derived)));

    server_thread.join();
}

void pipe_multiterminal()
{
    const char* msg = "Hello, IPC";

    std::thread client_thread_1([msg]() {
        ipc::Node client_node_1("multiterminal", ipc::NodeType::kSender, ipc::ChannelType::kNamedPipe);
        std::string full_msg = std::string(msg) + " - Message #1";

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        EXPECT_TRUE(client_node_1.Send(full_msg.c_str(), full_msg.size() + 1));
    });

    std::thread client_thread_2([msg]() {
        ipc::Node client_node_2("multiterminal", ipc::NodeType::kSender, ipc::ChannelType::kNamedPipe);
        std::string full_msg = std::string(msg) + " - Message #2";

        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        EXPECT_TRUE(client_node_2.Send(full_msg.c_str(), full_msg.size() + 1));
    });

    std::thread client_thread_3([msg]() {
        ipc::Node client_node_3("multiterminal", ipc::NodeType::kSender, ipc::ChannelType::kNamedPipe);
        std::string full_msg = std::string(msg) + " - Message #3";

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        EXPECT_TRUE(client_node_3.Send(full_msg.c_str(), full_msg.size() + 1));
    });

    ipc::Node server_node("multiterminal", ipc::NodeType::kReceiver, ipc::ChannelType::kNamedPipe);
    for (int i = 0; i < 3; ++i) {
        auto rec = server_node.Receive();
        if (!rec) {
            fprintf(stderr, "Failed to Receive message\n");
            exit(1);
        }
        const char* res = static_cast<const char*>(rec.get()->Data());
        std::string expected_msg = std::string(msg) + " - Message #" + std::to_string(i + 1);
        EXPECT_STREQ(res, expected_msg.c_str());
    }

    client_thread_1.join();
    client_thread_2.join();
    client_thread_3.join();
}

TEST(PIPE, basic)
{
    pipe_basic();
}

TEST(PIPE, loop)
{
    pipe_loop();
}

TEST(PIPE, struct)
{
    pipe_struct();
}

TEST(PIPE, class)
{
    pipe_class();
    pipe_subclass();
}

TEST(PIPE, multiterminal)
{
    pipe_multiterminal();
}