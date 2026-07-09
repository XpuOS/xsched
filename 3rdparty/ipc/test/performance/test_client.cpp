#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "ipc/ipc.h"

// Configuration
const int NUM_ITERATIONS = 1000;
const int WARMUP_ITERATIONS = 100;

int64_t current_timestamp()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

int main()
{
    // Initialize separate channels for sending and receiving
    ipc::Node sender("ipc-latency-request", ipc::NodeType::kSender);
    ipc::Node receiver("ipc-latency-response", ipc::NodeType::kReceiver);

    std::cout << "Connecting to IPC server..." << std::endl;
    std::cout << "Sending on channel: ipc-latency-request" << std::endl;
    std::cout << "Receiving on channel: ipc-latency-response" << std::endl;

    int random = rand() % 1000;
    std::string base_msg = "IPC " + std::to_string(random);

    // Warmup
    std::cout << "Warming up for " << WARMUP_ITERATIONS << " iterations..." << std::endl;
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // Send message
        std::string msg = base_msg + " - Warmup #" + std::to_string(i + 1);
        sender.Send(msg.c_str(), msg.size() + 1);

        // Wait for reply
        auto reply = receiver.Receive();
        if (!reply) {
            std::cerr << "Error receiving reply during warmup" << std::endl;
            continue;
        }

        // Check if the reply matches the sent message
        const char* res = static_cast<const char*>(reply.get()->Data());
        if (strcmp(res, msg.c_str()) != 0) {
            std::cerr << "Warmup message mismatch: expected '" << msg << "', got '" << res << "'" << std::endl;
        }
    }

    // Actual latency test
    std::vector<double> latencies;
    latencies.reserve(NUM_ITERATIONS);

    std::cout << "Running latency test for " << NUM_ITERATIONS << " iterations..." << std::endl;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // Create message
        std::string msg = base_msg + " - Test #" + std::to_string(i + 1);
        int64_t send_time = current_timestamp();

        // Send message
        sender.Send(msg.c_str(), msg.size() + 1);

        // Wait for reply
        auto reply = receiver.Receive();
        if (!reply) {
            std::cerr << "Error receiving reply" << std::endl;
            continue;
        }

        // Calculate round trip time
        int64_t recv_time = current_timestamp();
        double latency = static_cast<double>(recv_time - send_time);
        latencies.push_back(latency);

        // Check if the reply matches the sent message
        const char* res = static_cast<const char*>(reply.get()->Data());
        if (strcmp(res, msg.c_str()) != 0) {
            std::cerr << "Test message mismatch: expected '" << msg << "', got '" << res << "'" << std::endl;
        }
    }

    // Calculate statistics
    double min_latency = *std::min_element(latencies.begin(), latencies.end());
    double max_latency = *std::max_element(latencies.begin(), latencies.end());
    double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    // Calculate median and percentiles
    std::sort(latencies.begin(), latencies.end());
    double median = latencies[latencies.size() / 2];
    double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nLatency Results (IPC channel, nanoseconds):" << std::endl;
    std::cout << "  Minimum: " << min_latency << " ns" << std::endl;
    std::cout << "  Maximum: " << max_latency << " ns" << std::endl;
    std::cout << "  Average: " << avg_latency << " ns" << std::endl;
    std::cout << "  Median:  " << median << " ns" << std::endl;
    std::cout << "  95th percentile: " << p95 << " ns" << std::endl;
    std::cout << "  99th percentile: " << p99 << " ns" << std::endl;

    return 0;
}
