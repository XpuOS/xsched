# C++ IPC Library

## English | [简体中文](docs/README_zh-CN.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/leinfinitr/ipc/blob/main/LICENSE)

## IPC communication library with optional communication methods

This repository implements a C++ IPC library based on the **Linux System V IPC interface** and **Windows communication interface**, encapsulating various IPC mechanisms such as message queues and named pipes. This library aims to simplify communication operations between processes.

### Characteristics

- Supports both Linux and Windows operating systems, as well as multiple compilers such as GCC, MinGW, MSVC, etc.
- When creating a communication node abstraction `IPC::node`, the `IPC::ChannelType` parameter can be used to specify the underlying IPC channel type, such as `messageQueue`, `NamedPipe`, etc.

### Compile and Test

1. Clone the repository: `git clone https://github.com/leinfinitr/ipc.git`
2. Navigate to the project directory: `cd ipc`
3. Initialize submodules: `git submodule update --init --recursive`
4. Compile: `make`
5. After compilation, the following files will be generated in the `output` directory:
   1. `ipc-test-xxx` executable test file
   2. Static library file `libipc.a` (Linux) or `ipc.lib` (Windows).

- Correctness Test: `/output/bin/ipc-test-correctness`
- Performance Test: run `/output/bin/ ipc-test-performance-server` and `/output/bin/ipc-test-performance-client` sequentially on different terminals.

### Communication method support

- ✅ Realized
- 🔘 Unrealized
- 🚧 Implementing

<table>
<tr>
<th rowspan="2" align="center" class="vertical-center">Communication method</th>
<th colspan="2" align="center">OS</th>
</tr>
<tr>
<th align="center">Windows</th>
<th align="center">Linux</th>
</tr>
<tr>
<td align="center">Named pipe</td>
<td align="center">(Windows NamedPipe) ✅</td>
<td align="center">🔘</td>
</tr>
<tr>
<td align="center">Message queue</td>
<td align="center">(Boost Interprocess) ✅</td>
<td align="center">(System V IPC) ✅</td>
</tr>
<tr>
<td align="center">Shared memory</td>
<td align="center">🚧</td>
<td align="center">🚧</td>
</tr>
</table>

### Usage method

#### Introduce IPC library

For projects built using CMake, the IPC library can be introduced through the following methods:

```cmake
set(IPC_LIB_PATH "/path/to/libipc.a")
set(IPC_INCLUDE_DIR "/path/to/ipc/include")
add_library(ipc STATIC IMPORTED)
set_target_properties(ipc PROPERTIES
    IMPORTED_LOCATION "${IPC_LIB_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${IPC_INCLUDE_DIR}"
)

target_link_libraries(your_target ipc)
target_include_directories(your_target PRIVATE ${IPC_INCLUDE_DIR})
```

Alternatively, it can be used directly as a third-party library in CMakeLists.txt:

```cmake
add_subdirectory(ipc EXCLUDE_FROM_ALL)
set(IPC_INCLUDE_DIR "/path/to/ipc/include")
target_link_libraries(your_target ipc)
target_include_directories(your_target PRIVATE ${IPC_INCLUDE_DIR})
```

#### Code Writing

```cpp
#include <ipc/ipc.h>

// Create two IPC nodes named 'Wow', using NamedPipe or MessageQueue (default) at the bottom 
// ipc::node receiver("Wow", ipc::NodeType::Receiver, ipc::ChannelType::NamedPipe);
// ipc::node receiver("Wow", ipc::NodeType::Receiver, ipc::ChannelType::MessageQueue);
ipc::node receiver("Wow", ipc::NodeType::kReceiver);
ipc::node sender("Wow", ipc::NodeType::kSender);
auto rec = receiver.Receive();   // Receive message (will block the process until the message is received)
sender.Send(data, sizeof(data)); // Send a message
```

### Example

Two example programs are provided in the `examples` directory: `sender.cpp` and `receiver.cpp`.

The usage method is as follows:

```bash
cd examples
# Compile
make
# Run receiver
make run_receiver
# Run sender on the new terminal
make run_sender
# At this time, the receiver can receive the message
Received message: Hello, IPC!
```

### Performance

- **Windows**: Intel (R) Core (TM) Ultra 5 225
- **Linux**: Intel (R) Core (TM) Ultra 9 185H

<table>
<tr>
<th rowspan="2" align="center" class="vertical-center">Communication latency / µs</th>
<th colspan="3" align="center">Windows</th>
<th colspan="2" align="center">Linux</th>
</tr>
<tr>
<th align="center">Message queue</th>
<th align="center">Named pipe</th>
<th align="center"><a href="https://github.com/mutouyun/cpp-ipc">cpp-ipc</a></th>
<th align="center">Message queue</th>
<th align="center"><a href="https://github.com/mutouyun/cpp-ipc">cpp-ipc</a></th>
</tr>
<tr>
<td align="center">Average</td>
<td align="center">0.952</td>
<td align="center">21.9</td>
<td align="center">253.7</td>
<td align="center">61.5</td>
<td align="center">47.0</td>
</tr>
<tr>
<td align="center">Median</td>
<td align="center">0.900</td>
<td align="center">19.7</td>
<td align="center">220.3</td>
<td align="center">54.0</td>
<td align="center">45.9</td>
</tr>
<tr>
<td align="center">P95</td>
<td align="center">1.10</td>
<td align="center">28.5</td>
<td align="center">488.4</td>
<td align="center">89.1</td>
<td align="center">60.1</td>
</tr>
<tr>
<td align="center">P99</td>
<td align="center">1.20</td>
<td align="center">56.5</td>
<td align="center">588.9</td>
<td align="center">119.5</td>
<td align="center">79.1</td>
</tr>
</table>
