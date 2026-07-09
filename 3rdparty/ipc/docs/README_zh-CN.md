# C++ IPC Library

## [English](../README.md) | 简体中文

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/leinfinitr/ipc/blob/main/LICENSE)

## 通信方式可选的 IPC 通信库

本仓库基于 **Linux System V IPC 接口** 和 **Windows 通信接口** 实现了一个 C++ IPC 库，为消息队列、命名管道等多种 IPC 机制实现了封装。该库旨在简化进程间通信操作。

### 特点

- 同时支持 Linux 和 Windows 操作系统，以及 GCC、MinGW、MSVC 等多种编译器。
- 在创建通信节点抽象 `ipc::node` 时，可以通过 `ipc::ChannelType` 参数指定底层 IPC 通道类型，如 `MessageQueue`、`NamedPipe` 等。

### 编译与测试

1. 克隆仓库：`git clone https://github.com/leinfinitr/ipc.git`
2. 进入项目目录：`cd ipc`
3. 获取子模块：`git submodule update --init --recursive`
4. 编译：`make`
5. 编译完成后将在 `output` 目录下生成
   1. `ipc-test-xxx` 可执行测试文件
   2. `libipc.a`(Linux) 或者 `ipc.lib`(Windows) 静态库文件。

- 单元测试：`/output/bin/ipc-test-correctness`
- 性能测试：在不同终端依次运行 `/output/bin/ipc-test-performance-server` 和 `/output/bin/ipc-test-performance-client`

### 通信方式支持

- ✅ 已实现
- 🔘 未实现
- 🚧 正在实现

<table>
<tr>
<th rowspan="2" align="center" class="vertical-center">通信方式</th>
<th colspan="2" align="center">操作系统</th>
</tr>
<tr>
<th align="center">Windows</th>
<th align="center">Linux</th>
</tr>
<tr>
<td align="center">命名管道</td>
<td align="center">(Windows NamedPipe) ✅</td>
<td align="center">🔘</td>
</tr>
<tr>
<td align="center">消息队列</td>
<td align="center">(Boost Interprocess) ✅</td>
<td align="center">(System V IPC) ✅</td>
</tr>
<tr>
<td align="center">共享内存</td>
<td align="center">🚧</td>
<td align="center">🚧</td>
</tr>
</table>

### 使用方式

#### 引入静态库

对于使用 CMake 构建的项目，可以通过以下方式引入：

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

或者作为第三方库在 CMakeLists.txt 中直接使用：

```cmake
add_subdirectory(ipc EXCLUDE_FROM_ALL)
set(IPC_INCLUDE_DIR "/path/to/ipc/include")
target_link_libraries(your_target ipc)
target_include_directories(your_target PRIVATE ${IPC_INCLUDE_DIR})
```

#### 代码编写

```cpp
#include <ipc/ipc.h>

// 创建两个名为 "Wow" 的 IPC 节点，底层使用命名管道或者消息队列（默认值）
// ipc::node receiver("Wow", ipc::NodeType::kReceiver, ipc::ChannelType::NamedPipe);
// ipc::node receiver("Wow", ipc::NodeType::kReceiver, ipc::ChannelType::MessageQueue);
ipc::node receiver("Wow", ipc::NodeType::kReceiver);
ipc::node sender("Wow", ipc::NodeType::kSender);
auto rec = receiver.Receive();    // 接收消息（会阻塞进程直至接收到消息）
sender.Send(data, sizeof(data));  // 发送消息
```

### 示例（Linux）

在 `examples` 目录下提供了两个示例程序：`sender.cpp` 和 `receiver.cpp`。

运行方式如下：

```bash
cd examples
# 编译
make
# 运行 receiver
make run_receiver
# 在新终端运行 sender
make run_sender
# 此时 receiver 能接收到消息
Received message: Hello, IPC!
```

### 性能

- **Windows**: Intel (R) Core (TM) Ultra 5 225
- **Linux**: Intel (R) Core (TM) Ultra 9 185H

<table>
<tr>
<th rowspan="2" align="center" class="vertical-center">通信延迟 / µs</th>
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
