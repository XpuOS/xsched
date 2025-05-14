# How to Support XSched on a New Platform

## Step 1: Implement interception code

We support automatical generation tools, developers can follow the steps below (taking `opencl` as an example) to use them.

### 1. Setup directory structure

```shell
python3 tools/autogen/setup_template.py ./platforms/example --platform opencl
```

- `--platform`: the name of the platform, which will be used as the directory name.

### 2. Gather platform headers

For single-header platform, just copy the header to `platforms/example/hal/include/xsched/opencl/hal`
For platform with multiple headers, use `tools/autogen/merge_headers.py` to merge them into a single header.

```shell
python3 tools/autogen/merge_headers.py /usr/include/CL/ \
    -o platforms/example/hal/include/xsched/opencl/hal/cl.h \
    -e *.hpp -e cl_dx* -e cl_d3d* -e cl_icd.h -e cl_layer.h -e cl_va_api_media_sharing_intel.h
```

- `-o(--output)`: Output file path (default is "./merged.h")
- `-e(--exclude)`: Pattern to exclude header files (can be used multiple times, supports glob patterns like *.hpp)
- `-I(--include-dir)`: Additional include directories to search for headers (can be used multiple times, e.g., `-I /usr/include`)

### 3. Generate interception code (`driver.h` & `intercept.cpp`)

```shell
python3 tools/autogen/gen.py \
    --source platforms/example/hal/include/xsched/opencl/hal/cl.h \
    --platform opencl \
    --prefix cl \
    --lib /usr/lib/x86_64-linux-gnu/libOpenCL.so \
    --driver platforms/example/hal/include/xsched/opencl/hal/driver.h \
    --intercept platforms/example/shim/src/intercept.cpp
```

- `-s(--source)`: Path to the header source
- `-I(--include)`: Path to the additional include directory
- `--platform`: Platform name
- `--prefix`: Prefix for the function names
- `--lib`: Driver library file
- `--driver`: Output driver header file
- `--intercept`: Output intercept source file

## Step 2: Define HwQueue & HwCommand abstraction

For `HwQueue`, we have implemented its parent class - `preempt::HwQueue`(refer to [hw_queue.h](../../preempt/include/xsched/preempt/hal/hw_queue.h)). What you need is to inherit this parent class, and finish interfaces in the table below.

<table>
  <tr>
    <th align="center">Preemption Level</th>
    <th align="center">Interface</th>
    <th align="center">Description</th>
  </tr>
  <tr>
    <td align="center" rowspan="6">Level-1</td>
    <td align="center">GetDevice()</td>
    <td align="left">Get device type of HwQueue</td>
  </tr>
  <tr>
    <td align="center">GetHandle()</td>
    <td align="left">Get HwQueue handle</td>
  </tr>
  <tr>
    <td align="center">SupportDynamicLevel()</td>
    <td align="left">Whether the platform supports multiple preemption levels</td>
  </tr>
  <tr>
    <td align="center">GetMaxSupportedLevel()</td>
    <td align="left">The max supported preemption level</td>
  </tr>
  <tr>
    <td align="center">Launch(HwCommand)</td>
    <td align="left">Launch a HwCommand by calling HwCommand->Enqueue()</td>
  </tr>
  <tr>
    <td align="center">Synchronize()</td>
    <td align="left">Wait for all commands in the HwQueue to complete</td>
  </tr>
    <tr>
    <td align="center" rowspan="2">Level-2</td>
    <td align="center">Deactivate()</td>
    <td align="left">Deactivate the HwQueue to prevent all its commands from being selected for execution </td>
  </tr>
  <tr>
    <td align="center">Reactivate()</td>
    <td align="left">Reactivate the HwQueue to allow all its commands to be selected for execution</td>
  </tr>
    <tr>
    <td align="center" rowspan="2">Level-3</td>
    <td align="center">Interrupt()</td>
    <td align="left">Interrupt the running command of the HwQueue</td>
  </tr>
  <tr>
    <td align="center">Restore()</td>
    <td align="left">Restore the interrupted command of the HwQueue</td>
  </tr>
</table>

Note that only the Level-1 interfaces are mandatory for supporting a new XPU, while the Level-2 and Level-3 interfaces are optional since they necessitate additional hardware capabilities.

For `HwCommand`, we have also implemented its parent class - `preempt::HwCommand`. The meaning of interfaces have been explain in detail in [hw_command.h](../../preempt/include/xsched/preempt/hal/hw_command.h), you can implement these interfaces by your need.

## Step 3: Finish XShim Lib

Finish functions which need extral handling in `platforms/example/shim/src/xshim.cpp`, such as `XCreateCommandQueueWithProperties()`, `XCreateKernel()`, `XEnqueueNativeKernel()`, etc. The main objectives of these additional treatments are as follows:

1. Create `XQueue` synchronously when creating hardware queue.
2. Encapsulate the launch kernel as `HwCommand` and submit it to `XQueue`.
3. Ensure correctness by extra kernel arguments lifecycle management and synchronization handling.

Then change function calling path in `platforms/example/shim/src/intercept.cpp` to call `XShim` functions instead of original driver API. For example, change `Driver::CreateKernel` to `XCreateKernel()`.
