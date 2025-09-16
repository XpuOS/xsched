# Transparent Scheduling with XSched

This example shows how XSched can transparently schedule tasks.

## Requirements

- An NVIDIA GPU.
- CUDA runtime with NVCC.

## Build XSched

Make sure you have already built XSched with CUDA support.

```psl
# go to the root directory of XSched
cd xsched
# by default, XSched will be installed to xsched/output
make cuda INSTALL_PATH=<install_path>
```

## Build the Example

```psl
cd xsched\examples\1_transparency
make # cuda runtime and NVCC are required
```

## Run the Example

This example is a simple vector addition program, but running many times.

```psl
.\app.exe
```

You will see the output like this:

```psl
Task 0 completed in 73 ms
Task 1 completed in 73 ms
Task 2 completed in 74 ms
Task 3 completed in 73 ms
Task 4 completed in 73 ms
```

Now, you can open a new terminal and run two apps simultaneously.

```psl
# In the first terminal
.\app.exe

# In the second terminal
.\app.exe
```

You will see the output like this:

```psl
# In the first terminal
Task 0 completed in 225 ms
Task 1 completed in 225 ms
Task 2 completed in 225 ms
Task 3 completed in 227 ms
Task 4 completed in 224 ms
```

```psl
# In the second terminal
Task 33 completed in 74 ms
Task 34 completed in 91 ms
Task 35 completed in 227 ms
Task 36 completed in 226 ms
Task 37 completed in 227 ms
Task 38 completed in 227 ms
```

The two apps have the same performance (2 ~ 3 times the time as the single app) as the GPU hardware schedules them fairly.

## Run with XSched

Now, let's use XSched to prioritize one of the apps.

First, refer to the [API Interception](../../README.md) section to set up the API interception.

Secondly, we start the XSched server (xserver). It is a daemon process for scheduling the GPU processes.

```psl
# Open a new terminal
<install_path>\bin\xserver.exe HPF 50000
# HPF: Highest Priority First
# 50000: server listening port, which is used to connect with XCli, our command line tool for XSched
```

Next, we can run the two apps with XSched. XSched is transparent to the application because it uses a shim library to intercept the CUDA calls and convert them into XQueue operations. Users can set the configuration of the shim library through environment variables.

In the first terminal, we run the app and set it to high priority.

```psl
# the process will be scheduled according to the global (GLB) scheduler, i.e., the xserver
$env:XSCHED_SCHEDULER = "GLB"

# automatically create an XQueue for each created HwQueue (in this case, CUDA stream)
$env:XSCHED_AUTO_XQUEUE = "ON"

# automatically set the priority of the created XQueue to 1 (bigger is higher priority)
$env:XSCHED_AUTO_XQUEUE_PRIORITY = "1"

# automatically enable Level-1 preemption for the created XQueue
# higher levels can achieve faster preemption but needs corresponding implementation
$env:XSCHED_AUTO_XQUEUE_LEVEL = "1"

# automatically set the threshold of the created XQueue to 16 (default is 16)
# threshold is the number of in-flight commands in the XQueue, >= 1
# smaller threshold leads to faster preemption but higher execution overhead
$env:XSCHED_AUTO_XQUEUE_THRESHOLD = "16"

# automatically set the command batch size of the created XQueue to 8 (default is 8)
# batch size is the number of commands that XSched will launch at a time,
# >= 1 && <= threshold, recommended to be half of the threshold
$env:XSCHED_AUTO_XQUEUE_BATCH_SIZE = "8"

# Intercept the CUDA calls using the shim library (merge into nvcuda.dll).
# For cuda, nvcuda.dll compiled by XSched implements all the symbols in nvcuda.dll.
# We rename `nvcuda.dll` in `C:\Windows\System32` to `nvcuda_original.dll`, 
# and copy `nvcuda.dll` in `output\bin` to `C:\Windows\System32`,
# Then we can intercept the CUDA calls.

# run the app
.\app.exe
```

In the second terminal, we run the app and set it to low priority.

```psl
$env:XSCHED_SCHEDULER = "GLB"
$env:XSCHED_AUTO_XQUEUE = "ON"
$env:XSCHED_AUTO_XQUEUE_PRIORITY = "0"
$env:XSCHED_AUTO_XQUEUE_LEVEL = "1"
$env:XSCHED_AUTO_XQUEUE_THRESHOLD = "4"
$env:XSCHED_AUTO_XQUEUE_BATCH_SIZE = "2"
.\app.exe
```

You will see the output like this:

```psl
# In the first terminal
Task 37 completed in 79 ms
Task 38 completed in 74 ms
Task 39 completed in 74 ms
Task 40 completed in 75 ms
Task 41 completed in 77 ms
Task 42 completed in 74 ms
Task 43 completed in 74 ms
Task 44 completed in 74 ms
Task 45 completed in 78 ms
```

```psl
# In the second terminal
Task 0 completed in 117 ms
Task 1 completed in 82 ms
Task 2 completed in 159 ms
Task 3 completed in 228 ms
Task 4 completed in 223 ms
Task 5 completed in 147 ms
```

The high priority app achieves a similar performance as the standalone execution, while the low priority app is slowed down.

## XCLI

You can also use `xcli` to check the status of the XQueues and manage them.

```psl
# Open a new terminal
cd <install_path>\bin

# show help
.\xcli.exe -h

# show all XQueues, ctrl-c to stop
# -f: refresh frequency
.\xcli.exe top -f 10

# give a hint to set the priority of XQueue 0xaf246296bbdf3260 to 2
.\xcli.exe hint -x 0xaf246296bbdf3260 -p 2
```
