# Transparent Scheduling with XSched

This example shows how XSched can transparently schedule tasks.

## Requirements

- An NVIDIA GPU.
- CUDA runtime with NVCC.

## Build XSched

Make sure you have already built XSched with CUDA support.

```bash
cd xsched # go to the root directory of XSched
make cuda # by default, XSched will be installed to xsched/output
```

## Build the Example

```bash
cd xsched/examples/1_transparency
make # cuda runtime and NVCC are required
```

## Run the Example

This example is a simple vector addition program, but running many times.

```bash
./app
```

You will see the output like this:

```txt
Task 0 completed in 66 ms
Task 1 completed in 66 ms
Task 2 completed in 66 ms
Task 3 completed in 66 ms
Task 4 completed in 66 ms
```

Now, you can open a new terminal and run two apps simultaneously.

```bash
# In the first terminal
./app

# In the second terminal
./app
```

You will see the output like this:

```txt
# In the first terminal
Task 0 completed in 114 ms
Task 1 completed in 134 ms
Task 2 completed in 143 ms
Task 3 completed in 145 ms
Task 4 completed in 132 ms
```

```txt
# In the second terminal
Task 13 completed in 78 ms
Task 14 completed in 115 ms
Task 15 completed in 133 ms
Task 16 completed in 144 ms
Task 17 completed in 145 ms
Task 18 completed in 131 ms
```

The two apps have the same performance (double the time as the single app) as the GPU hardware schedules them fairly.

## Run with XSched

Now, let's use XSched to prioritize one of the apps.

First, we should start the XSched server (xserver). It is a daemon process for scheduling the GPU processes.

```bash
# Open a new terminal
cd xsched/output/bin/xserver HPF 50000
# HPF: Highest Priority First
# 50000: server listening port, which is used to connect with XCli, our command line tool for XSched
```

Next, we can run the two apps with XSched. XSched is transparent to the application because it uses a shim library to intercept the CUDA calls and convert them into XQueue operations. Users can set the configuration of the shim library through environment variables.

In the first terminal, we run the app and set it to high priority.

```bash
# the process will be scheduled according to the global (GLB) scheduler, i.e., the xserver
export XSCHED_SCHEDULER=GLB

# automatically create an XQueue for each created HwQueue (in this case, CUDA stream)
export XSCHED_AUTO_XQUEUE=ON

# automatically set the priority of the created XQueue to 1 (bigger is higher priority)
export XSCHED_AUTO_XQUEUE_PRIORITY=1

# automatically enable Level-1 preemption for the created XQueue
# higher levels can achieve faster preemption but needs corresponding implementation
export XSCHED_AUTO_XQUEUE_LEVEL=1

# automatically set the threshold of the created XQueue to 16 (default is 16)
# threshold is the number of in-flight commands in the XQueue, >= 1
# smaller threshold leads to faster preemption but higher execution overhead
export XSCHED_AUTO_XQUEUE_THRESHOLD=16

# automatically set the command batch size of the created XQueue to 8 (default is 8)
# batch size is the number of commands that XSched will launch at a time,
# >= 1 && <= threshold, recommended to be half of the threshold
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8

# intercept the CUDA calls using the shim library (libcuda.so -> libshimcuda.so)
# for cuda, libshimcuda.so implements all the symbols in libcuda.so, and we set
# LD_LIBRARY_PATH to the path of the XSched library to intercept the CUDA calls
# for other platforms like opencl, we may use LD_PRELOAD to intercept the calls
# replace <XSCHED_ROOT> with the path of the XSched root directory
export LD_LIBRARY_PATH=<XSCHED_ROOT>/output/lib:$LD_LIBRARY_PATH

# run the app
./app
```

In the second terminal, we run the app and set it to low priority.

```bash
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_PRIORITY=0
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=4
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=2
# replace <XSCHED_ROOT> with the path of the XSched root directory
export LD_LIBRARY_PATH=<XSCHED_ROOT>/output/lib:$LD_LIBRARY_PATH
./app
```

You will see the output like this:

```txt
# In the first terminal
Task 25 completed in 67 ms
Task 26 completed in 67 ms
Task 27 completed in 67 ms
Task 28 completed in 67 ms
Task 29 completed in 69 ms
Task 30 completed in 67 ms
```

```txt
# In the second terminal
Task 1 completed in 207 ms
Task 2 completed in 195 ms
Task 3 completed in 165 ms
Task 4 completed in 184 ms
Task 5 completed in 188 ms
Task 6 completed in 189 ms
Task 7 completed in 172 ms
```

The high priority app achieves a similar performance as the standalone execution, while the low priority app is slowed down.

## XCLI

You can also use `xcli` to check the status of the XQueues and manage them.

```bash
# Open a new terminal
cd xsched/output/bin

# show help
./xcli -h

# show all XQueues, ctrl-c to stop
# -f: refresh frequency
./xcli top -f 10

# give a hint to set the priority of XQueue 0xaf246296bbdf3260 to 2
./xcli hint -x 0xaf246296bbdf3260 -p 2
```
