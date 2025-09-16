# Transparent Scheduling with XSched

This example shows how XSched can transparently schedule tasks.

## Requirements

- Intel® Core™ Ultra Processors with Intel® Arc™ Graphics (Meteor Lake) or Intel® Arc™ A-Series Graphics, or Intel GPUs newer than them.
- Intel® Arc™ & Iris® Xe Graphics Driver - Windows.
- Python environment with `torch`, `intel_extension_for_pytorch`, and `torchvision` installed.

Refer to the [Intel® Arc™ & Iris® Xe Graphics - Windows](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html) and [Intel® Extension for PyTorch* Installation Guide](https://pytorch-extension.intel.com/installation?platform=gpu) for the installation guide.

## Build XSched

Make sure you have already built XSched with LevelZero support.

```psl
# go to the root directory of XSched
cd xsched
# by default, XSched will be installed to xsched/output
make levelzero INSTALL_PATH=<install_path>
```

## Build the Environment

```psl
python -m venv xsched
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```

Or you can use the provided `requirements.txt` file.

```psl
python -m venv xsched
pip install -r requirements.txt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```

## Run the Example

This example is a simple resnet152 inference app running on Intel GPU.

```psl
python .\app.py
```

You will see the output like this:

```psl
Task 0 completed in 0.067143 seconds
Task 1 completed in 0.067355 seconds
Task 2 completed in 0.067996 seconds
Task 3 completed in 0.067269 seconds
Task 4 completed in 0.067042 seconds
Task 5 completed in 0.067216 seconds
```

Now, you can open a new terminal and run two apps simultaneously.

```psl
# In the first terminal
python .\app.py

# In the second terminal
python .\app.py
```

You will see the output like this:

```psl
# In the first terminal
Task 0 completed in 0.131807 seconds
Task 1 completed in 0.142609 seconds
Task 2 completed in 0.136347 seconds
Task 3 completed in 0.129500 seconds
Task 4 completed in 0.126840 seconds
```

```psl
# In the second terminal
Task 0 completed in 0.135713 seconds
Task 1 completed in 0.131230 seconds
Task 2 completed in 0.144202 seconds
Task 3 completed in 0.159661 seconds
Task 4 completed in 0.138011 seconds
```

The two apps have the same performance (2 times the time as the single app) as the GPU hardware schedules them fairly.

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

Next, we can run the two apps with XSched. XSched is transparent to the application because it uses a shim library to intercept the LevelZero calls and convert them into XQueue operations. Users can set the configuration of the shim library through environment variables.

In the first terminal, we run the app and set it to high priority.

```psl
# the process will be scheduled according to the global (GLB) scheduler, i.e., the xserver
$env:XSCHED_SCHEDULER = "GLB"

# automatically create an XQueue for each created HwQueue
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

# run the app
$env:XSCHED_SCHEDULER = "GLB";$env:XSCHED_AUTO_XQUEUE = "ON";$env:XSCHED_AUTO_XQUEUE_PRIORITY = "1";$env:XSCHED_AUTO_XQUEUE_LEVEL = "1";$env:XSCHED_AUTO_XQUEUE_THRESHOLD = "16";$env:XSCHED_AUTO_XQUEUE_BATCH_SIZE = "8"
python .\app.py
```

In the second terminal, we run the app and set it to low priority.

```psl
$env:XSCHED_SCHEDULER = "GLB";$env:XSCHED_AUTO_XQUEUE = "ON";$env:XSCHED_AUTO_XQUEUE_PRIORITY = "0";$env:XSCHED_AUTO_XQUEUE_LEVEL = "1";$env:XSCHED_AUTO_XQUEUE_THRESHOLD = "4";$env:XSCHED_AUTO_XQUEUE_BATCH_SIZE = "2"
python .\app.py
```

You will see the output like this:

```psl
# In the first terminal
Task 0 completed in 0.092000 seconds
Task 1 completed in 0.086002 seconds
Task 2 completed in 0.092999 seconds
Task 3 completed in 0.091999 seconds
Task 4 completed in 0.085000 seconds
Task 5 completed in 0.107000 seconds
Task 6 completed in 0.095001 seconds
Task 7 completed in 0.083000 seconds
Task 8 completed in 0.108998 seconds
```

```psl
# In the second terminal
Task 83 completed in 0.066999 seconds
Task 84 completed in 0.114000 seconds
Task 85 completed in 0.068001 seconds
Task 86 completed in 0.079001 seconds
Task 87 completed in 0.168000 seconds
Task 88 completed in 0.278999 seconds
Task 89 completed in 0.270000 seconds
Task 90 completed in 0.255000 seconds
Task 91 completed in 0.280000 seconds
Task 92 completed in 0.288003 seconds
Task 93 completed in 0.288997 seconds
Task 94 completed in 0.286000 seconds
```

The runtime of high priority applications is reduced by 30% compared to not scheduled, while the low priority app is slowed down.

## Slice Mode

For LevelZero platform, we provide CommandList Slicing to further reduce the preemption latency of high priority tasks. To activate this feature, simply set the environment variabl:

```psl
$env:XSCHED_LEVELZERO_SLICE_CNT = "4"
```

Then use xsched, you will see the output like this:

```psl
# In the first terminal
Task 0 completed in 0.080000 seconds
Task 1 completed in 0.078010 seconds
Task 2 completed in 0.079990 seconds
Task 3 completed in 0.077999 seconds
Task 4 completed in 0.078008 seconds
Task 5 completed in 0.080003 seconds
Task 6 completed in 0.074990 seconds
Task 7 completed in 0.083010 seconds
Task 8 completed in 0.077990 seconds
```

```psl
# In the second terminal
Task 83 completed in 0.082000 seconds
Task 84 completed in 0.077006 seconds
Task 85 completed in 0.117993 seconds
Task 86 completed in 0.075999 seconds
Task 87 completed in 0.076008 seconds
Task 88 completed in 1.240998 seconds
Task 89 completed in 3.023995 seconds
Task 90 completed in 2.871410 seconds
```

The smaller the `XSCHED_LEVELZERO_SLICE_CNT`, the lower the latency of high priority tasks, and the corresponding cost is that the latency of low priority tasks is higher.

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
