# Scheduling with Window Activity for Terminal Applications

This example shows how XSched can transparently schedule tasks based on the X11 window activity for terminal applications.
We use the [X11 Monitor](../../service/tools/x11_monitor) and the "[Active Window First](../../sched/src/policy/awf.cpp)" policy mentioned in the [GUI window](../6_gui_window) example.
Further, to support terminal applications, we build an [X11 Launcher](../../service/tools/x11_launcher) to bind the application process ID with the X11 window id of the terminal window.

## Build XSched

Make sure you have already installed X11 development package and built XSched with CUDA support.

```bash
sudo apt update
sudo apt install libx11-dev

# go to the root directory of XSched
cd xsched
# by default, XSched will be installed to xsched/output
# if X11 development package is installed, XSched will automatically build the X11 Monitor and X11 Launcher
make cuda INSTALL_PATH=<install_path>
```

## Build the Example

```bash
cd xsched/examples/7_terminal_window
make # cuda runtime and NVCC are required
```

## Run the Example

**Make sure that you are using X11 window system.** For example, you can select "Ubuntu on Xorg" as the window system in the Ubuntu login screen.

First, we should start the xserver with Active Window First (AWF) policy.

```bash
# open a new terminal
<install_path>/bin/xserver AWF 50000
```

Second, start the X11 Monitor to monitor the window activity.

```bash
# open a new terminal
<install_path>/bin/x11_monitor
```

Next, we can run the two apps with the X11 Launcher.

```bash
# open a new terminal
# same environment variables as 1_transparent_sched example
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=8
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=4
export LD_LIBRARY_PATH=<install_path>/lib:$LD_LIBRARY_PATH

# launch the app with X11 Launcher
<install_path>/bin/x11_launcher ./app
```

In another terminal, we run the app again.

```bash
# open a new terminal
# same environment variables as 1_transparent_sched example
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=8
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=4
export LD_LIBRARY_PATH=<install_path>/lib:$LD_LIBRARY_PATH

# launch the app with X11 Launcher
<install_path>/bin/x11_launcher ./app
```

## Demo

[Demo](https://github.com/user-attachments/assets/a0a23e9d-435e-40ad-a86b-5662f21d563b)

By clicking the terminal windows of the two apps, the window focus will be changed, and you will see that the latencies of the tasks in the active (or focused) window will be lower than the other (background) window.
