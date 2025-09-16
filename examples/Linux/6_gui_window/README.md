# Scheduling with Window Activity for GUI Applications

This example shows how XSched can transparently schedule tasks based on the X11 window activity for GUI applications.
We build an [X11 Monitor](../../service/tools/x11_monitor) to monitor the window activity and report the active (or focused) window id to XSched.
We also implement an "[Active Window First](../../sched/src/policy/awf.cpp)" policy to prioritize the tasks in the active window.
