# XSched Examples

| No. | Name                | Description                                                                                                       | Platforms |
| --- | ------------------- | ----------------------------------------------------------------------------------------------------------------- | --------- |
| 1   | transparent_sched   | Two simple vector addition programs that run many times, demonstrating the transparent scheduling of tasks.       | CUDA HIP  |
| 2   | give_hints          | An example of showing how to create XQueue and give hints to XSched using XQueue APIs & Hint APIs                 | CUDA HIP  |
| 3   | intra_process_sched | Using XQueue APIs & Hint & LocalScheduler for scheduling within a process                                         | CUDA HIP  |
| 4   | manual_sched        | Manually scheduling within a process using XQueue APIs including Resume & Suspend                                 | CUDA HIP  |
| 5   | infer_serving       | Integrate XSched into Triton Inference Server to enable priority-based scheduling of multi-model inference tasks. | CUDA      |
| 6   | gui_window          | Scheduling tasks based on the X11 window activity for GUI applications.                                           | LevelZero |
| 7   | terminal_window     | Scheduling tasks based on the X11 window activity for terminal applications.                                      | CUDA HIP  |
