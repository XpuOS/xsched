# XSched Examples

| No. | Name                | Description                                                                                                       | Platforms |
| --- | ------------------- | ----------------------------------------------------------------------------------------------------------------- | --------- |
| 1   | transparent schedule| Two simple vector addition programs that run many times, demonstrating the transparent scheduling of tasks.       | CUDA, HIP |
| 2   | give hints          | An example of showing how to create XQueue and give hints to XSched using XQueue APIs & Hint APIs                 | CUDA, HIP |
| 3   | intra-process schedule | Using XQueue APIs & Hint & LocalScheduler for scheduling within a process                                      | CUDA, HIP |
| 4   | manual schedule     | Manually scheduling within a process using XQueue APIs including Resume & Suspend                                 | CUDA, HIP |
| 5   | inference serving   | Integrate XSched into the [TensorRT backend of NVIDIA Triton Inference Server](https://github.com/triton-inference-server/tensorrt_backend/tree/r22.06) to enable priority-based scheduling of multi-model inference tasks. | CUDA      |
| 6   | gui window          | Scheduling tasks based on the X11 window activity for GUI applications.                                           | LevelZero |
| 7   | terminal window     | Scheduling tasks based on the X11 window activity for terminal applications.                                      | CUDA, HIP |
| 8   | llama.cpp with XSched | Integrate XSched into Llama.cpp Inference Server to enable priority-based scheduling of inference tasks. | CUDA |
