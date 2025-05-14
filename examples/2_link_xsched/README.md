# Schedule by Linking XSched

*Refer to [xqueue.h](../../include/xsched/xqueue.h) for API list and more details.*

## C/C++ API

### Code Example

```c
... // Other includes 
#include "xsched/xsched.h"

... // Other code
// LevelZero
zeCommandQueueCreate(hContext, hDevice, &cmdQueueDesc, &hwQueue);
// CUDA
cudaStreamCreate(&hwQueue, flags);
... // Other code

// Create XQueue
XQueueHandle hXQueue;
XQueueCreate(hXQueue, hCommandQueue, 1, 0);

// Using XSched API to Control XQueue
XQueueSetPreemptLevel(hXQueue, 2);
XQueueWaitAll(hXQueue);
...
```

### Link XSched

```cmake
# use path hints to find XSched CMake
# or use cmake -DCMAKE_PREFIX_PATH=<install_path>/lib/cmake instead
find_package(XSched REQUIRED HINTS "<install_path>/lib/cmake")
... # add your target
target_link_libraries(<your_target> XSched::preempt XSched::halcuda ...)
```

## Python API

### Using Example

```python
from xsched import XSched

...
# Create XQueue
hXQueue = XSched.XQueueCreate(hwQueue, 1, 0)

# Using XSched API to Control XQueue
XSched.XQueueSetPreemptLevel(hXQueue, 2)
XSched.XQueueWaitAll(hXQueue)
...
```
