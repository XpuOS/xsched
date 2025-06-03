# XSched: Preemptive Scheduling for XPUs



## Usage

```c
#include "xsched/xsched.h"

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

