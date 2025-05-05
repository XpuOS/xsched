import os
import argparse


def cmake(platform):
    return f"""
add_hal_lib({platform})
add_shim_lib({platform})
"""


def cmd_h(platform):
    return f"""
#pragma once

#include "xsched/{platform}/hal.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::{platform}
{{

}}  // namespace xsched::{platform}
"""


def queue_h(platform):
    class_name = f"{platform.capitalize()}Queue"
    return f"""
#pragma once

#include "xsched/types.h"
#include "xsched/{platform}/hal.h"
#include "xsched/preempt/hal/hw_queue.h"

namespace xsched::{platform}
{{

class {class_name} : public preempt::HwQueue
{{
public:
    {class_name}();
    virtual ~{class_name}();
    
    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;

    virtual XDevice       GetDevice()            override;
    virtual HwQueueHandle GetHandle()            override;
    virtual bool          SupportDynamicLevel()  override;
    virtual XPreemptLevel GetMaxSupportedLevel() override;
}};

}}  // namespace xsched::{platform}
"""


def hal_h(platform):
    func_name = f"{platform.capitalize()}QueueCreate"
    return f"""
#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {{
#endif

// create a HwQueue

// create HwCommands

#ifdef __cplusplus
}}
#endif
"""


def shim_h(platform):
    return f"""
#pragma once

#include "xsched/{platform}/hal/driver.h"
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="directory to create")
    parser.add_argument("-p", "--platform", type=str, required=True, help="platform name")
    args = parser.parse_args()

    path = args.directory
    plat = args.platform
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, f"hal/include/xsched/{plat}/hal"), exist_ok=True)
    os.makedirs(os.path.join(path, "hal/src"), exist_ok=True)
    os.makedirs(os.path.join(path, f"shim/include/xsched/{plat}/shim"), exist_ok=True)
    os.makedirs(os.path.join(path, "shim/src"), exist_ok=True)

    # create template files
    # CMakeLists.txt
    with open(os.path.join(path, "CMakeLists.txt"), "w") as f:
        f.write(cmake(plat))

    # hw commands
    with open(os.path.join(path, f"hal/include/xsched/{plat}/hal/{plat}_command.h"), "w") as f:
        f.write(cmd_h(plat))
    with open(os.path.join(path, f"hal/src/{plat}_command.cpp"), "w") as f:
        f.write(f"#include \"xsched/{plat}/hal/{plat}_command.h\"")
    
    # hw queue
    with open(os.path.join(path, f"hal/include/xsched/{plat}/hal/{plat}_queue.h"), "w") as f:
        f.write(queue_h(plat)) 
    with open(os.path.join(path, f"hal/src/{plat}_queue.cpp"), "w") as f:
        f.write(f"#include \"xsched/{plat}/hal/{plat}_queue.h\"")
    
    # hal.h
    with open(os.path.join(path, f"hal/include/xsched/{plat}/hal.h"), "w") as f:
        f.write(hal_h(plat))
    
    # shim.h
    with open(os.path.join(path, f"shim/include/xsched/{plat}/shim/shim.h"), "w") as f:
        f.write(shim_h(plat))
    with open(os.path.join(path, f"shim/src/shim.cpp"), "w") as f:
        f.write(f"#include \"xsched/{plat}/shim/shim.h\"")