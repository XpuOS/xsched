#pragma once

#include "xsched/types.h"

#define XSCHED_DEFAULT_PREEMPT_LEVEL              kPreemptLevelBlock
#define XSCHED_DEFAULT_COMMAND_THRESHOLD          16
#define XSCHED_DEFAULT_COMMAND_BATCH_SZIE         8

#define XSCHED_AUTO_XQUEUE_ENV_NAME               "XSCHED_AUTO_XQUEUE"             // = str[ON/OFF]      , default = OFF
#define XSCHED_AUTO_XQUEUE_LEVEL_ENV_NAME         "XSCHED_AUTO_XQUEUE_LEVEL"       // = int[1, 3]        , default = 1
#define XSCHED_AUTO_XQUEUE_THRESHOLD_ENV_NAME     "XSCHED_AUTO_XQUEUE_THRESHOLD"   // = int[1, max_int64], default = 16
#define XSCHED_AUTO_XQUEUE_BATCH_SIZE_ENV_NAME    "XSCHED_AUTO_XQUEUE_BATCH_SIZE"  // = int[1, threshold], default = 8
#define XSCHED_AUTO_XQUEUE_PRIORITY_ENV_NAME      "XSCHED_AUTO_XQUEUE_PRIORITY"    // = int[-256, 255]   , default = 0
#define XSCHED_AUTO_XQUEUE_UTILIZATION_ENV_NAME   "XSCHED_AUTO_XQUEUE_UTILIZATION" // = int[0, 100]
#define XSCHED_AUTO_XQUEUE_TIMESLICE_ENV_NAME     "XSCHED_AUTO_XQUEUE_TIMESLICE"   // = int[100, 100000]
#define XSCHED_AUTO_XQUEUE_LAXITY_ENV_NAME        "XSCHED_AUTO_XQUEUE_LAXITY"      // = int

#define XSCHED_ASCEND_LIB_ENV_NAME     "XSCHED_ASCEND_LIB"
#define XSCHED_CUDA_LIB_ENV_NAME       "XSCHED_CUDA_LIB"
#define XSCHED_CUDART_LIB_ENV_NAME     "XSCHED_CUDART_LIB"
#define XSCHED_CUDLA_LIB_ENV_NAME      "XSCHED_CUDLA_LIB"
#define XSCHED_HIP_LIB_ENV_NAME        "XSCHED_HIP_LIB"
#define XSCHED_LEVELZERO_LIB_ENV_NAME  "XSCHED_LEVELZERO_LIB"
#define XSCHED_OPENCL_LIB_ENV_NAME     "XSCHED_OPENCL_LIB"
#define XSCHED_VPI_LIB_ENV_NAME        "XSCHED_VPI_LIB"
// NEW_PLATFORM: New platform lib env names go here.

#define XSCHED_LEVELZERO_SLICE_CNT_ENV_NAME       "XSCHED_LEVELZERO_SLICE_CNT"

#define XSCHED_SERVER_DEFAULT_PORT   50000
#define XSCHED_SERVER_CHANNEL_NAME   "xsched-server"
#define XSCHED_CLIENT_CHANNEL_PREFIX "xsched-client-"

#define XSCHED_POLICY_ENV_NAME  "XSCHED_POLICY" // e.g., export XSCHED_POLICY=HPF
#define XSCHED_POLICY_NAME_GBL  "GBL" // Global Scheduler
#define XSCHED_POLICY_NAME_AMG  "AMG" // Application Managed
#define XSCHED_POLICY_NAME_HPF  "HPF" // Highest Priority First
#define XSCHED_POLICY_NAME_RR   "RR"  // Round Robin
#define XSCHED_POLICY_NAME_UP   "UP"  // Utilization Partition
#define XSCHED_POLICY_NAME_PUP  "PUP" // Process Utilization Partition
#define XSCHED_POLICY_NAME_EDF  "EDF" // Earliest Deadline First
#define XSCHED_POLICY_NAME_LAX  "LAX" // Laxity-based
#define XSCHED_POLICY_NAME_KEDF "KEDF" // K-Earliest Deadline First
// NEW_POLICY: New policy type names go here.
