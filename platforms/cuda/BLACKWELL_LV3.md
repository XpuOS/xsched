# Experimental Blackwell Level 3: CHANNEL

This backend adds state-preserving compute preemption on the validated SM120
configuration below. Select it explicitly with `XSCHED_CUDA_LV3_IMPL=CHANNEL`
and preload `libcuda_channel_tracker.so` before CUDA initialization. The default
backend selection is unchanged. This change does not add SM120 Guardian Level 2.

## Mechanism and scope

The backend queries private ToolsRm context handles, locates the owning compute
group in the process's tracked RM allocations, and disables all compute channels
in that group with `NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS`. The request uses
synchronous preemption, `bOnlyDisableScheduling=false`, `bRewindGpPut=false`, and
no event. Resume re-enables the original channel snapshot. XSched does not
instrument kernels or replay completed instructions on this path.

The control domain is a compute group, not an individual CUDA stream. Only one
managed queue may own a context/group. Other compute streams in that group also
stop. Use separate CUDA contexts for independently controlled tasks, and account
for any cross-context event/data dependencies when scheduling them.

The optional preload library observes this process's successful RM allocations
and frees. It removes `RTLD_DEEPBIND` only from libcuda loads so ioctl calls can
be intercepted. It does not enumerate or control other processes' objects.

## Validated configuration

- Linux x86_64; NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU, SM120.
- NVIDIA driver **595.71.05**, CUDA Toolkit **13.0**, native `sm_120` kernels.
- Explicit application-managed queues; one managed owner per compute group.

Queue creation requires the v2 tracker symbol and the exact driver version from
`/proc/driver/nvidia/version`. An unavailable tracker, unsupported driver or GPU,
unrecognized handle, or duplicate owner returns an error without registering a
queue or silently falling back to Level 1. Other driver versions require ABI
review and hardware validation before changing the version check.

## Queue contract

- `CudaQueueGetChannelInfo` reports context ownership, compute-group scope,
  preserved in-flight work, independent-copy behavior and lack of drop support.
- `XQueueSuspend(..., kQueueSuspendFlagSyncHwQueue)` confirms compute suspension,
  not command completion or complete context quiescence. Independent copy-engine
  transfers may continue. Do not free buffers based on suspension alone.
- Ordinary `HwQueueSynchronize` retains completion semantics and may block until
  another host thread resumes the queue.
- An unmanaged stream waiting for an event recorded by a CHANNEL queue waits for
  record submission before inserting the native CUDA dependency. Record
  submission can itself wait if the source queue is paused.
- `kQueueResumeFlagDropCommands` returns `kXSchedErrorNotSupported` before changing
  queue state, including when a CHANNEL queue has been switched to Level 1.
- Destroy XQueue, then HwQueue, then stream/context. Destroying a paused CHANNEL
  XQueue resumes and drains its retained work. Ownership can then be reacquired.
- The shim rejects destruction of an owned context, and conservatively rejects
  primary-context reset while any CHANNEL queue owns a context on that device.
  The tracker rejects freeing owned groups/ancestors and allocation/free within
  paused or uncertain groups.
- Level 2 on this backend uses the same stronger pause operation. Level 1 uses
  ordinary blocking. Change levels only on a resumed queue with no concurrent
  queue operations. Suspend/resume controls on an AsyncXQueue are serialized.

A failed batch disable may have affected some channels. The tracker retains the
snapshot, attempts to enable the entire set, and never treats a successful
rollback as successful suspension. Enable and rollback use up to three attempts;
the HAL may retry pause up to three times. An uncertain state prevents lifecycle
changes until restoration succeeds. Persistent runtime failures terminate with
a diagnostic because the HAL control methods return `void`. This is fail-stop
handling, not a recoverable-error API.

## Limitations

- This is not transparent independent scheduling of multiple streams sharing a
  context. Automatic per-stream queue creation conflicts with group ownership;
  the existing stream-create shim does not propagate `AutoCreate` failures to
  the CUDA caller. An unmanaged stream is not automatically included in the
  owner's queue-idle detection or service accounting. Use explicit APP-mode
  ownership and controlled task layouts until group-aware integration exists.
- Do not bypass the shim, release the last primary-context reference while a
  queue exists, or race context lifetime operations with queue creation/destruction.
  Green Context lifetime is not covered by these guards.
- CUDA graphs, cooperative/cluster/device launches, external semaphores, fork
  after CUDA initialization, other GPUs/drivers, and high-contention multi-context
  scaling have not been validated by this change.
- Synchronous RM calls carry driver/firmware/hardware costs. The backend does not
  promise a latency bound or a performance improvement for short tasks/time slices.
- Adding HAL virtual methods changes the internal C++ ABI. Rebuild preempt, HAL,
  shim and any out-of-tree HAL subclasses together.

## Build and test

From the repository root, using an initialized recursive checkout:

```bash
cmake -S . -B build-channel -DPLATFORM_CUDA=ON -DSHIM_SOFTLINK=ON \
  -DBUILD_TEST=ON -DBUILD_SERVICE=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_INSTALL_PREFIX="$PWD/output-channel"
cmake --build build-channel -j8
cmake --install build-channel
ctest --test-dir build-channel -R cuda_channel_tracker_test --output-on-failure
python3 platforms/cuda/test/scripts/run_lv3_contracts.py \
  --install output-channel --build build-channel \
  --output /tmp/xsched-channel-contracts-new --repeats 3
```

The output directory must be new. The suite runs GPU tests serially with timeouts
and saves metadata, binary hashes, stdout/stderr and a summary. It controls only
the test process's channels and requires no root privileges or driver changes.

The CPU test substitutes the syscall in the production tracker and covers
ownership, binding, lifecycle guards, partial disable, rollback, retry, uncertain
state recovery and overflow. It does not inject actual driver/firmware failures.

`sm120_lv3` checks real shim submission, concurrent streams, 3D grids/blocks,
repeated suspension of unfinished kernels, exact non-idempotent counts and
thread-local arithmetic, queued kernel/copy ordering, Level-1 switching and queue
recreation. Functions are preloaded to avoid lazy loading invalidating the
concurrent-work setup. `sm120_channel_contract` checks independent contexts,
copy/event dependencies, completion synchronization, unsupported-operation
rejection, paused destruction, reuse and concurrent host controls on an idle
queue. GPU tests exit normally and exercise cleanup; they do not use `_Exit`.

## Implementation map

- `hal/src/level3/channel.cpp`: HAL ownership, ToolsRm lookup and pause/resume.
- `hal/include/xsched/cuda/hal/level3/channel_control.h`: internal tracker ABI.
- `tools/channel_tracker.cpp`: RM allocation tracking, disable/enable and recovery.
- `preempt/src/xqueue/async_xqueue.cpp`: control serialization and suspension boundary.
- `shim/src/shim.cpp`: event dependencies and context-lifetime guards.
- `test/scripts/run_lv3_contracts.py`: repeatable correctness suite.

RM layouts are derived from NVIDIA's matching open driver definitions:
[control header](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.71.05/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fifo.h),
[allocation/control ABI](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.71.05/src/common/sdk/nvidia/inc/nvos.h).
These definitions do not make the private ToolsRm interface a stable CUDA API.
