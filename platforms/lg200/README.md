# Loongson LG200 device discovery

This module discovers LG200 DRM devices owned by the `loonggpu` kernel driver
and reads optional utilization and local-memory attributes exported through
sysfs.

This change intentionally does **not** add an LG200 `HwQueue` or `HwCommand`.
Generic application callbacks can already use XSched's
`HwCommandCreateCallback` interface. Hardware scheduling and queue preemption
belong in a separate adapter backed by LoongGPU driver capabilities.

## Build

```shell
cmake -S . -B build-lg200 \
  -DPLATFORM_LG200=ON \
  -DBUILD_TEST=ON
cmake --build build-lg200 --target lg200_info lg200_resource_test
```

Run `lg200_info --require-device` on a Loongson system to require at least one
matching DRM device.

## API

- `Lg200DeviceEnumerate()` enumerates `cardN` nodes whose driver name starts
  with `loonggpu` or whose product identity contains `LG200`.
- `Lg200DeviceReadStats()` reads `gpu_busy_percent`,
  `mem_info_vram_used`, and `mem_info_vram_total` when the driver exports them.

Statistics are optional. An unavailable utilization value is returned as
`-1.0`; unavailable memory counters are returned as zero.
