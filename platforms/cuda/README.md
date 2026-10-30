# CUDA Platform Support for XSched

## Supported preemption level

<table>
  <tr>
    <th align="center">Platform</th>
    <th align="center">XPU</th>
    <th align="center">Shim</th>
    <th align="center">Level-1</th>
    <th align="center">Level-2</th>
    <th align="center">Level-3</th>
  </tr>
  <tr>
    <td align="center" rowspan="5">CUDA</td>
    <td align="center">NVIDIA Ampere GPUs (sm86)</td>
    <td align="center" rowspan="5">✅</td>
    <td align="center" rowspan="5">✅</td>
    <td align="center">🚧</td>
    <td align="center">🚧</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Volta GPUs (sm70)</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Kepler GPUs (sm35)</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Blackwell GPUs (sm120)</td>
    <td align="center">❌</td>
    <td align="center"><a href="BLACKWELL_LV3.md">🚧 CHANNEL, opt-in</a></td>
  </tr>
  <tr>
    <td align="center">Other NVIDIA GPUs</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
</table>

## Usage

Build the CUDA platform adapter:

```bash
make cuda
```

For transparent scheduling, see the [Linux transparent scheduling example](../../examples/Linux/1_transparent_sched/README.md).

## Experimental Blackwell Level 3

See [the CHANNEL backend documentation](BLACKWELL_LV3.md) for the validated
configuration, compute-group scope, build/test instructions and limitations.
