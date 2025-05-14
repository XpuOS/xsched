# LevelZero Platform Support for XSched

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
    <td align="center" rowspan="2">LevelZero</a></td>
    <td align="center">Intel GPUs</td>
    <td align="center" rowspan="2">✅</td>
    <td align="center" rowspan="2">✅</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
  <tr>
    <td align="center">Intel Integrated NPUs</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
  </tr>
</table>

## Usage

### 1. Compile project

```bash
make levelzero
```

### 2. Run XServer

```bash
cd output/bin
export LD_LIBRARY_PATH=path/to/xsched/output/lib:$LD_LIBRARY_PATH;
./xserver
```

### 3. Set environment variables before running the application

```bash
export LD_LIBRARY_PATH=path/to/xsched/output/lib:$LD_LIBRARY_PATH;
export XSCHED_SCHEDULER=GLB XSCHED_AUTO_XQUEUE=1;
```

### 4. Run the application

```bash
# Take test as an example
cd platforms/levelzero/test
python3 npu.py
```
