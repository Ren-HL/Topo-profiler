# GPU Server Topology and Bandwidth Profiler

A professional tool for analyzing and testing GPU server topology, bandwidth, and latency.
It automatically detects the underlying hardware platform and generates detailed performance reports.

## 📋 Features

* **Multi-platform support**: Automatically detects the GPU hardware platform on startup without requiring manual configuration.

  | Vendor   | Status      |
  | :------- | :---------- |
  | NVIDIA   | ✅ Supported |
  | ILUVATAR | ✅ Supported |
  | METAX | ✅ Supported |
  | MOORE THREADS | ✅ Supported |
  | HYGON | ✅ Supported |

* **Comprehensive inspection**:

  * **CPU Information**: Cores, architecture, cache size, NUMA nodes, etc.
  * **GPU Topology**: GPU device info, PCIe bus details, P2P (Peer-to-Peer) capability matrix.
  * **Bandwidth Tests**: GPU-to-GPU (uni/bidirectional), Host-to-GPU (Pageable/Pinned memory).
  * **Latency Tests**: GPU interconnect latency, host-device transfer latency.

* **Multi-format output**: Generates human-readable logs (`.log`), data-analysis-ready CSV tables (`.csv`), and machine-parsable JSON files (`.json`).

## 🛠️ Environment Requirements

Before running the tool, ensure your environment meets the following:

1. **Python 3.6+**
2. **Required library**: Install `numpy`

   ```bash
   pip install numpy
   ```
3. **Driver support**:

   * **NVIDIA**: CUDA drivers and Toolkit installed (ensure `libcudart.so` is loadable).
   * **Iluvatar / Metax / Moore Threads / Hygon**: Corresponding vendor software stack installed.

## 🚀 Quick Start

### Basic Run

By default, the tool outputs a human-readable log file (`server_topo.log`):

```bash
python3 main.py
```

### Full Example

Enable bidirectional bandwidth tests, output CSV and JSON formats,
and repeat each test 5 times to obtain stable averages:

```bash
python3 main.py --format human csv json --bidirectional --repeat 5 --output my_server_test
```

---

## ⚙️ Command Line Options

Run `python3 main.py --help` for brief help text. Below is the full list:

| Option              | Short | Type   | Default     | Description                                                                                                                                                        |
| :------------------ | :---- | :----- | :---------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--output`          | `-o`  | String | server_topo | **Output file prefix**.<br>Example: if set to `result`, files like `result.log` / `result.json` will be generated.                                                 |
| `--format`          | N/A   | List   | human       | **Output formats**.<br>Available: <br>• `human` (readable `.log`)<br>• `csv` (structured tables)<br>• `json` (full metadata dump)<br>Example: `--format human csv` |
| `--bidirectional`   | N/A   | Flag   | False       | **Enable bidirectional bandwidth tests**.<br>By default only unidirectional tests are performed.                                                                   |
| `--buffer`          | N/A   | Int    | 256         | **Bandwidth test buffer size (MiB)**.<br>Larger values usually produce results closer to theoretical peak bandwidth.                                               |
| `--fallback-buffer` | N/A   | Int    | 128         | **Fallback buffer size (MiB)** if the main buffer allocation fails.                                                                                                |
| `--repeat`          | N/A   | Int    | 3           | **Number of test repetitions**.<br>Results are averaged to reduce system variance.                                                                                 |
| `--include-self`    | N/A   | Flag   | False       | **Include self-tests** (GPU(i) → GPU(i)).                                                                                                                          |
| `--log-level`       | N/A   | String | INFO        | **Log verbosity**.<br>Options: `DEBUG`, `INFO`, `WARN`, `ERROR`.                                                                                                   |
| `--ramp`            | N/A   | Flag   | False       | **Use ramped buffer sizes** and report maximum bandwidth.<br>Enables testing with progressively increasing buffer sizes to find the peak bandwidth.                |
| `--ramp-min`        | N/A   | Int    | 1           | **Minimum ramp buffer size in MiB**.<br>Default is 1 MiB for ramped testing.                                                                                       |
| `--warmup`          | N/A   | Int    | 5           | **Warm-up iterations before timing**.<br>Default is 5 warm-up iterations to stabilize measurements.                                                                |



## 📄 Output Files

Depending on the `--format` option, the script generates the following:

### 1. Human-Readable Log (`.log`)

Contains structured tables suitable for direct reading. Includes:

* CPU hardware details (`lscpu`)
* GPU device and PCIe link status (Gen/Width)
* P2P capability matrix (Yes/No)
* Local topology (`smi topo -m`)
* Bandwidth matrix (GB/s)
* Latency matrix (µs)

### 2. CSV Data (`.csv`)

Multiple CSV files ideal for Excel or Pandas analysis:

* `*_p2p_capability.csv`: P2P capability info
* `*_topology.csv`: raw topology text
* `*_gpu2gpu_bw_*.csv`: GPU-to-GPU bandwidth tests
* `*_gpu2gpu_lat_*.csv`: GPU-to-GPU latency tests
* `*_gpu2host.csv`: host-device bandwidth & latency

### 3. JSON Data (`.json`)

A full hierarchical structure containing all metadata and results—suitable for automation pipelines or monitoring systems.

## ⚠️ FAQ

**Q: “lscpu command failed”**<br>
A: The script depends on the Linux `lscpu` command. If missing, CPU info will be empty, but GPU tests are unaffected.

---

**Q: “No supported GPU platform detected”**<br>
A: The script detects the platform by checking for `cuda` or `corex` in the library paths.
Ensure `LD_LIBRARY_PATH` is correctly set to your CUDA or CoreX library directories.

---

**Q: Is it abnormal if the topology information on the HYGON platform is empty?**  \
A: On the **HYGON** platform, if the `smi` version is lower than **1.6.x**, the **topology viewing command** will **not be supported**. At this time, in the output:

```
### 3. Native Topology (hy-smi topo -m) ###
```

The corresponding topology section showing as empty is considered a **normal phenomenon** and does not affect the results of other bandwidth and latency tests.

---
**Q: Out-of-memory error (OOM) during tests?**<br>
A: Reduce the buffer size, e.g.: `--buffer 64`.

## 🛠️ Technical Support

If you encounter issues, please contact our technical support team.


