# GPU Server Topology And Bandwidth Profiler

这是一个用于分析和测试 GPU 服务器拓扑结构、带宽（Bandwidth）及延迟（Latency）的专业工具。它能够自动检测底层硬件平台，并生成详细的性能报告。

## 📋 功能特性

* **多平台支持**：程序启动时自动检测底层 GPU 硬件平台，无需用户手动指定参数。
  | 平台厂商 | 状态 |
  | :---- | :--- |
  | NVIDIA (英伟达) | ✅ 已支持 |
  | ILUVATAR (天数智芯) | ✅ 已支持 |
  | METAX（沐曦） | ✅ 已支持 |
  | MOORE THREADS（摩尔线程） | ✅ 已支持 |
  | HYGON（海光） | ✅ 已支持 |

* **全面检测**：
  * **CPU 信息**：核心数、架构、缓存大小、NUMA 节点等。
  * **GPU 拓扑**：GPU 设备信息、PCIe 总线信息、P2P (Peer-to-Peer) 支持矩阵。
  * **带宽测试**：GPU 到 GPU（单向/双向）、Host 到 GPU（Pageable/Pinned 内存）。
  * **延迟测试**：GPU 互联延迟、Host 与 Device 传输延迟。
* **多格式输出**：支持生成人类可读的日志 (`.log`)、数据分析用的 CSV 表格 (`.csv`) 以及程序解析用的 JSON (`.json`) 文件。

## 🛠️ 环境依赖

在运行之前，请确保环境满足以下要求：

1. **Python 3.6+**
2. **依赖库**：需安装 `numpy`

   ```bash
   pip install numpy
   ```
3. **驱动支持**：
   * **NVIDIA**: 需安装 CUDA 驱动及 Toolkit (确保 `libcudart.so` 可被加载)。

   * **Iluvatar / Metax / Moore Threads / Hygon**: 需安装相应供应商的软件栈。

## 🚀 快速开始

### 基础运行

默认情况下，工具会将结果输出为人类可读的日志文件 (`server_topo.log`)。

```bash
python3 main.py
```

### 完整运行示例

启用双向带宽测试，并同时输出 CSV 和 JSON 格式，测试重复 5 次以获取更稳定的平均值：

```bash
python3 main.py --format human csv json --bidirectional --repeat 5 --output my_server_test
```

---

## ⚙️ 运行参数 (Command Line Options)

该脚本提供丰富的命令行选项来控制测试行为。使用 `python3 main.py --help` 可查看简要说明。

| 参数                  | 简写   | 类型     | 默认值           | 说明                                                                                                                                           |
| :------------------- | :--- | :----- | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `--output`          | `-o` | String | server_topo | **输出文件前缀**。<br>例如设置为 `result`，则生成 `result.log`, `result.json` 等。                                                                             |
| `--format`       | N/A  | List   | human       | **输出格式**。<br>可选值：<br>• `human`: 生成易读的 `.log` 报告<br>• `csv`: 生成详细的 `.csv` 数据表<br>• `json`: 生成包含所有元数据的 `.json` 文件<br>用法示例：`--format human csv` |
| `--bidirectional`   | N/A  | Flag   | False         | **启用双向带宽测试**。<br>默认仅测试单向（Unidirectional）。开启此项会同时测试两个 GPU 互发数据的带宽，增加了测试负载。                                                               |
| `--buffer`          | N/A  | Int    | 256      | **测试缓冲区大小 (MiB)**。<br>用于带宽测试的显存大小。较大的缓冲区通常能测出更接近理论峰值的带宽。                                                                                     |
| `--fallback-buffer` | N/A  | Int    | 128         | **回退缓冲区大小 (MiB)**。<br>如果主缓冲区分配失败（例如显存不足），将尝试使用此大小进行测试。                                                                                       |
| `--repeat`          | N/A  | Int    | 3           | **重复测试次数**。<br>对每项测试运行 N 次并取平均值，以消除系统抖动影响。                                                                                                   |
| `--include-self`    | N/A  | Flag   | False         | **包含自身对自身的测试**。<br>在矩阵中包含 GPU(i) 到 GPU(i) 的回环测试数据。                                                                                       |
| `--log-level`       | N/A  | String | INFO        | **日志详细等级**。<br>可选值：`DEBUG`, `INFO`, `WARN`, `ERROR`。                                                                                   |
| `--ramp`            | N/A  | 标志  | False       | **使用逐步增加的缓冲区大小并报告最大带宽**。<br>启用此选项后，将使用递增的缓冲区大小进行测试，以找出带宽的峰值。                                                       |
| `--ramp-min`        | N/A  | 整数  | 1           | **最小递增缓冲区大小（MiB）**。<br>默认值为 1 MiB，用于递增测试。                                                                          |
| `--warmup`          | N/A  | 整数  | 5           | **计时前的预热迭代次数**。<br>默认进行 5 次预热迭代，以稳定测量结果。                                                                           |


## 📄 输出文件说明

根据 `--format` 参数的选择，脚本会生成以下文件：

### 1. Human Readable Log (`.log`)

包含格式化好的表格，适合直接阅读。内容包括：

* CPU 硬件详情 (lscpu)
* GPU 设备和 PCIe 链路状态 (Gen/Width)
* P2P 能力矩阵 (Yes/No)
* 本地拓扑 (smi topo -m)
* 带宽矩阵 (GB/s)
* 延迟矩阵 (us)

### 2. CSV Data (`.csv`)

生成多个 CSV 文件，方便导入 Excel 或 Pandas 进行绘图分析：

* `*_p2p_capability.csv`: P2P 支持情况
* `*_topology.csv`: 原始拓扑结构文本
* `*_gpu2gpu_bw_*.csv`: 各种模式下的 GPU 互联带宽数据
* `*_gpu2gpu_lat_*.csv`: GPU 互联延迟数据
* `*_gpu2host.csv`: 主机与设备间的带宽和延迟

### 3. JSON Data (`.json`)

包含所有测试元数据和结果的层级化数据结构，适合集成到监控系统或自动化流水线中。

## ⚠️ 常见问题

**Q: 提示 "lscpu command failed**<br>
A: 脚本依赖 Linux 的 `lscpu` 命令获取 CPU 信息。如果您的系统缺少该命令，CPU 信息部分将为空，但不影响 GPU 测试。

**Q: 提示 "No supported GPU platform detected"？**<br>
A: 脚本通过检查库路径中是否包含 `cuda` 或 `corex` 来判断平台。请确保环境变量 `LD_LIBRARY_PATH` 已正确设置，指向 CUDA 或 CoreX 的库文件目录。


**Q: HYGON 平台拓扑信息为空是否异常？**<br>
A: 在 **海光 (HYGON)** 平台上，如果 `smi` 版本低于 **1.6.x**，将 **不支持拓扑查看命令**。此时在输出中：

```
### 3. Native Topology (hy-smi topo -m) ###
```

对应的拓扑部分显示为空属于 **正常现象**，不影响其他带宽与延迟测试结果。

**Q: 测试过程中出现显存分配错误 (OOM)？**<br>
A: 尝试减小 `--buffer` 的值，例如使用 `--buffer 64`。

## 🛠️ 技术支持
如遇到问题，请联系我们技术支持团队
