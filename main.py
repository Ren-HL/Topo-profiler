#!/usr/bin/env python3
"""
GPU server topology & bandwidth profiler 
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from typing import Any

from pathlib import Path

import numpy as np

from profiler import (
    GpuProfiler,
    LOG_LEVELS,
    set_log_level,
    log_info,
    log_warn,
    log_error,
)
from nvidiaApi import NvidiaApi, get_libcudart_path
from iluvatarApi import IluvatarApi
from metax_hpccAPI import HpccApi
from hygonApi import HygonApi


# ---------------- CPU Information ----------------
def _get_cpu_mhz():
    try:
        proc = subprocess.run(
            ["lscpu", "-e=MHZ,MAXMHZ,MINMHZ"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            return None

        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None

        for line in lines[1:]:
            parts = [p.strip() for p in line.split()]
            if not parts:
                continue

            mhz = parts[0]
            maxmhz = parts[1] if len(parts) > 1 else None
            minmhz = parts[2] if len(parts) > 2 else None

            if mhz != "-" or maxmhz != "-" or minmhz != "-":
                return {
                    "mhz": mhz if mhz != "-" else "N/A",
                    "maxmhz": maxmhz if maxmhz != "-" else "N/A",
                    "minmhz": minmhz if minmhz != "-" else "N/A",
                }

        return None
    except Exception:
        return None

def get_cpu_info() -> dict[str, Any]:
    cpu_info: dict[str, Any] = {}
    try:
        proc = subprocess.run(
            ["lscpu"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            log_warn("lscpu command failed: {}", proc.stderr)
            return cpu_info

        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = re.split(r":\s+", line, 1)
            if len(parts) != 2:
                continue

            key, value = parts[0].strip(), parts[1].strip()
            if key == "Model name":
                cpu_info["model"] = value
            elif key == "CPU(s)":
                cpu_info["total_cores"] = int(value)
            elif key == "Thread(s) per core":
                cpu_info["threads_per_core"] = int(value)
            elif key == "Core(s) per socket":
                cpu_info["cores_per_socket"] = int(value)
            elif key == "Socket(s)":
                cpu_info["sockets"] = int(value)
            elif key == "CPU MHz":
                cpu_info["base_freq_mhz"] = float(value)
            elif key == "CPU max MHz":
                cpu_info["max_freq_mhz"] = float(value)
            elif key == "CPU min MHz":
                cpu_info["min_freq_mhz"] = float(value)
            elif key == "L1d cache":
                cpu_info["l1d_cache"] = value
            elif key == "L1i cache":
                cpu_info["l1i_cache"] = value
            elif key == "L2 cache":
                cpu_info["l2_cache"] = value
            elif key == "L3 cache":
                cpu_info["l3_cache"] = value
            elif key == "Architecture":
                cpu_info["architecture"] = value
            elif key == "CPU family":
                cpu_info["family"] = value
            elif key == "Model":
                cpu_info["model_number"] = value
            elif key == "Stepping":
                cpu_info["stepping"] = value
            elif key == "BogoMIPS":
                cpu_info["bogomips"] = float(value)
            elif key == "NUMA node(s)":
                cpu_info["numa_nodes"] = int(value)

            if (
                cpu_info.get("base_freq_mhz") is None and
                cpu_info.get("max_freq_mhz") is None and
                cpu_info.get("min_freq_mhz") is None
            ):
                freq = _get_cpu_mhz()
                if freq:
                    cpu_info["base_freq_mhz"] = freq.get("mhz")
                    cpu_info["max_freq_mhz"] = freq.get("maxmhz")
                    cpu_info["min_freq_mhz"] = freq.get("minmhz")

    except Exception as e:
        log_warn("Failed to parse CPU information: {}", e)

    return cpu_info


# ---------------- Public formatting/output tool ----------------
def format_matrix(mat, labels, is_boolean=False):
    lines = []
    header_line = "       " + "".join(f"{lbl:>12}" for lbl in labels)
    lines.append(header_line)
    for idx, row in enumerate(mat):
        row_strs = []
        for val in row:
            if np.isnan(val):
                row_strs.append(f"{'---':>12}")
            elif is_boolean:
                row_strs.append(f"{'YES' if val else 'NO':>12}")
            else:
                row_strs.append(f"{val:12.2f}")
        lines.append(f"{labels[idx]:>6} " + "".join(row_strs))
    return "\n".join(lines)


def write_human_readable_log(
    fname: Path,
    cpu_info,
    meta,
    num_gpus,
    gpu_labels,
    p2p_capable,
    topo_str,
    args,
    bw_gpu_uni_disabled_read,
    bw_gpu_uni_disabled_write,
    bw_gpu_bi_disabled,
    lat_gpu_disabled_read,
    lat_gpu_disabled_write,
    bw_gpu_uni_enabled_read,
    bw_gpu_uni_enabled_write,
    bw_gpu_bi_enabled,
    lat_gpu_enabled_read,
    lat_gpu_enabled_write,
    pageable_h2d,
    pageable_d2h,
    pinned_h2d,
    pinned_d2h,
    lat_pageable_h2d,
    lat_pageable_d2h,
    lat_pinned_h2d,
    lat_pinned_d2h,
) -> None:
    """Human-readable log output"""
    with open(fname, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GPU SERVER TOPOLOGY & BANDWIDTH PROFILER RESULTS\n")
        f.write(
            f"Measurement Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        )
        f.write(
            f"Buffer Size: {args.buffer} MiB (fallback: {args.fallback_buffer} MiB)\n"
        )
        f.write(f"Repetitions: {args.repeat} (mean values reported)\n")

        total_cores = cpu_info.get("total_cores", "N/A")
        sockets = cpu_info.get("sockets", "N/A")
        f.write(f"CPUs Detected: {total_cores} cores ({sockets} sockets)\n")
        f.write(f"GPUs Detected: {num_gpus}\n")
        f.write("=" * 80 + "\n\n")

        # 0. CPU Information
        f.write("### 0. CPU Information ###\n")
        f.write("(Source: lscpu command)\n")
        if cpu_info:
            cpu_log_lines = [
                f"Model:           {cpu_info.get('model', 'N/A')}",
                f"Architecture:    {cpu_info.get('architecture', 'N/A')}",
                f"Sockets:         {cpu_info.get('sockets', 'N/A')}",
                f"Cores per Socket:{cpu_info.get('cores_per_socket', 'N/A')}",
                f"Threads per Core:{cpu_info.get('threads_per_core', 'N/A')}",
                f"Total Cores:     {cpu_info.get('total_cores', 'N/A')}",
                f"Base Freq (MHz): {cpu_info.get('base_freq_mhz', 'N/A')}",
                f"Max Freq (MHz):  {cpu_info.get('max_freq_mhz', 'N/A')}",
                f"Min Freq (MHz):  {cpu_info.get('min_freq_mhz', 'N/A')}",
                f"BogoMIPS:        {cpu_info.get('bogomips', 'N/A')}",
                f"NUMA node(s):    {cpu_info.get('numa_nodes', 'N/A')}",
                "Cache Sizes:",
                f"  L1d:           {cpu_info.get('l1d_cache', 'N/A')}",
                f"  L1i:           {cpu_info.get('l1i_cache', 'N/A')}",
                f"  L2:            {cpu_info.get('l2_cache', 'N/A')}",
                f"  L3:            {cpu_info.get('l3_cache', 'N/A')}",
            ]
            f.write("\n".join(cpu_log_lines) + "\n")
        else:
            f.write("  No CPU information available (lscpu failed or not supported)\n")
        f.write("\n" + "-" * 60 + "\n\n")

        # 1. GPU Device and PCIe
        f.write("### 1. GPU Device & PCIe Information ###\n")
        f.write("(Source: {})\n".format(meta[0]["source"] if meta else "unknown"))
        for m in meta:
            # print(
            #     "DEBUG:",
            #     "domain =", repr(m["pci_domain"]), type(m["pci_domain"]),
            #     "bus =", repr(m["pci_bus"]), type(m["pci_bus"]),
            #     "device =", repr(m["pci_device"]), type(m["pci_device"]),
            # )#因为报错看数据类型

            f.write(f"\nGPU{m['index']} - {m['name']}:\n")
            if isinstance(m["pci_domain"], int):
                pci_loc_line = (
                    f"  PCI Location:       Domain 0x{m['pci_domain']:04x}, "
                    f"Bus 0x{m['pci_bus']:02x}, Device 0x{m['pci_device']:02x}\n"
                )
            else:
                pci_loc_line = (
                    f"  PCI Location:       Domain {m['pci_domain']}, "
                    f"Bus {m['pci_bus']}, Device {m['pci_device']}\n"
                )
            f.write(f"  PCI Bus ID:         {m['pci_bus_id']}\n")
            f.write(pci_loc_line)
            f.write(
                f"  PCIe Max:           Gen {m['max_pcie_gen']} x{m['max_pcie_width']}\n"
            )
            f.write(
                f"  PCIe Current:       Gen {m['current_pcie_gen']} x{m['current_pcie_width']}\n"
            )
            f.write(f"  Total Memory:       {m['total_memory_gb']} GB\n")
            f.write(f"  Compute Capability: {m['compute_capability']}\n")
        f.write("\n" + "-" * 60 + "\n\n")

        # 2. P2P Capability Matrix
        f.write("### 2. P2P Capability Matrix ###\n")
        f.write("(YES = Direct Peer-to-Peer capable between GPU pairs)\n")
        f.write(format_matrix(p2p_capable, gpu_labels, is_boolean=True) + "\n\n")
        f.write("-" * 60 + "\n\n")

        # 3. Native Topology
        f.write(
            "### 3. Native Topology ({} topo -m) ###\n".format(
                meta[0]["source"] if meta else "SMI"
            )
        )
        f.write(topo_str + "\n\n")
        f.write("-" * 60 + "\n\n")

        # 4. GPU→GPU Bandwidth
        f.write("### 4. GPU→GPU Bandwidth (GB/s) ###\n")
        f.write(f"(Mean of {args.repeat} measurements)\n\n")

        f.write("P2P DISABLED:\n")
        f.write("Unidirectional (Read):\n")
        f.write(format_matrix(bw_gpu_uni_disabled_read, gpu_labels) + "\n")
        f.write("\nUnidirectional (Write):\n")
        f.write(format_matrix(bw_gpu_uni_disabled_write, gpu_labels) + "\n")
        if bw_gpu_bi_disabled is not None:
            f.write("\nBidirectional:\n")
            f.write(format_matrix(bw_gpu_bi_disabled, gpu_labels) + "\n")

        f.write("\nP2P ENABLED:\n")
        f.write("Unidirectional (Read):\n")
        f.write(format_matrix(bw_gpu_uni_enabled_read, gpu_labels) + "\n")
        f.write("\nUnidirectional (Write):\n")
        f.write(format_matrix(bw_gpu_uni_enabled_write, gpu_labels) + "\n")
        if bw_gpu_bi_enabled is not None:
            f.write("\nBidirectional:\n")
            f.write(format_matrix(bw_gpu_bi_enabled, gpu_labels) + "\n")
        f.write("\n" + "-" * 60 + "\n\n")

        # 5. GPU→GPU Latency
        f.write("### 5. GPU→GPU Latency (us) ###\n")
        f.write(f"(Mean of {args.repeat} measurements)\n\n")

        f.write("P2P DISABLED:\nUnidirectional (Read):\n")
        f.write(format_matrix(lat_gpu_disabled_read, gpu_labels) + "\n")

        f.write("\nP2P DISABLED:\nUnidirectional (Write):\n")
        f.write(format_matrix(lat_gpu_disabled_write, gpu_labels) + "\n")

        f.write("\nP2P ENABLED:\nUnidirectional (P2P Read):\n")
        f.write(format_matrix(lat_gpu_enabled_read, gpu_labels) + "\n")

        f.write("\nP2P ENABLED:\nUnidirectional (P2P Write):\n")
        f.write(format_matrix(lat_gpu_enabled_write, gpu_labels) + "\n")
        f.write("\n" + "-" * 60 + "\n\n")

        # 6. GPU↔Host
        f.write("### 6. GPU↔Host Bandwidth & Latency ###\n")
        f.write(
            f"(Mean of {args.repeat} measurements; latency in microseconds)\n\n"
        )

        header_fmt = "{:<12} | {:>5} | {:>14} | {:>14} | {:>14} | {:>14}\n"
        row_fmt = "{:<12} | {:>5} | {:14.2f} | {:14.2f} | {:14.2f} | {:14.2f}\n"

        f.write(
            header_fmt.format(
                "HostMemType",
                "GPU",
                "H->D BW (GB/s)",
                "D->H BW (GB/s)",
                "H->D Lat (us)",
                "D->H Lat (us)",
            )
        )
        f.write("-" * 90 + "\n")

        for i, lbl in enumerate(gpu_labels):
            f.write(
                row_fmt.format(
                    "pageable",
                    lbl,
                    pageable_h2d[i],
                    pageable_d2h[i],
                    lat_pageable_h2d[i],
                    lat_pageable_d2h[i],
                )
            )
        for i, lbl in enumerate(gpu_labels):
            f.write(
                row_fmt.format(
                    "pinned",
                    lbl,
                    pinned_h2d[i],
                    pinned_d2h[i],
                    lat_pinned_h2d[i],
                    lat_pinned_d2h[i],
                )
            )
        f.write("\n" + "-" * 60 + "\n\n")

    log_info("LOG output written to: {}", fname)


def save_csv_gpu2gpu(mat, labels, fname: Path, is_boolean=False) -> None:
    with open(fname, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([""] + labels)
        for i, lbl in enumerate(labels):
            row = []
            for j in range(len(labels)):
                if np.isnan(mat[i, j]):
                    row.append("")
                elif is_boolean:
                    row.append("YES" if mat[i, j] else "NO")
                else:
                    row.append(f"{mat[i, j]:.2f}")
            writer.writerow([lbl] + row)


def save_csv_host_mem(
    gpu_labels,
    pageable_h2d,
    pageable_d2h,
    pinned_h2d,
    pinned_d2h,
    lat_pageable_h2d,
    lat_pageable_d2h,
    lat_pinned_h2d,
    lat_pinned_d2h,
    fname: Path,
) -> None:
    with open(fname, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(
            [
                "HostMemType",
                "GPU",
                "H_to_D_GBps",
                "D_to_H_GBps",
                "H->D Lat (us)",
                "D->H Lat (us)",
            ]
        )
        for i, lbl in enumerate(gpu_labels):
            writer.writerow(
                [
                    "pageable",
                    lbl,
                    f"{pageable_h2d[i]:.2f}",
                    f"{pageable_d2h[i]:.2f}",
                    f"{lat_pageable_h2d[i]:.2f}",
                    f"{lat_pageable_d2h[i]:.2f}",
                ]
            )
        for i, lbl in enumerate(gpu_labels):
            writer.writerow(
                [
                    "pinned",
                    lbl,
                    f"{pinned_h2d[i]:.2f}",
                    f"{pinned_d2h[i]:.2f}",
                    f"{lat_pinned_h2d[i]:.2f}",
                    f"{lat_pinned_d2h[i]:.2f}",
                ]
            )


def _merge_topo_header(header_parts):
    merged = []
    i = 0
    n = len(header_parts)
    while i < n:
        if (
            i + 2 < n
            and header_parts[i] == "GPU"
            and header_parts[i + 1] == "NUMA"
            and header_parts[i + 2] == "ID"
        ):
            merged.append("GPU NUMA ID")
            i += 3
            continue
        if (
            i + 1 < n
            and header_parts[i + 1] == "Affinity"
            and header_parts[i] in ("CPU", "NUMA")
        ):
            merged.append(f"{header_parts[i]} Affinity")
            i += 2
            continue
        if (#hygon
            i + 1 < n
            and header_parts[i + 1] == "Node"
            and header_parts[i] in ("NUMA")
        ):
            merged.append(f"{header_parts[i]} Node")
            i += 2
            continue
        merged.append(header_parts[i])
        i += 1
    return merged


def save_csv_topo(topo_str: str, fname: Path) -> None:
    """Save the text of topo-m as a CSV file"""
    lines = [ln.rstrip() for ln in topo_str.split("\n") if ln.strip()]
    if not lines:
        return
    header_parts = lines[0].split()
    merged_header = _merge_topo_header(header_parts)

    with open(fname, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([""] + merged_header)

        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            row_label = parts[0]
            row_values = parts[1:]
            writer.writerow([row_label] + row_values)


def save_json_all(
    platform: str,
    cpu_info,
    meta,
    gpu_labels,
    p2p_capable,
    topo_str,
    args,
    bw_gpu_uni_disabled_read,
    bw_gpu_uni_disabled_write,
    bw_gpu_bi_disabled,
    lat_gpu_disabled_read,
    lat_gpu_disabled_write,
    bw_gpu_uni_enabled_read,
    bw_gpu_uni_enabled_write,
    bw_gpu_bi_enabled,
    lat_gpu_enabled_read,
    lat_gpu_enabled_write,
    pageable_h2d,
    pageable_d2h,
    pinned_h2d,
    pinned_d2h,
    lat_pageable_h2d,
    lat_pageable_d2h,
    lat_pinned_h2d,
    lat_pinned_d2h,
    fname: Path,
) -> None:
    data = {
        "measurement_info": {
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "platform": platform,
            "buffer_mib": args.buffer,
            "fallback_buffer_mib": args.fallback_buffer,
            "repeat": args.repeat,
            "bidirectional_enabled": bool(args.bidirectional),
            "include_self_gpu_pairs": bool(args.include_self),
        },
        "cpu_info": cpu_info,
        "gpu_metadata": meta,
        "gpu_labels": gpu_labels,
        "p2p_capability_matrix": p2p_capable.tolist(),
        "topology_native": {
            "raw": topo_str,
        },
        "gpu_to_gpu": {
            "bandwidth_GBps": {
                "p2p_disabled": {
                    "unidirectional_read": bw_gpu_uni_disabled_read.tolist(),
                    "unidirectional_write": bw_gpu_uni_disabled_write.tolist(),
                    "bidirectional": (
                        None
                        if bw_gpu_bi_disabled is None
                        else bw_gpu_bi_disabled.tolist()
                    ),
                },
                "p2p_enabled": {
                    "unidirectional_read": bw_gpu_uni_enabled_read.tolist(),
                    "unidirectional_write": bw_gpu_uni_enabled_write.tolist(),
                    "bidirectional": (
                        None
                        if bw_gpu_bi_enabled is None
                        else bw_gpu_bi_enabled.tolist()
                    ),
                },
            },
            "latency_us": {
                "p2p_disabled": {
                    "unidirectional_read": lat_gpu_disabled_read.tolist(),
                    "unidirectional_write": lat_gpu_disabled_write.tolist(),
                },
                "p2p_enabled": {
                    "unidirectional_read": lat_gpu_enabled_read.tolist(),
                    "unidirectional_write": lat_gpu_enabled_write.tolist(),
                },
            },
        },
        "gpu_to_host": {
            "bandwidth_GBps": {
                "pageable": {
                    "H_to_D": pageable_h2d.tolist(),
                    "D_to_H": pageable_d2h.tolist(),
                },
                "pinned": {
                    "H_to_D": pinned_h2d.tolist(),
                    "D_to_H": pinned_d2h.tolist(),
                },
            },
            "latency_us": {
                "pageable": {
                    "H_to_D": lat_pageable_h2d.tolist(),
                    "D_to_H": lat_pageable_d2h.tolist(),
                },
                "pinned": {
                    "H_to_D": lat_pinned_h2d.tolist(),
                    "D_to_H": lat_pinned_d2h.tolist(),
                },
            },
        },
    }

    with open(fname, "w") as jf:
        json.dump(data, jf, indent=2)
    log_info("JSON output written to: {}", fname)


# ---------------- Platform API selection ----------------
def detect_platform_by_smi():
    """
    Attempt to automatically identify GPU manufacturers 
    using the SMI command on various platforms.
    return:
        ("NVIDIA", "nvidia-smi")
        ("Iluvatar", "ixsmi")
        ("MetaX_Hpcc", "ht-smi")
    """
    candidates = [
        ("NVIDIA", "nvidia-smi"),
        ("Iluvatar", "ixsmi"),
        ("MetaX_Hpcc", "ht-smi"),
        ("Hygon", "hy-smi"),
    ]
    for name, cmd in candidates:
        try:
            proc = subprocess.run(
                [cmd, "-h"],         # -L List GPU
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                return name, cmd
        except FileNotFoundError:
            continue
    raise RuntimeError("Can not identify the GPU platform via any SMI command.")
def create_platform_api():
    # 1. First identify GPU by SMI
    try:
        platform, smi_cmd = detect_platform_by_smi()
        if platform == "NVIDIA":
            return NvidiaApi(), "NVIDIA"
        elif platform == "Iluvatar":
            return IluvatarApi(), "Iluvatar"
        elif platform == "MetaX_Hpcc":
            return HpccApi(), "MetaX_Hpcc"
        elif platform == "Hygon":
            return HygonApi(), "Hygon"
    except Exception:
        pass
    # 2. fallback：identify GPU by libcudart path 
    path = get_libcudart_path().lower()
    if "corex" in path or "iluvatar" in path:
        return IluvatarApi(), "Iluvatar"
    if "metax" in path or "ht" in path:
        return HpccApi(), "MetaX_Hpcc"
    if "cuda" in path:
        return NvidiaApi(), "NVIDIA"
    raise RuntimeError("Unknown GPU platform")

# def create_platform_api():
#     """
# Based on the libcudart path judgment platform:
# The path contains 'corex' → Iluvatar platform
# Otherwise, if it contains 'cuda' → NVIDIA platform
#     """
#     try:
#         path = get_libcudart_path()
#         lower = (path or "").lower()
#         log_info("libcudart path: {}", path)

#         if "corex" in lower:
#             log_info("Detected Iluvatar platform")
#             return IluvatarApi(), "Iluvatar"

#         if "cuda" in lower:
#             log_info("Detected NVIDIA platform")
#             return NvidiaApi(), "NVIDIA"

#         log_error(
#             "Cannot detect platform via libcudart path = {} "
#             "(no 'corex' or 'cuda' in path)",
#             path,
#         )
#     except Exception as e:
#         log_error("Platform detection failed: {}", e)

#     raise RuntimeError(
#         "No supported GPU platform detected. "
#         "Currently supported: NVIDIA (libcudart path contains 'cuda') "
#         "and Iluvatar (libcudart path contains 'corex')."
#     )


# ---------------- Main function ----------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPU-server topology & bandwidth profiler (c5, g3-style output)"
    )
    parser.add_argument(
        "-o", "--output", help="Output filename prefix", default="server_topo"
    )
    parser.add_argument(
        "--format",
        choices=["human", "csv", "json"],
        nargs="+",
        default=["human"],
        help="Output formats",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Measure simultaneous bidirectional GPU↔GPU",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=256,
        help="Buffer size in MiB for measurement",
    )
    parser.add_argument(
        "--fallback-buffer",
        type=int,
        default=128,
        help="Fallback buffer size in MiB for host transfer if initial allocation fails",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include GPU→self entries in GPU↔GPU matrix",
    )
    parser.add_argument(
        "--log-level",
        choices=list(LOG_LEVELS.keys()),
        default="INFO",
        help="Verbosity (ERROR, WARN, INFO, DEBUG)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of measurement repetitions (default: 3)",
    )

    args = parser.parse_args()
    set_log_level(args.log_level)

    if args.repeat < 1:
        log_error("Repeat count must be at least 1")
        return 1

    try:
        # platform API
        api, platform = create_platform_api()
        profiler = GpuProfiler(api)

        # CPU & GPU Information
        cpu_info = get_cpu_info()
        num_gpus = profiler.get_device_count()
        gpu_labels = profiler.get_gpu_labels(num_gpus)
        log_info("GPU labels: {}", gpu_labels)

        meta = profiler.query_gpu_metadata(num_gpus)
        try:
            topo_str = profiler.get_native_topology_str()
        except Exception as e:
            log_warn("Failed to get topology: {}", e)
            topo_str = ""

        p2p_capable = profiler.get_p2p_capability_matrix(num_gpus)

        buffer_bytes = args.buffer * 1024 * 1024
        fallback_bytes = args.fallback_buffer * 1024 * 1024
        log_info(
            "Using buffer size: {} MiB (fallback: {} MiB), {} repetitions",
            args.buffer,
            args.fallback_buffer,
            args.repeat,
        )

        # 1) P2P disabled
        log_info("\n=== P2P DISABLED measurements ===")
        profiler.reset_devices(num_gpus)
        (
            bw_gpu_uni_disabled_read,
            bw_gpu_uni_disabled_write,
            bw_gpu_bi_disabled,
            lat_gpu_disabled_read,
            lat_gpu_disabled_write,
        ) = profiler.measure_gpu_to_gpu(
            num_gpus=num_gpus,
            buffer_bytes=buffer_bytes,
            bidirectional=args.bidirectional,
            skip_self=(not args.include_self),
            description="P2P disabled",
            repeat=args.repeat,
        )

        # 2) P2P enabled
        log_info("\n=== P2P ENABLED measurements ===")
        profiler.reset_devices(num_gpus)
        profiler.enable_peer_access(num_gpus)
        (
            bw_gpu_uni_enabled_read,
            bw_gpu_uni_enabled_write,
            bw_gpu_bi_enabled,
            lat_gpu_enabled_read,
            lat_gpu_enabled_write,
        ) = profiler.measure_gpu_to_gpu(
            num_gpus=num_gpus,
            buffer_bytes=buffer_bytes,
            bidirectional=args.bidirectional,
            skip_self=(not args.include_self),
            description="P2P enabled",
            repeat=args.repeat,
        )
        profiler.disable_peer_access(num_gpus)

        # 3) GPU↔Host
        log_info("\n=== GPU↔Host measurements ===")
        (
            pageable_h2d,
            pageable_d2h,
            pinned_h2d,
            pinned_d2h,
            lat_pageable_h2d,
            lat_pageable_d2h,
            lat_pinned_h2d,
            lat_pinned_d2h,
        ) = profiler.measure_gpu_to_host(
            num_gpus=num_gpus,
            buffer_bytes=buffer_bytes,
            fallback_bytes=fallback_bytes,
            repeat=args.repeat,
        )

        out_prefix = Path(args.output)

        # human log
        if "human" in args.format:
            log_file = out_prefix.with_suffix(".log")
            write_human_readable_log(
                fname=log_file,
                cpu_info=cpu_info,
                meta=meta,
                num_gpus=num_gpus,
                gpu_labels=gpu_labels,
                p2p_capable=p2p_capable,
                topo_str=topo_str,
                args=args,
                bw_gpu_uni_disabled_read=bw_gpu_uni_disabled_read,
                bw_gpu_uni_disabled_write=bw_gpu_uni_disabled_write,
                bw_gpu_bi_disabled=bw_gpu_bi_disabled,
                lat_gpu_disabled_read=lat_gpu_disabled_read,
                lat_gpu_disabled_write=lat_gpu_disabled_write,
                bw_gpu_uni_enabled_read=bw_gpu_uni_enabled_read,
                bw_gpu_uni_enabled_write=bw_gpu_uni_enabled_write,
                bw_gpu_bi_enabled=bw_gpu_bi_enabled,
                lat_gpu_enabled_read=lat_gpu_enabled_read,
                lat_gpu_enabled_write=lat_gpu_enabled_write,
                pageable_h2d=pageable_h2d,
                pageable_d2h=pageable_d2h,
                pinned_h2d=pinned_h2d,
                pinned_d2h=pinned_d2h,
                lat_pageable_h2d=lat_pageable_h2d,
                lat_pageable_d2h=lat_pageable_d2h,
                lat_pinned_h2d=lat_pinned_h2d,
                lat_pinned_d2h=lat_pinned_d2h,
            )

        # CSV
        if "csv" in args.format:
            save_csv_gpu2gpu(
                p2p_capable,
                gpu_labels,
                out_prefix.with_name(out_prefix.name + "_p2p_capability.csv"),
                is_boolean=True,
            )
            save_csv_topo(
                topo_str,
                out_prefix.with_name(out_prefix.name + "_topology.csv"),
            )
            # GPU-GPU BW
            save_csv_gpu2gpu(
                bw_gpu_uni_disabled_read,
                gpu_labels,
                out_prefix.with_name(
                    out_prefix.name + "_gpu2gpu_bw_uni_read_p2p_disabled.csv"
                ),
            )
            save_csv_gpu2gpu(
                bw_gpu_uni_disabled_write,
                gpu_labels,
                out_prefix.with_name(
                    out_prefix.name + "_gpu2gpu_bw_uni_write_p2p_disabled.csv"
                ),
            )
            if bw_gpu_bi_disabled is not None:
                save_csv_gpu2gpu(
                    bw_gpu_bi_disabled,
                    gpu_labels,
                    out_prefix.with_name(
                        out_prefix.name + "_gpu2gpu_bw_bi_p2p_disabled.csv"
                    ),
                )
            save_csv_gpu2gpu(
                bw_gpu_uni_enabled_read,
                gpu_labels,
                out_prefix.with_name(
                    out_prefix.name + "_gpu2gpu_bw_uni_read_p2p_enabled.csv"
                ),
            )
            save_csv_gpu2gpu(
                bw_gpu_uni_enabled_write,
                gpu_labels,
                out_prefix.with_name(
                    out_prefix.name + "_gpu2gpu_bw_uni_write_p2p_enabled.csv"
                ),
            )
            if bw_gpu_bi_enabled is not None:
                save_csv_gpu2gpu(
                    bw_gpu_bi_enabled,
                    gpu_labels,
                    out_prefix.with_name(
                        out_prefix.name + "_gpu2gpu_bw_bi_p2p_enabled.csv"
                    ),
                )
            # GPU-GPU latency
            save_csv_gpu2gpu(
                lat_gpu_disabled_read,
                gpu_labels,
                out_prefix.with_name(
                    out_prefix.name + "_gpu2gpu_lat_uni_read_p2p_disabled.csv"
                ),
            )
            save_csv_gpu2gpu(
                lat_gpu_disabled_write,
                gpu_labels,
                out_prefix.with_name(
                    out_prefix.name + "_gpu2gpu_lat_uni_write_p2p_disabled.csv"
                ),
            )
            save_csv_gpu2gpu(
                lat_gpu_enabled_read,
                gpu_labels,
                out_prefix.with_name(
                    out_prefix.name + "_gpu2gpu_lat_uni_read_p2p_enabled.csv"
                ),
            )
            save_csv_gpu2gpu(
                lat_gpu_enabled_write,
                gpu_labels,
                out_prefix.with_name(
                    out_prefix.name + "_gpu2gpu_lat_uni_write_p2p_enabled.csv"
                ),
            )
            # GPU-Host
            save_csv_host_mem(
                gpu_labels,
                pageable_h2d,
                pageable_d2h,
                pinned_h2d,
                pinned_d2h,
                lat_pageable_h2d,
                lat_pageable_d2h,
                lat_pinned_h2d,
                lat_pinned_d2h,
                out_prefix.with_name(out_prefix.name + "_gpu2host.csv"),
            )

        # JSON
        if "json" in args.format:
            json_file = out_prefix.with_suffix(".json")
            save_json_all(
                platform,
                cpu_info,
                meta,
                gpu_labels,
                p2p_capable,
                topo_str,
                args,
                bw_gpu_uni_disabled_read,
                bw_gpu_uni_disabled_write,
                bw_gpu_bi_disabled,
                lat_gpu_disabled_read,
                lat_gpu_disabled_write,
                bw_gpu_uni_enabled_read,
                bw_gpu_uni_enabled_write,
                bw_gpu_bi_enabled,
                lat_gpu_enabled_read,
                lat_gpu_enabled_write,
                pageable_h2d,
                pageable_d2h,
                pinned_h2d,
                pinned_d2h,
                lat_pageable_h2d,
                lat_pageable_d2h,
                lat_pinned_h2d,
                lat_pinned_d2h,
                json_file,
            )

        log_info("\n=== All measurements completed successfully ===")
        return 0

    except Exception as e:
        log_error("Fatal error: {}", e)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
