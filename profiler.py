#!/usr/bin/env python3
"""
GPU Profiler class - Platform-independent implementation
- Only relies on the InfiniApi abstract interface
- All CUDA API calls have been changed to infiniXxx
- Includes all measurement and query functions
"""

from __future__ import annotations
from typing import Any, Tuple, List, Optional
import ctypes
import subprocess
import re
import os
import sys
import csv
from io import StringIO

import numpy as np

from infiniAPI import InfiniApi


# --------- General logging tool ---------
LOG_LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}
_current_log_level = LOG_LEVELS["INFO"]


def set_log_level(level_str: str) -> None:
    global _current_log_level
    _current_log_level = LOG_LEVELS.get(level_str.upper(), _current_log_level)


def log_debug(msg: str, *args: Any) -> None:
    if _current_log_level >= LOG_LEVELS["DEBUG"]:
        print("[DEBUG]", msg.format(*args))


def log_info(msg: str, *args: Any) -> None:
    if _current_log_level >= LOG_LEVELS["INFO"]:
        print("[INFO]", msg.format(*args))


def log_warn(msg: str, *args: Any) -> None:
    if _current_log_level >= LOG_LEVELS["WARN"]:
        print("[WARN]", msg.format(*args))


def log_error(msg: str, *args: Any) -> None:
    print("[ERROR]", msg.format(*args), file=sys.stderr)


# CUDA Constant
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaErrorPeerAccessNotEnabled = 999
cudaErrorPeerAccessAlreadyEnabled = 1000
cudaHostAllocDefault = 0


def _strip_ansi(s: str) -> str:
    """Remove the ANSI escape sequence"""
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


class GpuProfiler:
    """GPU Profiler - Platform-independent implementation"""
    
    def __init__(self, api: InfiniApi) -> None:
        self.api = api

    def _check_infini(self, err: int, ctx: str = "", dev: int | None = None) -> None:
        """Check the result of the infini API call"""
        if err != 0:
            try:
                err_bytes = self.api.infiniGetErrorString(err)
                err_str = err_bytes.decode("utf-8") if err_bytes else "<unknown>"
            except Exception:
                err_str = "<unknown>"
            
            if dev is None:
                log_error("Infini error in {}: code={} ({})", ctx, err, err_str)
            else:
                log_error("Infini error on dev {} in {}: code={} ({})", dev, ctx, err, err_str)
            raise RuntimeError(f"Infini error {err} ({err_str}) in {ctx}")

    # -------------------- Basic information --------------------
    def get_device_count(self) -> int:
        """Obtain the number of Gpus"""
        cnt = ctypes.c_int()
        self._check_infini(self.api.infiniGetDeviceCount(ctypes.byref(cnt)), "infiniGetDeviceCount")
        if cnt.value <= 0:
            raise RuntimeError("No GPUs found")
        log_info("Found {} GPU(s)", cnt.value)
        return int(cnt.value)

    def get_gpu_labels(self, num_gpus: int) -> List[str]:
        """Generate a list of GPU tags"""
        return [f"GPU{i}" for i in range(num_gpus)]

    # -------------------- Device Management --------------------
    def reset_devices(self, num_gpus: int) -> None:
        """Reset all GPU devices"""
        for i in range(num_gpus):
            self._check_infini(self.api.infiniSetDevice(i), f"infiniSetDevice({i})", dev=i)
            self._check_infini(self.api.infiniDeviceReset(), f"infiniDeviceReset({i})", dev=i)

    # -------------------- P2P capability --------------------
    def _device_can_access_peer(self, src: int, dst: int) -> bool:
        """Check the P2P access capability"""
        can = ctypes.c_int(0)
        err = self.api.infiniDeviceCanAccessPeer(ctypes.byref(can), src, dst)
        if err != 0:
            try:
                err_bytes = self.api.infiniGetErrorString(err)
                err_str = err_bytes.decode("utf-8") if err_bytes else "<unknown>"
            except Exception:
                err_str = "<unknown>"
            log_warn("infiniDeviceCanAccessPeer({}->{}) failed: code={} ({})", src, dst, err, err_str)
            return False
        return bool(can.value)

    def enable_peer_access(self, num_gpus: int) -> None:
        """Enable all available P2P access"""
        log_info("Enabling peer access for capable GPU pairs")
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i == j:
                    continue
                if self._device_can_access_peer(i, j):
                    self._check_infini(self.api.infiniSetDevice(i), f"infiniSetDevice({i})", dev=i)
                    err = self.api.infiniDeviceEnablePeerAccess(j, 0)
                    if err == 0:
                        log_info("Enabled peer access {}->{}", i, j)
                    elif err == cudaErrorPeerAccessAlreadyEnabled:
                        log_debug("Peer access {}->{} already enabled", i, j)
                    else:
                        self._check_infini(err, f"infiniDeviceEnablePeerAccess({i}->{j})", dev=i)

    def disable_peer_access(self, num_gpus: int) -> None:
        """Disable all P2P access"""
        log_info("Disabling peer access for all GPU pairs")
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i == j:
                    continue
                self._check_infini(self.api.infiniSetDevice(i), f"infiniSetDevice({i})", dev=i)
                err = self.api.infiniDeviceDisablePeerAccess(j)
                if err == 0:
                    log_debug("Disabled peer access {}->{}", i, j)
                elif err == cudaErrorPeerAccessNotEnabled:
                    log_debug("Peer access {}->{} not enabled", i, j)
                else:
                    self._check_infini(err, f"infiniDeviceDisablePeerAccess({i}->{j})", dev=i)

    def get_p2p_capability_matrix(self, num_gpus: int) -> np.ndarray:
        """Obtain the P2P capability matrix"""
        p2p_capable = np.zeros((num_gpus, num_gpus), dtype=bool)
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i == j:
                    p2p_capable[i, j] = False
                    continue
                capable = self._device_can_access_peer(i, j)
                p2p_capable[i, j] = capable
                if capable:
                    log_info("GPU {} can access GPU {} (P2P capable)", i, j)
                else:
                    log_info("GPU {} cannot access GPU {}", i, j)
        return p2p_capable

    # -------------------- Topological information --------------------

    def _get_hygon_native_topology_str(self) -> str:#Revised Version
        """Obtain Hygon native topology information (using --showtopo)"""
        smi = self.api.smi or "hy-smi"
        proc = subprocess.run(
            [smi, "--showtopo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{smi} --showtopo failed: {proc.stderr}")
        
        raw = proc.stdout
        raw2 = raw.replace("\t", "    ")
        lines = raw2.splitlines()
        link_block: list[str] = []
        numa_block: list[str] = []

        # Automatic detection device prefix (DCU or HCU)
        device_prefix = None
        for ln in lines:
            stripped = _strip_ansi(ln).strip()
            if "Link Type between" in stripped:
                if "DCUs" in stripped:
                    device_prefix = "DCU"
                elif "HCUs" in stripped:
                    device_prefix = "HCU"
                break
        
        if not device_prefix:
            # No detection was made. Attempting to infer from the NUMA information.
            for ln in lines:
                stripped = _strip_ansi(ln).strip()
                if stripped.startswith("DCU["):
                    device_prefix = "DCU"
                    break
                elif stripped.startswith("HCU["):
                    device_prefix = "HCU"
                    break
        
        if not device_prefix:
            # Nothing was found. Return to the original output.
            return raw2

        # Extract the Link Type table
        in_link = False
        link_pattern = f"Link Type between {device_prefix}s"
        
        for ln in lines:
            ln2 = _strip_ansi(ln).rstrip("\n")
            stripped = ln2.strip()

            if not in_link and link_pattern in stripped:
                in_link = True
                continue

            if in_link:
                if re.match(r"^=+$", stripped):
                    in_link = False
                    continue
                link_block.append(ln2)

        # Extract NUMA information (compatible with two formats)
        for ln in lines:
            ln2 = _strip_ansi(ln).rstrip("\n")
            stripped = ln2.strip()
            if stripped.startswith(f"{device_prefix}[") and (
                "Numa Node" in stripped or "Numa Affinity" in stripped
            ):
                numa_block.append(ln2)

        if not link_block and not numa_block:
            return raw2

        # Parse NUMA information (compatible with two formats)
        numa_map: dict[int, dict[str, str]] = {}
        for ln in numa_block:
            s = _strip_ansi(ln).strip()
            
            # Server A format: DCU[0] : Numa Node: 3
            # Server B format: HCU[0] : (Topology) Numa Node 0
            m_node = re.match(
                rf"{device_prefix}\[(\d+)\].*Numa Node[:\s]+(\S+)",
                s
            )
            m_aff = re.match(
                rf"{device_prefix}\[(\d+)\].*Numa Affinity[:\s]+(\S+)",
                s
            )
            
            if m_node:
                idx = int(m_node.group(1))
                val = m_node.group(2)
                numa_map.setdefault(idx, {})["node"] = val
            elif m_aff:
                idx = int(m_aff.group(1))
                val = m_aff.group(2)
                numa_map.setdefault(idx, {})["affinity"] = val

        # Calculate the width of the new column
        node_header = "NUMA Node"
        aff_header  = "NUMA Affinity"
        node_vals = [info.get("node", "N/A") for info in numa_map.values()]
        aff_vals  = [info.get("affinity", "N/A") for info in numa_map.values()]
        node_w = max(len(node_header), max((len(str(v)) for v in node_vals), default=0))
        aff_w  = max(len(aff_header),  max((len(str(v)) for v in aff_vals),  default=0))

        # Merge information
        base_width = max(len(_strip_ansi(l)) for l in link_block) if link_block else 0
        out_lines: list[str] = []
        header_added = False

        for ln in link_block:
            bare = _strip_ansi(ln)
            stripped = bare.strip()
            padded = bare.ljust(base_width)

            # Check the header (including DCU[0] or HCU[0])
            if (not header_added) and (f"{device_prefix}[0]" in stripped):
                header_added = True
                hdr = padded + "  " + f"{node_header:<{node_w}}  {aff_header:<{aff_w}}"
                out_lines.append(hdr)
                continue

            # Data row for testing
            if stripped.startswith(f"{device_prefix}["):
                m = re.match(rf"{device_prefix}\[(\d+)\]", stripped)
                if m:
                    idx = int(m.group(1))
                    info = numa_map.get(idx, {})
                    node = str(info.get("node", "N/A"))
                    aff  = str(info.get("affinity", "N/A"))
                    row = padded + "  " + f"{node:<{node_w}}  {aff:<{aff_w}}"
                    out_lines.append(row)
                    continue
            
            out_lines.append(bare)

        return "\n".join(out_lines)

    def get_native_topology_str(self) -> str:
        """Obtain native topology information (using smi topo-m)"""
        smi = self.api.smi or "nvidia-smi"
        smi_basename = os.path.basename(smi)
        if smi_basename in ("hy-smi", "rocm-smi"):
            return self._get_hygon_native_topology_str()
        
        proc = subprocess.run(
            [smi, "topo", "-m"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{smi} topo -m failed: {proc.stderr}")

        raw = proc.stdout
        raw2 = raw.replace("\t", "    ")
        lines = raw2.splitlines()
        header = None
        rows: list[list[str]] = []

        def merge_header_tokens(tokens: list[str]) -> list[str]:
            merged: list[str] = []
            i = 0
            while i < len(tokens):
                if tokens[i] == "Node" and i + 1 < len(tokens) and tokens[i + 1] == "Affinity":
                    merged.append("Node Affinity")
                    i += 2
                    continue
                if tokens[i] == "CPU" and i + 1 < len(tokens) and tokens[i + 1] == "Affinity":
                    merged.append("CPU Affinity")
                    i += 2
                    continue
                if tokens[i] == "NUMA" and i + 1 < len(tokens) and tokens[i + 1] == "Affinity":
                    merged.append("NUMA Affinity")
                    i += 2
                    continue
                if (tokens[i] == "GPU" and i + 2 < len(tokens) and 
                    tokens[i + 1] == "NUMA" and tokens[i + 2] == "ID"):
                    merged.append("GPU NUMA ID")
                    i += 3
                    continue
                merged.append(tokens[i])
                i += 1
            return merged

        for ln in lines:
            ln2 = _strip_ansi(ln).rstrip("\n")
            if header is None and re.match(r"^\s*GPU0\s", ln2):
                tokens = re.split(r"\s+", ln2.strip())
                tokens = merge_header_tokens(tokens)
                header = [""] + tokens
                continue
            if header is not None and ln2.strip().startswith("GPU"):
                tokens = re.split(r"\s+", ln2.strip())
                rows.append(tokens)

        if header is None:
            return raw2

        cols = len(header)
        widths = [len(h) for h in header]
        for row in rows:
            for i, tok in enumerate(row):
                if i < cols:
                    widths[i] = max(widths[i], len(tok))

        fmt = "     ".join("{:<" + str(widths[i]) + "}" for i in range(cols))
        out_lines = [fmt.format(*header)]
        for row in rows:
            row2 = row + [""] * (cols - len(row))
            out_lines.append(fmt.format(*row2))

        return "\n".join(out_lines)

    # -------------------- GPU metadata --------------------
    def _parse_metax_smi_for_device_info(self, num_gpus: int, smi: str):

        meta = []

        # ----------  ht-smi -L ----------
        try:
            proc_L = subprocess.run(
                [smi, "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            lines_L = proc_L.stdout.splitlines()
        except Exception as e:
            log_warn("Failed to run ht-smi -L: {}", e)
            return meta

        # basic structure
        gpu_basic = {}
        for line in lines_L:
            line = line.strip()
            if not line.startswith("GPU#"):
                continue

            parts = line.split()
            idx = int(parts[0][4:])
            name = parts[1]
            pci_bus_id = parts[2]

            # Analyzing PCI domains / buses / devices
            try:
                dom_s, bus_s, dev_s = pci_bus_id.split(":")
                dev_s, _ = dev_s.split(".")
                pci_domain = int(dom_s, 16)
                pci_bus = int(bus_s, 16)
                pci_device = int(dev_s, 16)
            except Exception:
                pci_domain = pci_bus = pci_device = "N/A"

            uuid = None
            if "UUID:" in line:
                uuid = line.split("UUID:")[1].strip(" )")

            gpu_basic[idx] = {
                "index": idx,
                "name": name,
                "pci_bus_id": pci_bus_id,
                "pci_domain": pci_domain,
                "pci_bus": pci_bus,
                "pci_device": pci_device,
                "uuid": uuid,
                "max_pcie_gen": "N/A",
                "max_pcie_width": "N/A",
                "current_pcie_gen": "N/A",
                "current_pcie_width": "N/A",
                "total_memory_gb": "N/A",
                "compute_capability": "N/A",
                "source": "ht-smi",
            }

        # ---------- Parse the main output of ht-smi (to obtain total_memory_gb) ----------
        try:
            proc_main = subprocess.run(
                [smi],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            lines = proc_main.stdout.splitlines()
        except Exception as e:
            log_warn("Failed to run ht-smi: {}", e)
            return list(gpu_basic.values())

        cur_idx = None
        for ln in lines:
            ln = ln.strip()

            m = re.match(r"^\|\s*(\d+)\s+.*\|\s+[0-9a-fA-F:.]+\s+\|", ln)
            if m:
                cur_idx = int(m.group(1))
                continue

            if cur_idx is not None and "MiB" in ln and "|" in ln:
                parts = ln.split("|")
                if len(parts) >= 3:
                    mem_field = parts[2].strip().split()[0]  # 863/65536
                    try:
                        _, total_mib = mem_field.split("/")
                        total_mib = float(total_mib)
                        gpu_basic[cur_idx]["total_memory_gb"] = round(total_mib / 1024.0, 2)
                    except:
                        pass
                cur_idx = None

        # Guaranteed to return results sorted by index.
        for d in gpu_basic.values():
            pci_bus_id = d.get("pci_bus_id")
            pcie_gen = pcie_width = pcie_current_gen = pcie_current_width = "N/A"
            if isinstance(pci_bus_id, str) and pci_bus_id not in ("", "N/A"):
                try:
                    pcie = self._query_pcie_link_info_from_sysfs(pci_bus_id)
                    pcie_gen = self.get_pcie_gen(pcie.get("max_speed"))
                    pcie_width = pcie.get("max_width") or "N/A"
                    pcie_current_gen = self.get_pcie_gen(pcie.get("cur_speed"))
                    pcie_current_width = pcie.get("cur_width") or "N/A"
                except Exception as e:
                    log_warn("PCIe sysfs query failed: {}", e)

            d["max_pcie_gen"] = pcie_gen
            d["max_pcie_width"] = pcie_width
            d["current_pcie_gen"] = pcie_current_gen
            d["current_pcie_width"] = pcie_current_width
        return [gpu_basic[i] for i in sorted(gpu_basic)]
    

    def get_pcie_gen(self, speed) -> str:
        """
        speed: It can be '16GT/s'/'16.0GT/s'/'16' / 16/16.0
        Return: '1'... '6' or 'N/A'
        """
        if speed is None:
            return "N/A"

        # Extract the GT/s numbers in the string
        if isinstance(speed, str):
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GT/s", speed)
            if m:
                gt = float(m.group(1))
            else:
                # Direct transmission of "16" is allowed
                try:
                    gt = float(speed.strip())
                except Exception:
                    return "N/A"
        else:
            try:
                gt = float(speed)
            except Exception:
                return "N/A"

        speed_to_gen = {2.5: "1", 5.0: "2", 8.0: "3", 16.0: "4", 32.0: "5", 64.0: "6"}
        # Avoid floating-point errors
        for k, v in speed_to_gen.items():
            if abs(gt - k) < 1e-6:
                return v
        return "N/A"

    def _query_pcie_link_info_from_sysfs(self, pci_bus_id: str) -> dict:
        """
        Query PCIe link information from sysfs.

        Return dict:
            {
                "max_speed": str | None,
                "max_width": str | None,
                "cur_speed": str | None,
                "cur_width": str | None,
            }
        """
        base = f"/sys/bus/pci/devices/{pci_bus_id}"

        def _read(path):
            try:
                with open(path, "r") as f:
                    return f.read().strip()
            except Exception:
                return None

        info = {
            "cur_speed": _read(f"{base}/current_link_speed"),
            "cur_width": _read(f"{base}/current_link_width"),
            "max_speed": _read(f"{base}/max_link_speed"),
            "max_width": _read(f"{base}/max_link_width"),
        }

        return info

    def _parse_hygon_smi_for_device_info(self, num_gpus: int, smi: str):
        """
        Parse hy-smi output (compatible with HY-SMI 1.4.x / 1.6.x),
        auto-fallback for unsupported options and fix broken CSV.
        """

        meta = []
        smi = self.api.smi or "hy-smi"

        # -------- 1. Try high-version command first --------
        cmd_candidates = [
            [smi, "--showproductname", "--showbus", "--showmemavailable", "--csv"],  # HY-SMI >= 1.6
            [smi, "--showproductname", "--showbus", "--csv"],                         # HY-SMI 1.4.x
        ]

        proc = None
        for cmd in cmd_candidates:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                break
        else:
            log_warn("{} query failed: {}", smi, proc.stderr if proc else "unknown error")
            return meta

        lines = proc.stdout.strip().splitlines()
        if not lines:
            return meta

        # -------- 2. Fix broken CSV (vendor contains comma) --------
        header = lines[0]
        header_parts = header.split(",")
        expected_fields = len(header_parts)
        has_time_column = "time" in header.lower()

        fixed_lines = [header]

        for idx, line in enumerate(lines[1:], start=1):
            parts = line.split(",")

            if len(parts) == expected_fields:
                fixed_lines.append(line)
                continue

            # Vendor contains comma → one extra column
            if len(parts) == expected_fields + 1:
                if has_time_column:
                    # time, device, series, vendor1, vendor2, pci, mem?
                    time_val = parts[0]
                    device = parts[1]
                    series = parts[2]
                    vendor = f'"{parts[3]},{parts[4]}"'
                    rest = parts[5:]
                    fixed_lines.append(",".join([time_val, device, series, vendor] + rest))
                else:
                    # device, series, vendor1, vendor2, pci, mem?
                    device = parts[0]
                    series = parts[1]
                    vendor = f'"{parts[2]},{parts[3]}"'
                    rest = parts[4:]
                    fixed_lines.append(",".join([device, series, vendor] + rest))
                continue

            log_warn("Unexpected CSV format at line {}: {}", idx, line)
            fixed_lines.append(line)

        fixed_csv = "\n".join(fixed_lines)
        reader = csv.DictReader(StringIO(fixed_csv))

        # -------- 3. Header-driven field access (machine-independent) --------
        FIELD_ALIASES = {
            "series": ["Card Series", "Card series"],
            "vendor": ["Card Vendor", "Card vendor"],
            "pci": ["PCI Bus"],
            "mem": ["Available memory size (MiB)"],
        }

        def pick(row, keys, default=None):
            for k in keys:
                if k in row and row[k]:
                    return row[k].strip()
            return default

        # -------- 4. Build meta --------
        for i, row in enumerate(reader):
            if i >= num_gpus:
                break

            card_series = pick(row, FIELD_ALIASES["series"], "N/A")
            card_vendor = pick(row, FIELD_ALIASES["vendor"], "N/A")
            pci_bus_raw = pick(row, FIELD_ALIASES["pci"], None)
            mem_mib = pick(row, FIELD_ALIASES["mem"], None)

            pci_bus_id = pci_bus_raw.split("-->")[0].strip() if pci_bus_raw else None

            pci_domain = pci_bus = pci_device = None
            if pci_bus_id:
                try:
                    domain, bus, devfunc = pci_bus_id.split(":")
                    dev, _ = devfunc.split(".")
                    pci_domain = int(domain, 16)
                    pci_bus = int(bus, 16)
                    pci_device = int(dev, 16)
                except Exception as e:
                    log_warn("PCI parsing failed for '{}': {}", pci_bus_id, e)

            try:
                total_mem = float(mem_mib) / 1024.0 if mem_mib else 0.0
            except Exception:
                total_mem = 0.0

            # PCIe info
            # pcie_gen = pcie_width = pcie_current_gen = pcie_current_width = "N/A"
            # if pci_bus_id:
            #     try:
            #         pcie = self._query_pcie_link_info_from_lspci(pci_bus_id)
            #         pcie_gen = self.get_pcie_gen(pcie.get("max_speed"))
            #         pcie_width = pcie.get("max_width", "N/A")
            #         pcie_current_gen = self.get_pcie_gen(pcie.get("cur_speed"))
            #         pcie_current_width = pcie.get("cur_width", "N/A")
            #     except Exception as e:
            #         log_warn("PCIe info query failed for GPU{}: {}", i, e)
            # PCIe info (from sysfs)
            pcie_gen = pcie_width = pcie_current_gen = pcie_current_width = "N/A"
            if pci_bus_id:
                try:
                    pcie = self._query_pcie_link_info_from_sysfs(pci_bus_id)

                    pcie_gen = self.get_pcie_gen(pcie.get("max_speed"))
                    pcie_width = pcie.get("max_width") or "N/A"

                    pcie_current_gen = self.get_pcie_gen(pcie.get("cur_speed"))
                    pcie_current_width = pcie.get("cur_width") or "N/A"
                except Exception as e:
                    log_warn("PCIe sysfs query failed for GPU{}: {}", i, e)

            name = f"{card_series} ({card_vendor})" if card_vendor != "N/A" else card_series

            meta.append({
                "index": i,
                "name": name,
                "pci_bus_id": pci_bus_id,
                "pci_domain": pci_domain,
                "pci_bus": pci_bus,
                "pci_device": pci_device,
                "max_pcie_gen": pcie_gen,
                "max_pcie_width": pcie_width,
                "current_pcie_gen": pcie_current_gen,
                "current_pcie_width": pcie_current_width,
                "total_memory_gb": round(total_mem, 2),
                "compute_capability": "N/A",
                "source": "hy-smi",
            })

        return meta

    def _parse_mthreads_gmi_for_device_info(self, num_gpus: int, smi: str) -> List[dict[str, Any]]:
        """
        Moore Threads (MUSA) device metadata parser.
        Output schema keys must match the NVIDIA path.
        """
        smi_basename = os.path.basename(smi)

        proc = subprocess.run(
            [smi, "--query"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            log_warn("{} --query failed: {}", smi, proc.stderr)
            return []

        lines = proc.stdout.splitlines()
        header_re = re.compile(r"^GPU(\d+)\s+([0-9a-fA-F:]+\.[0-9])\s*$")

        def _new_meta_entry(idx: int, bdf: str) -> dict[str, Any]:
            pci_domain = pci_bus = pci_device = "N/A"
            try:
                dom_s, bus_s, rest = bdf.split(":")
                dev_s, _fn = rest.split(".")
                pci_domain = int(dom_s, 16)
                pci_bus = int(bus_s, 16)
                pci_device = int(dev_s, 16)
            except Exception:
                pass

            return {
                "index": idx,
                "name": "N/A",
                "pci_bus_id": bdf,
                "pci_domain": pci_domain,
                "pci_bus": pci_bus,
                "pci_device": pci_device,
                "max_pcie_gen": "N/A",
                "max_pcie_width": "N/A",
                "current_pcie_gen": "N/A",
                "current_pcie_width": "N/A",
                "total_memory_gb": "N/A",
                "compute_capability": "N/A",
                "source": smi_basename,
            }

        meta_list: List[dict[str, Any]] = []
        cur: dict[str, Any] | None = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            m = header_re.match(line)
            if m:
                # flush previous
                if cur is not None:
                    pci_bus_id = cur.get("pci_bus_id")
                    if isinstance(pci_bus_id, str) and pci_bus_id not in ("", "N/A"):
                        try:
                            pcie = self._query_pcie_link_info_from_sysfs(pci_bus_id)
                            cur["max_pcie_gen"] = self.get_pcie_gen(pcie.get("max_speed"))
                            cur["max_pcie_width"] = pcie.get("max_width") or cur["max_pcie_width"]
                            cur["current_pcie_gen"] = self.get_pcie_gen(pcie.get("cur_speed"))
                            cur["current_pcie_width"] = pcie.get("cur_width") or cur["current_pcie_width"]
                        except Exception as e:
                            log_warn("PCIe sysfs query failed for GPU{}: {}", cur.get("index"), e)
                    meta_list.append(cur)

                idx = int(m.group(1))
                bdf = m.group(2)
                cur = _new_meta_entry(idx, bdf)
                i += 1
                continue

            if cur is None:
                i += 1
                continue

            # Product Name : ...
            if "Product Name" in line and ":" in line:
                cur["name"] = line.split(":", 1)[1].strip()
                i += 1
                continue

            # FB Memory Usage -> Total : XXXXMiB
            if line.startswith("Total") and ":" in line and "MiB" in line and cur.get("total_memory_gb") == "N/A":
                val = line.split(":", 1)[1].strip()
                m2 = re.match(r"^([0-9.]+)\s*MiB$", val, re.IGNORECASE)
                if m2:
                    try:
                        mib = float(m2.group(1))
                        cur["total_memory_gb"] = round(mib / 1024.0, 2)
                    except Exception:
                        pass
                i += 1
                continue

            # Optional: if mthreads-gmi provides PCIe info, we can prefill; final truth comes from lspci.
            if re.search(r"\bPCIe Generation\b", line, re.IGNORECASE):
                for j in range(i + 1, min(i + 8, len(lines))):
                    lj = lines[j].strip()
                    if lj.startswith("Max") and ":" in lj:
                        cur["max_pcie_gen"] = lj.split(":", 1)[1].strip() or cur["max_pcie_gen"]
                    elif lj.startswith("Current") and ":" in lj:
                        cur["current_pcie_gen"] = lj.split(":", 1)[1].strip() or cur["current_pcie_gen"]
                i += 1
                continue

            if re.search(r"\bLink Width\b", line, re.IGNORECASE) or re.search(r"\bPcie Lane Width\b", line, re.IGNORECASE):
                for j in range(i + 1, min(i + 8, len(lines))):
                    lj = lines[j].strip()
                    if lj.startswith("Max") and ":" in lj:
                        cur["max_pcie_width"] = lj.split(":", 1)[1].strip() or cur["max_pcie_width"]
                    elif lj.startswith("Current") and ":" in lj:
                        cur["current_pcie_width"] = lj.split(":", 1)[1].strip() or cur["current_pcie_width"]
                i += 1
                continue

            i += 1

        if cur is not None:
            pci_bus_id = cur.get("pci_bus_id")
            if isinstance(pci_bus_id, str) and pci_bus_id not in ("", "N/A"):
                try:
                    pcie = self._query_pcie_link_info_from_sysfs(pci_bus_id)
                    cur["max_pcie_gen"] = self.get_pcie_gen(pcie.get("max_speed"))
                    cur["max_pcie_width"] = pcie.get("max_width") or cur["max_pcie_width"]
                    cur["current_pcie_gen"] = self.get_pcie_gen(pcie.get("cur_speed"))
                    cur["current_pcie_width"] = pcie.get("cur_width") or cur["current_pcie_width"]
                except Exception as e:
                    log_warn("PCIe sysfs query failed for GPU{}: {}", cur.get("index"), e)
            meta_list.append(cur)

        meta_list.sort(key=lambda d: d.get("index", 0))
        return meta_list[:num_gpus]
    def _parse_smi_for_device_info(self, num_gpus: int) -> List[dict[str, Any]]:
        """Parse the SMI command to obtain device information"""
        meta: List[dict[str, Any]] = []
        smi = self.api.smi or "nvidia-smi"
        smi_basename = os.path.basename(smi)
        is_ixsmi = smi_basename == "ixsmi"

        # --- MetaX do not support --query-gpu, need to use this ---#
        if smi_basename in ("ht-smi", "mx-smi"):
            return self._parse_metax_smi_for_device_info(num_gpus, smi)
        # --- Hygon do not support --query-gpu, need to use this ---#
        if smi_basename in ("hy-smi", "rocm-smi"):
            return self._parse_hygon_smi_for_device_info(num_gpus, smi)
        # --- Moore Threads (MUSA): mthreads-gmi --query is not csv, parse text ---#
        if smi_basename == "mthreads-gmi":
            return self._parse_mthreads_gmi_for_device_info(num_gpus, smi)

        if is_ixsmi:
            query_fields = "name,memory.total,pci.bus_id,pci.domain,pci.bus,pci.device"
            expected_min_parts = 6
        else:
            query_fields = "name,memory.total,pci.bus_id,pci.domain,pci.bus,pci.device,compute_cap"
            expected_min_parts = 7

        try:
            proc = subprocess.run(
                [smi, f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                log_warn("{} query failed: {}", smi, proc.stderr)
                return meta

            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            for i in range(min(num_gpus, len(lines))):
                parts = [p.strip() for p in lines[i].split(",")]
                if len(parts) < expected_min_parts:
                    continue

                name = parts[0]
                total_mem = float(parts[1]) / 1024.0
                pci_bus_id = parts[2]
                pci_domain = int(parts[3], 16) if parts[3] else "N/A"
                pci_bus = int(parts[4], 16) if parts[4] else "N/A"
                pci_device = int(parts[5], 16) if parts[5] else "N/A"

                if not is_ixsmi and len(parts) >= 7:
                    cc = parts[6] if parts[6] else "N/A"
                else:
                    cc = "N/A"

                pcie_gen = "N/A"
                pcie_width = "N/A"
                pcie_current_gen = "N/A"
                pcie_current_width = "N/A"

                proc_pcie = subprocess.run(
                    [
                        smi, "-i", str(i),
                        "--query-gpu=pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current",
                        "--format=csv,noheader,nounits",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if proc_pcie.returncode == 0:
                    pcie_lines = [line.strip() for line in proc_pcie.stdout.splitlines() if line.strip()]
                    if pcie_lines:
                        pcie_parts = [p.strip() for p in pcie_lines[0].split(",")]
                        if len(pcie_parts) >= 4:
                            pcie_gen = int(pcie_parts[0]) if pcie_parts[0].isdigit() else "N/A"
                            pcie_width = int(pcie_parts[1]) if pcie_parts[1].isdigit() else "N/A"
                            pcie_current_gen = int(pcie_parts[2]) if pcie_parts[2].isdigit() else "N/A"
                            pcie_current_width = int(pcie_parts[3]) if pcie_parts[3].isdigit() else "N/A"

                meta.append({
                    "index": i,
                    "name": name,
                    "pci_bus_id": pci_bus_id,
                    "pci_domain": pci_domain,
                    "pci_bus": pci_bus,
                    "pci_device": pci_device,
                    "max_pcie_gen": pcie_gen,
                    "max_pcie_width": pcie_width,
                    "current_pcie_gen": pcie_current_gen,
                    "current_pcie_width": pcie_current_width,
                    "total_memory_gb": round(total_mem, 2),
                    "compute_capability": cc,
                    "source": smi_basename,
                })
        except Exception as e:
            log_warn("Device info parsing failed: {}", e)

        return meta

    def query_gpu_metadata(self, num_gpus: int) -> List[dict[str, Any]]:
        """Query GPU metadata"""
        meta = self._parse_smi_for_device_info(num_gpus)
        for i in range(num_gpus):
            if i >= len(meta):
                meta.append({
                    "index": i,
                    "name": f"GPU{i} (unknown)",
                    "pci_bus_id": "N/A",
                    "pci_domain": "N/A",
                    "pci_bus": "N/A",
                    "pci_device": "N/A",
                    "max_pcie_gen": "N/A",
                    "max_pcie_width": "N/A",
                    "current_pcie_gen": "N/A",
                    "current_pcie_width": "N/A",
                    "total_memory_gb": "N/A",
                    "compute_capability": "N/A",
                    "source": "unknown",
                })
        log_info("Retrieved device info for {} GPUs", num_gpus)
        return meta

    # -------------------- Memory management --------------------
    def _malloc(self, dev: int, size: int) -> Any:
        """Allocate device memory"""
        self._check_infini(self.api.infiniSetDevice(dev), f"infiniSetDevice({dev})", dev=dev)
        ptr = ctypes.c_void_p()
        self._check_infini(self.api.infiniMalloc(ctypes.byref(ptr), size), "infiniMalloc", dev=dev)
        return ptr

    def _free(self, ptr: Any) -> None:
        """Release device memory"""
        if ptr:
            self._check_infini(self.api.infiniFree(ptr), "infiniFree")

    def _alloc_device_with_fallback(self, dev: int, size: int, fallback: int) -> Tuple[Any, int]:
        """Allocate device memory (with downgrade)"""
        try:
            ptr = self._malloc(dev, size)
            return ptr, size
        except Exception:
            log_warn("GPU {}: allocate {} bytes failed, fallback to {} bytes", dev, size, fallback)
            ptr = self._malloc(dev, fallback)
            return ptr, fallback

    def _alloc_host(self, size: int) -> Any:
        """Allocate memory for pinned host"""
        ptr = ctypes.c_void_p()
        self._check_infini(self.api.infiniHostAlloc(ctypes.byref(ptr), size, cudaHostAllocDefault), "infiniHostAlloc")
        return ptr

    def _free_host(self, ptr: Any) -> None:
        """Release the memory of pinned host"""
        if ptr:
            self._check_infini(self.api.infiniFreeHost(ptr), "infiniFreeHost")

    # -------------------- Streams and events --------------------
    def _create_stream(self, dev: int) -> Any:
        """Create a stream"""
        self._check_infini(self.api.infiniSetDevice(dev), f"infiniSetDevice({dev})", dev=dev)
        stream = ctypes.c_void_p()
        self._check_infini(self.api.infiniStreamCreate(ctypes.byref(stream)), "infiniStreamCreate", dev=dev)
        return stream

    def _destroy_stream(self, stream: Any) -> None:
        """Destruction stream"""
        if stream:
            self._check_infini(self.api.infiniStreamDestroy(stream), "infiniStreamDestroy")

    def _create_event(self) -> Any:
        """Create an event"""
        evt = ctypes.c_void_p()
        self._check_infini(self.api.infiniEventCreate(ctypes.byref(evt)), "infiniEventCreate")
        return evt

    def _destroy_event(self, event: Any) -> None:
        """Destruction event"""
        if event:
            self._check_infini(self.api.infiniEventDestroy(event), "infiniEventDestroy")

    def _with_events(
        self,
        stream: Any,
        repeat: int,
        fn,
        *args,
        warmup: int = 0,
        **kwargs,
    ) -> float:
        """Measure average time (ms) using events, with optional warm-up."""
        evt_s = self._create_event()
        evt_t = self._create_event()

        self._check_infini(self.api.infiniStreamSynchronize(stream), "infiniStreamSynchronize")

        for _ in range(max(warmup, 0)):
            fn(*args, **kwargs)
        self._check_infini(self.api.infiniStreamSynchronize(stream), "infiniStreamSynchronize")

        self._check_infini(self.api.infiniEventRecord(evt_s, stream), "infiniEventRecord(start)")
        for _ in range(repeat):
            fn(*args, **kwargs)
        self._check_infini(self.api.infiniEventRecord(evt_t, stream), "infiniEventRecord(end)")
        self._check_infini(self.api.infiniEventSynchronize(evt_t), "infiniEventSynchronize")

        ms = ctypes.c_float()
        self._check_infini(self.api.infiniEventElapsedTime(ctypes.byref(ms), evt_s, evt_t), "infiniEventElapsedTime")

        self._destroy_event(evt_s)
        self._destroy_event(evt_t)
        return ms.value / max(repeat, 1)

    # -------------------- GPU↔GPU measurement --------------------
    def _peer_memcpy(
        self,
        src_device: int,
        dst_device: int,
        src_ptr: Any,
        dst_ptr: Any,
        size: int,
        stream: Any,
    ) -> None:
        """Perform P2P memory copy"""
        self.api.infiniMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, size, stream)

    def _measure_peer_bandwidth(
        self,
        src_device: int,
        dst_device: int,
        src_ptr: Any,
        dst_ptr: Any,
        buffer_bytes: int,
        stream: Any,
        repeat: int = 100,
        warmup: int = 0,
    ) -> float:
        """Measure unidirectional bandwidth for a single buffer size."""
        def launch_copy():
            self._peer_memcpy(src_device, dst_device, src_ptr, dst_ptr, buffer_bytes, stream)

        elapsed_ms = self._with_events(stream, repeat, launch_copy, warmup=warmup)
        elapsed_s = elapsed_ms / 1000.0
        bw = (buffer_bytes / (1024.0**3)) / elapsed_s if elapsed_s > 0 else float("inf")
        return bw if src_device != dst_device else bw * 2

    def _measure_peer_bandwidth_max(
        self,
        src_device: int,
        dst_device: int,
        src_ptr: Any,
        dst_ptr: Any,
        buffer_bytes: int,
        stream: Any,
        repeat: int,
        ramp_sizes: list[int] | None,
        warmup: int,
    ) -> float:
        """Measure unidirectional bandwidth with ramp sizes and return max."""
        if not ramp_sizes:
            return self._measure_peer_bandwidth(
                src_device, dst_device, src_ptr, dst_ptr, buffer_bytes, stream, repeat, warmup
            )

        max_bw = 0.0
        for sz in ramp_sizes:
            if sz <= 0 or sz > buffer_bytes:
                continue
            bw = self._measure_peer_bandwidth(
                src_device, dst_device, src_ptr, dst_ptr, sz, stream, repeat, warmup
            )
            if bw > max_bw:
                max_bw = bw
        return max_bw


    def _measure_peer_bandwidth_bidi(
        self,
        devA: int,
        devB: int,
        ptrA: Any,
        ptrB: Any,
        buffer_bytes: int,
        streamA: Any,
        streamB: Any,
        repeat: int = 100,
        warmup: int = 0,
    ) -> float:
        """Measure bidirectional P2P bandwidth (safe version)."""

        # Allocate destination buffers
        dstB = self._malloc(devB, buffer_bytes)
        dstA = self._malloc(devA, buffer_bytes)

        # Create events on correct devices
        self._check_infini(self.api.infiniSetDevice(devA))
        evt_go_A = self._create_event()
        evt_startA = self._create_event()
        evt_doneA = self._create_event()

        self._check_infini(self.api.infiniSetDevice(devB))
        evt_go_B = self._create_event()
        evt_startB = self._create_event()
        evt_doneB = self._create_event()

        results = []

        for _ in range(max(warmup, 0)):
            # --- synchronize start gate ---
            self._check_infini(self.api.infiniSetDevice(devA))
            self._check_infini(self.api.infiniEventRecord(evt_go_A, streamA))
            self._check_infini(self.api.infiniStreamWaitEvent(streamA, evt_go_A, 0))

            self._check_infini(self.api.infiniSetDevice(devB))
            self._check_infini(self.api.infiniEventRecord(evt_go_B, streamB))
            self._check_infini(self.api.infiniStreamWaitEvent(streamB, evt_go_B, 0))

            # --- launch async transfers ---
            self._peer_memcpy(devA, devB, ptrA, dstB, buffer_bytes, streamB)
            self._peer_memcpy(devB, devA, ptrB, dstA, buffer_bytes, streamA)

            self._check_infini(self.api.infiniSetDevice(devA))
            self._check_infini(self.api.infiniStreamSynchronize(streamA))
            self._check_infini(self.api.infiniSetDevice(devB))
            self._check_infini(self.api.infiniStreamSynchronize(streamB))

        for _ in range(repeat):

            # --- synchronize start gate ---
            self._check_infini(self.api.infiniSetDevice(devA))
            self._check_infini(self.api.infiniEventRecord(evt_go_A, streamA))
            self._check_infini(self.api.infiniStreamWaitEvent(streamA, evt_go_A, 0))

            self._check_infini(self.api.infiniSetDevice(devB))
            self._check_infini(self.api.infiniEventRecord(evt_go_B, streamB))
            self._check_infini(self.api.infiniStreamWaitEvent(streamB, evt_go_B, 0))

            # --- record start events ---
            self._check_infini(self.api.infiniSetDevice(devA))
            self._check_infini(self.api.infiniEventRecord(evt_startA, streamA))

            self._check_infini(self.api.infiniSetDevice(devB))
            self._check_infini(self.api.infiniEventRecord(evt_startB, streamB))

            # --- launch async transfers ---
            self._peer_memcpy(devA, devB, ptrA, dstB, buffer_bytes, streamB)
            self._peer_memcpy(devB, devA, ptrB, dstA, buffer_bytes, streamA)

            # --- record done events ---
            self._check_infini(self.api.infiniSetDevice(devA))
            self._check_infini(self.api.infiniEventRecord(evt_doneA, streamA))

            self._check_infini(self.api.infiniSetDevice(devB))
            self._check_infini(self.api.infiniEventRecord(evt_doneB, streamB))

            # --- wait ---
            self._check_infini(self.api.infiniSetDevice(devA))
            self._check_infini(self.api.infiniStreamSynchronize(streamA))

            self._check_infini(self.api.infiniSetDevice(devB))
            self._check_infini(self.api.infiniStreamSynchronize(streamB))

            # --- compute elapsed time ---
            msA = ctypes.c_float()
            msB = ctypes.c_float()

            self._check_infini(self.api.infiniSetDevice(devA))
            self._check_infini(self.api.infiniEventElapsedTime(ctypes.byref(msA), evt_startA, evt_doneA))

            self._check_infini(self.api.infiniSetDevice(devB))
            self._check_infini(self.api.infiniEventElapsedTime(ctypes.byref(msB), evt_startB, evt_doneB))

            elapsed = max(msA.value, msB.value) / 1000.0
            if elapsed > 0:
                bw = (2 * buffer_bytes / (1024**3)) / elapsed
                results.append(bw)

        # Cleanup
        self._check_infini(self.api.infiniSetDevice(devA))
        self._destroy_event(evt_go_A)
        self._destroy_event(evt_startA)
        self._destroy_event(evt_doneA)

        self._check_infini(self.api.infiniSetDevice(devB))
        self._destroy_event(evt_go_B)
        self._destroy_event(evt_startB)
        self._destroy_event(evt_doneB)

        self._free(dstA)
        self._free(dstB)

        if not results:
            return 0.0

        return float(np.mean(results))


    def _measure_peer_latency(
        self,
        src_device: int,
        dst_device: int,
        src_ptr: Any,
        dst_ptr: Any,
        stream: Any,
        repeat: int = 100,
        warmup: int = 0,
    ) -> float:
        """Measurement delay (microseconds)"""
        LAT_BYTES = 16

        def launch_copy():
            self._peer_memcpy(src_device, dst_device, src_ptr, dst_ptr, LAT_BYTES, stream)

        elapsed_ms = self._with_events(stream, repeat, launch_copy, warmup=warmup)
        return elapsed_ms * 1000.0  # ms -> us

    def measure_gpu_to_gpu(
        self,
        num_gpus: int,
        buffer_bytes: int,
        bidirectional: bool,
        skip_self: bool,
        description: str,
        repeat: int,
        ramp_sizes: list[int] | None = None,
        warmup: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        """
        Measure the bandwidth and latency between Gpus
        Return (bw_uni_read, bw_uni_write, bw_bi, lat_read, lat_write)
        """
        log_info("Starting GPU-GPU measurement: {} ({} repetitions)", description, repeat)

        bw_uni_write = np.zeros((num_gpus, num_gpus), dtype=float)
        bw_uni_read = np.zeros((num_gpus, num_gpus), dtype=float)
        lat_uni_write = np.zeros((num_gpus, num_gpus), dtype=float)
        lat_uni_read = np.zeros((num_gpus, num_gpus), dtype=float)
        bw_bi = np.zeros((num_gpus, num_gpus), dtype=float) if bidirectional else None

        src_ptrs: list[Any] = []
        streams: list[Any] = []
        for i in range(num_gpus):
            src_ptrs.append(self._malloc(i, buffer_bytes))
            streams.append(self._create_stream(i))

        for i in range(num_gpus):
            for j in range(num_gpus):
                if skip_self and i == j:
                    bw_uni_write[i, j] = float("nan")
                    bw_uni_read[i, j] = float("nan")
                    lat_uni_write[i, j] = float("nan")
                    lat_uni_read[i, j] = float("nan")
                    if bw_bi is not None:
                        bw_bi[i, j] = float("nan")
                    continue

                dst = self._malloc(j, buffer_bytes)
                stream_j = streams[j]

                lat_uni_read[i, j] = self._measure_peer_latency(
                    i, j, src_ptrs[i], dst, stream_j, warmup=warmup
                )
                log_debug("{} p2p read {}->{} latency: {:.2f} us", description, i, j, lat_uni_read[i, j])

                lat_uni_write[i, j] = self._measure_peer_latency(
                    j, i, dst, src_ptrs[i], stream_j, warmup=warmup
                )
                log_debug("{} p2p write {}->{} latency: {:.2f} us", description, i, j, lat_uni_write[i, j])

                bw_uni_read[i, j] = self._measure_peer_bandwidth_max(
                    i, j, src_ptrs[i], dst, buffer_bytes, stream_j, repeat, ramp_sizes, warmup
                )
                log_debug("{} {}->{} unidirectional bandwidth (read) mean: {:.2f} GB/s",
                        description, i, j, bw_uni_read[i, j])

                bw_uni_write[i, j] = self._measure_peer_bandwidth_max(
                    j, i, dst, src_ptrs[i], buffer_bytes, stream_j, repeat, ramp_sizes, warmup
                )
                log_debug("{} {}->{} unidirectional bandwidth (write) mean: {:.2f} GB/s",
                        description, i, j, bw_uni_write[i, j])

                if bidirectional and bw_bi is not None:
                   stream_i = streams[i]
                   if ramp_sizes:
                       max_bw = 0.0
                       for sz in ramp_sizes:
                           if sz <= 0 or sz > buffer_bytes:
                               continue
                           bw = self._measure_peer_bandwidth_bidi(
                               i, j, src_ptrs[i], src_ptrs[j], sz, stream_i, stream_j, repeat, warmup
                           )
                           if bw > max_bw:
                               max_bw = bw
                       bw_bi[i, j] = max_bw
                   else:
                       bw_bi[i, j] = self._measure_peer_bandwidth_bidi(
                           i, j, src_ptrs[i], src_ptrs[j], buffer_bytes, stream_i, stream_j, repeat, warmup
                       )
                   log_debug("{} {}<->{} bidirectional bandwidth mean: {:.2f} GB/s",
                            description, i, j, bw_bi[i, j])

                self._free(dst)

        for ptr in src_ptrs:
            self._free(ptr)
        for stream in streams:
            self._destroy_stream(stream)

        log_info("Completed GPU-GPU measurement: {}", description)
        return bw_uni_read, bw_uni_write, bw_bi, lat_uni_read, lat_uni_write

    # -------------------- GPU↔Host measurement --------------------
    def _measure_host_bandwidth(
        self,
        device: int,
        dev_ptr: Any,
        host_ptr: Any,
        buffer_bytes: int,
        stream: Any,
        direction: int,
        repeat: int = 100,
        warmup: int = 0,
    ) -> float:
        """Measure the Host-GPU bandwidth for a single buffer size."""
        def launch_copy():
            self.api.infiniMemcpyAsync(dev_ptr if direction == cudaMemcpyHostToDevice else host_ptr,
                                        host_ptr if direction == cudaMemcpyHostToDevice else dev_ptr,
                                        buffer_bytes, direction, stream)

        elapsed_ms = self._with_events(stream, repeat, launch_copy, warmup=warmup)
        elapsed_s = elapsed_ms / 1000.0
        return (buffer_bytes / (1024.0**3)) / elapsed_s if elapsed_s > 0 else float("inf")

    def _measure_host_bandwidth_max(
        self,
        device: int,
        dev_ptr: Any,
        host_ptr: Any,
        buffer_bytes: int,
        stream: Any,
        direction: int,
        repeat: int,
        ramp_sizes: list[int] | None,
        warmup: int,
    ) -> float:
        """Measure Host-GPU bandwidth with ramp sizes and return max."""
        if not ramp_sizes:
            return self._measure_host_bandwidth(
                device, dev_ptr, host_ptr, buffer_bytes, stream, direction, repeat, warmup
            )

        max_bw = 0.0
        for sz in ramp_sizes:
            if sz <= 0 or sz > buffer_bytes:
                continue
            bw = self._measure_host_bandwidth(
                device, dev_ptr, host_ptr, sz, stream, direction, repeat, warmup
            )
            if bw > max_bw:
                max_bw = bw
        return max_bw

    def measure_gpu_to_host(
        self,
        num_gpus: int,
        buffer_bytes: int,
        fallback_bytes: int,
        repeat: int,
        latency_iters: int = 10,
        ramp_sizes: list[int] | None = None,
        warmup: int = 0,
    ):
        """
        Measure the Host-GPU bandwidth：
        pageable_h2d, pageable_d2h,
        pinned_h2d, pinned_d2h,
        lat_pageable_h2d, lat_pageable_d2h,
        lat_pinned_h2d, lat_pinned_d2h
        """

        log_info(
            "Starting GPU↔Host measurement: buffer={}B fallback={}B repeat={}",
            buffer_bytes, fallback_bytes, repeat
        )

        # Eight output arrays
        pageable_h2d = np.zeros(num_gpus, float)
        pageable_d2h = np.zeros(num_gpus, float)
        pinned_h2d   = np.zeros(num_gpus, float)
        pinned_d2h   = np.zeros(num_gpus, float)

        lat_pageable_h2d = np.zeros(num_gpus, float)
        lat_pageable_d2h = np.zeros(num_gpus, float)
        lat_pinned_h2d   = np.zeros(num_gpus, float)
        lat_pinned_d2h   = np.zeros(num_gpus, float)

        # Utility function: Delay Measurement (using 16B messages)
        def _measure_latency(dev: int, dev_ptr: Any, host_ptr: Any, direction: int):
            LAT_BYTES = 16

            stream = self._create_stream(dev)
            def _copy():
                self.api.infiniMemcpyAsync(
                    dev_ptr if direction == cudaMemcpyHostToDevice else host_ptr,
                    host_ptr if direction == cudaMemcpyHostToDevice else dev_ptr,
                    LAT_BYTES,
                    direction,
                    stream,
                )
            avg_ms = self._with_events(stream, latency_iters, _copy, warmup=warmup)
            self._destroy_stream(stream)
            return avg_ms * 1000.0  # ms→us

        # 1) pageable host (buffer of ordinary malloc)
        for i in range(num_gpus):
            dev_ptr, used_bytes = self._alloc_device_with_fallback(i, buffer_bytes, fallback_bytes)
            stream = self._create_stream(i)

            # Use Python's create_string_buffer
            host_buf = ctypes.create_string_buffer(used_bytes)
            host_ptr = ctypes.cast(host_buf, ctypes.c_void_p)

            # Use Python's create_string_buffer
            pageable_h2d[i] = self._measure_host_bandwidth_max(
                i, dev_ptr, host_ptr, used_bytes, stream, cudaMemcpyHostToDevice, repeat, ramp_sizes, warmup
            )
            pageable_d2h[i] = self._measure_host_bandwidth_max(
                i, dev_ptr, host_ptr, used_bytes, stream, cudaMemcpyDeviceToHost, repeat, ramp_sizes, warmup
            )

            # Delay measurement
            lat_pageable_h2d[i] = _measure_latency(i, dev_ptr, host_ptr, cudaMemcpyHostToDevice)
            lat_pageable_d2h[i] = _measure_latency(i, dev_ptr, host_ptr, cudaMemcpyDeviceToHost)

            # clean
            self._destroy_stream(stream)
            self._free(dev_ptr)

        # 2) pinned host（use CUDA HostAlloc）
        for i in range(num_gpus):
            dev_ptr, used_bytes = self._alloc_device_with_fallback(i, buffer_bytes, fallback_bytes)
            stream = self._create_stream(i)

            host_ptr = self._alloc_host(used_bytes)  # pinned host

            # Bandwidth measurement
            pinned_h2d[i] = self._measure_host_bandwidth_max(
                i, dev_ptr, host_ptr, used_bytes, stream, cudaMemcpyHostToDevice, repeat, ramp_sizes, warmup
            )
            pinned_d2h[i] = self._measure_host_bandwidth_max(
                i, dev_ptr, host_ptr, used_bytes, stream, cudaMemcpyDeviceToHost, repeat, ramp_sizes, warmup
            )

            # Delay measurement
            lat_pinned_h2d[i] = _measure_latency(i, dev_ptr, host_ptr, cudaMemcpyHostToDevice)
            lat_pinned_d2h[i] = _measure_latency(i, dev_ptr, host_ptr, cudaMemcpyDeviceToHost)

            # clean
            self._destroy_stream(stream)
            self._free(dev_ptr)
            self._free_host(host_ptr)

        log_info("Completed GPU↔Host measurements")

        return (
            pageable_h2d, pageable_d2h,
            pinned_h2d,   pinned_d2h,
            lat_pageable_h2d, lat_pageable_d2h,
            lat_pinned_h2d,   lat_pinned_d2h,
        )
