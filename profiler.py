#!/usr/bin/env python3
"""
GPU Profiler class - Platform-independent implementation
- Only relies on the InfiniApi abstract interface
- All CUDA API calls have been changed to infiniXxx
- Includes all measurement and query functions
"""

from __future__ import annotations
from typing import Any, Tuple, List
import ctypes
import subprocess
import re
import os
import sys

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
    def get_native_topology_str(self) -> str:
        """Obtain native topology information (using smi topo-m)"""
        smi = self.api.smi or "nvidia-smi"
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
        return [gpu_basic[i] for i in sorted(gpu_basic)]

    def _parse_smi_for_device_info(self, num_gpus: int) -> List[dict[str, Any]]:
        """Parse the SMI command to obtain device information"""
        meta: List[dict[str, Any]] = []
        smi = self.api.smi or "nvidia-smi"
        smi_basename = os.path.basename(smi)
        is_ixsmi = smi_basename == "ixsmi"

        # --- MetaX do not support --query-gpu, need to use this ---#
        if smi_basename in ("ht-smi", "mx-smi"):
            return self._parse_metax_smi_for_device_info(num_gpus, smi)

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

    def _with_events(self, stream: Any, repeat: int, fn, *args, **kwargs) -> float:
        """Measure the average time consumption (in milliseconds) using events"""
        evt_s = self._create_event()
        evt_t = self._create_event()

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
    ) -> float:
        """Measuring Unidirectional bandwidth"""
        def launch_copy():
            self._peer_memcpy(src_device, dst_device, src_ptr, dst_ptr, buffer_bytes, stream)

        elapsed_ms = self._with_events(stream, repeat, launch_copy)
        elapsed_s = elapsed_ms / 1000.0
        bw = (buffer_bytes / (1024.0**3)) / elapsed_s if elapsed_s > 0 else float("inf")
        return bw if src_device != dst_device else bw * 2


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
    ) -> float:
        """Measure bidirectional P2P bandwidth (safe version)"""

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
    ) -> float:
        """Measurement delay (microseconds)"""
        LAT_BYTES = 16

        def launch_copy():
            self._peer_memcpy(src_device, dst_device, src_ptr, dst_ptr, LAT_BYTES, stream)

        elapsed_ms = self._with_events(stream, repeat, launch_copy)
        return elapsed_ms * 1000.0  # ms -> us

    def measure_gpu_to_gpu(
        self,
        num_gpus: int,
        buffer_bytes: int,
        bidirectional: bool,
        skip_self: bool,
        description: str,
        repeat: int,
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

                lat_uni_read[i, j] = self._measure_peer_latency(i, j, src_ptrs[i], dst, stream_j)
                log_debug("{} p2p read {}->{} latency: {:.2f} us", description, i, j, lat_uni_read[i, j])

                lat_uni_write[i, j] = self._measure_peer_latency(j, i, dst, src_ptrs[i], stream_j)
                log_debug("{} p2p write {}->{} latency: {:.2f} us", description, i, j, lat_uni_write[i, j])

                bw_uni_read[i, j] = self._measure_peer_bandwidth(i, j, src_ptrs[i], dst, buffer_bytes, stream_j, repeat)
                log_debug("{} {}->{} unidirectional bandwidth (read) mean: {:.2f} GB/s",
                        description, i, j, bw_uni_read[i, j])

                bw_uni_write[i, j] = self._measure_peer_bandwidth(j, i, dst, src_ptrs[i], buffer_bytes, stream_j, repeat)
                log_debug("{} {}->{} unidirectional bandwidth (write) mean: {:.2f} GB/s",
                        description, i, j, bw_uni_write[i, j])

                if bidirectional and bw_bi is not None:
                   stream_i = streams[i]
                   bw_bi[i, j] = self._measure_peer_bandwidth_bidi(i, j, src_ptrs[i], src_ptrs[j], buffer_bytes, stream_i, stream_j, repeat)
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
    ) -> float:
        """Measure the Host-GPU bandwidth"""
        def launch_copy():
            self.api.infiniMemcpyAsync(dev_ptr if direction == cudaMemcpyHostToDevice else host_ptr,
                                        host_ptr if direction == cudaMemcpyHostToDevice else dev_ptr,
                                        buffer_bytes, direction, stream)

        elapsed_ms = self._with_events(stream, repeat, launch_copy)
        elapsed_s = elapsed_ms / 1000.0
        return (buffer_bytes / (1024.0**3)) / elapsed_s if elapsed_s > 0 else float("inf")

    def measure_gpu_to_host(
        self,
        num_gpus: int,
        buffer_bytes: int,
        fallback_bytes: int,
        repeat: int,
        latency_iters: int = 10,
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
            avg_ms = self._with_events(stream, latency_iters, _copy)
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
            pageable_h2d[i] = self._measure_host_bandwidth(
                i, dev_ptr, host_ptr, used_bytes, stream, cudaMemcpyHostToDevice, repeat
            )
            pageable_d2h[i] = self._measure_host_bandwidth(
                i, dev_ptr, host_ptr, used_bytes, stream, cudaMemcpyDeviceToHost, repeat
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
            pinned_h2d[i] = self._measure_host_bandwidth(
                i, dev_ptr, host_ptr, used_bytes, stream, cudaMemcpyHostToDevice, repeat
            )
            pinned_d2h[i] = self._measure_host_bandwidth(
                i, dev_ptr, host_ptr, used_bytes, stream, cudaMemcpyDeviceToHost, repeat
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
