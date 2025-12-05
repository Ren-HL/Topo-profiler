#!/usr/bin/env python3
"""
The implementation of InfiniApi on the NVIDIA platform
Load libcudart.so via ctypes.CDLL
Bind the infiniXxx function to the cudaXxx function
"""

from __future__ import annotations
from typing import Any
import ctypes
import os

from infiniAPI import InfiniApi


class NvidiaApi(InfiniApi):
    """NVIDIA CUDA Platform API Implementation"""
    
    smi = "nvidia-smi"
    
    def __init__(self):
        try:
            self._libcudart = ctypes.CDLL("libcudart.so")
        except OSError:
            raise RuntimeError(
                "Failed to load libcudart.so. "
                "Make sure CUDA is installed and LD_LIBRARY_PATH is set."
            )

    # ------------ Device Management ------------
    def infiniGetDeviceCount(self, count: Any) -> int:
        return self._libcudart.cudaGetDeviceCount(count)

    def infiniSetDevice(self, device: int) -> int:
        return self._libcudart.cudaSetDevice(device)

    def infiniDeviceReset(self) -> int:
        return self._libcudart.cudaDeviceReset()

    def infiniGetErrorString(self, error: int) -> bytes:
        self._libcudart.cudaGetErrorString.restype = ctypes.c_char_p
        self._libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
        return self._libcudart.cudaGetErrorString(error)

    # ------------ P2P capability ------------
    def infiniDeviceCanAccessPeer(self, can_access: Any, device: int, peer_device: int) -> int:
        return self._libcudart.cudaDeviceCanAccessPeer(can_access, device, peer_device)

    def infiniDeviceEnablePeerAccess(self, peer_device: int, flags: int) -> int:
        return self._libcudart.cudaDeviceEnablePeerAccess(peer_device, flags)

    def infiniDeviceDisablePeerAccess(self, peer_device: int) -> int:
        return self._libcudart.cudaDeviceDisablePeerAccess(peer_device)

    # ------------ Device memory management ------------
    def infiniMalloc(self, dev_ptr: Any, size: int) -> int:
        return self._libcudart.cudaMalloc(dev_ptr, size)

    def infiniFree(self, dev_ptr: Any) -> int:
        return self._libcudart.cudaFree(dev_ptr)

    # ------------ Pinned page memory ------------
    def infiniHostAlloc(self, ptr: Any, size: int, flags: int) -> int:
        return self._libcudart.cudaHostAlloc(ptr, size, flags)

    def infiniFreeHost(self, ptr: Any) -> int:
        return self._libcudart.cudaFreeHost(ptr)

    # ------------ Stream management ------------
    def infiniStreamCreate(self, p_stream: Any) -> int:
        return self._libcudart.cudaStreamCreate(p_stream)

    def infiniStreamDestroy(self, stream: Any) -> int:
        return self._libcudart.cudaStreamDestroy(stream)

    def infiniStreamWaitEvent(self, stream: Any, event: Any, flags: int) -> int:
        return self._libcudart.cudaStreamWaitEvent(stream, event, flags)

    def infiniStreamSynchronize(self, stream: Any) -> int:
        return self._libcudart.cudaStreamSynchronize(stream)

    # ------------ Memory copy ------------
    def infiniMemcpyPeerAsync(
        self, dst: Any, dst_device: int, src: Any, src_device: int, count: int, stream: Any
    ) -> int:
        return self._libcudart.cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream)

    def infiniMemcpyAsync(self, dst: Any, src: Any, count: int, kind: int, stream: Any) -> int:
        return self._libcudart.cudaMemcpyAsync(dst, src, count, kind, stream)

    # ------------ Event Management ------------
    def infiniEventCreate(self, event: Any) -> int:
        return self._libcudart.cudaEventCreate(event)

    def infiniEventRecord(self, event: Any, stream: Any) -> int:
        return self._libcudart.cudaEventRecord(event, stream)

    def infiniEventSynchronize(self, event: Any) -> int:
        return self._libcudart.cudaEventSynchronize(event)

    def infiniEventElapsedTime(self, ms: Any, start: Any, end: Any) -> int:
        return self._libcudart.cudaEventElapsedTime(ms, start, end)

    def infiniEventDestroy(self, event: Any) -> int:
        return self._libcudart.cudaEventDestroy(event)


def get_libcudart_path() -> str:
    """Return to the absolute path of libcudart.so"""
    
    # Method 1: dladdr
    try:
        class Dl_info(ctypes.Structure):
            _fields_ = [
                ("dli_fname", ctypes.c_char_p),
                ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p),
                ("dli_saddr", ctypes.c_void_p),
            ]

        libdl = ctypes.CDLL("libdl.so")
        dladdr = libdl.dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Dl_info)]
        dladdr.restype = ctypes.c_int

        lib = ctypes.CDLL("libcudart.so")
        info = Dl_info()
        sym_addr = ctypes.cast(lib.cudaGetDeviceCount, ctypes.c_void_p)
        
        if dladdr(sym_addr, ctypes.byref(info)) != 0 and info.dli_fname:
            path = info.dli_fname.decode("utf-8", errors="ignore")
            if os.path.isabs(path):
                return path
    except Exception:
        pass

    # Method 2: /proc/self/maps
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                if "cudart" not in line:
                    continue
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cand = parts[-1]
                if cand == "(deleted)" and len(parts) >= 7:
                    cand = parts[-2]
                if os.path.isabs(cand) and "cudart" in cand:
                    return cand
    except Exception:
        pass

    # Method 3: Return the short name
    return "libcudart.so"
