#!/usr/bin/env python3
"""
The implementation of HygonApi on the Hygon platform
Load libcudart.so via ctypes.CDLL
Bind the infiniXxx function to the cudaXxx function
"""

from __future__ import annotations
from typing import Any
import ctypes
import os

from infiniAPI import InfiniApi


class HygonApi(InfiniApi):
    """NVIDIA CUDA Platform API Implementation"""
    
    smi = "hy-smi" #还有rocm-smi，服务器上这两个命令都可以，云端好像不可
    
    def __init__(self):
        try:
            self._libcudart = ctypes.CDLL("libgalaxyhip.so")#实际上是hip库
        except OSError:
            raise RuntimeError(
                "Failed to load libgalaxyhip.so. "
                "Make sure HIP is installed and LD_LIBRARY_PATH is set."
            )

    # ------------ Device Management ------------
    def infiniGetDeviceCount(self, count: Any) -> int:
        return self._libcudart.hipGetDeviceCount(count)

    def infiniSetDevice(self, device: int) -> int:
        return self._libcudart.hipSetDevice(device)

    def infiniDeviceReset(self) -> int:
        return self._libcudart.hipDeviceReset()

    def infiniGetErrorString(self, error: int) -> bytes:
        self._libcudart.hipGetErrorString.restype = ctypes.c_char_p
        self._libcudart.hipGetErrorString.argtypes = [ctypes.c_int]
        return self._libcudart.hipGetErrorString(error)

    # ------------ P2P capability ------------
    def infiniDeviceCanAccessPeer(self, can_access: Any, device: int, peer_device: int) -> int:
        return self._libcudart.hipDeviceCanAccessPeer(can_access, device, peer_device)

    def infiniDeviceEnablePeerAccess(self, peer_device: int, flags: int) -> int:
        return self._libcudart.hipDeviceEnablePeerAccess(peer_device, flags)

    def infiniDeviceDisablePeerAccess(self, peer_device: int) -> int:
        return self._libcudart.hipDeviceDisablePeerAccess(peer_device)

    # ------------ Device memory management ------------
    def infiniMalloc(self, dev_ptr: Any, size: int) -> int:
        return self._libcudart.hipMalloc(dev_ptr, size)

    def infiniFree(self, dev_ptr: Any) -> int:
        return self._libcudart.hipFree(dev_ptr)

    # ------------ Pinned page memory ------------
    def infiniHostAlloc(self, ptr: Any, size: int, flags: int) -> int:
        return self._libcudart.hipHostAlloc(ptr, size, flags)

    def infiniFreeHost(self, ptr: Any) -> int:
        return self._libcudart.hipFreeHost(ptr)

    # ------------ Stream management ------------
    def infiniStreamCreate(self, p_stream: Any) -> int:
        return self._libcudart.hipStreamCreate(p_stream)

    def infiniStreamDestroy(self, stream: Any) -> int:
        return self._libcudart.hipStreamDestroy(stream)

    def infiniStreamWaitEvent(self, stream: Any, event: Any, flags: int) -> int:
        return self._libcudart.hipStreamWaitEvent(stream, event, flags)

    def infiniStreamSynchronize(self, stream: Any) -> int:
        return self._libcudart.hipStreamSynchronize(stream)

    # ------------ Memory copy ------------
    def infiniMemcpyPeerAsync(
        self, dst: Any, dst_device: int, src: Any, src_device: int, count: int, stream: Any
    ) -> int:
        return self._libcudart.hipMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream)

    def infiniMemcpyAsync(self, dst: Any, src: Any, count: int, kind: int, stream: Any) -> int:
        return self._libcudart.hipMemcpyAsync(dst, src, count, kind, stream)#hipMemcpyAsync方向和英伟达不一样

    # ------------ Event Management ------------
    def infiniEventCreate(self, event: Any) -> int:
        return self._libcudart.hipEventCreate(event)

    def infiniEventRecord(self, event: Any, stream: Any) -> int:
        return self._libcudart.hipEventRecord(event, stream)

    def infiniEventSynchronize(self, event: Any) -> int:
        return self._libcudart.hipEventSynchronize(event)

    def infiniEventElapsedTime(self, ms: Any, start: Any, end: Any) -> int:
        return self._libcudart.hipEventElapsedTime(ms, start, end)

    def infiniEventDestroy(self, event: Any) -> int:
        return self._libcudart.hipEventDestroy(event)


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

        lib = ctypes.CDLL("libgalaxyhip.so")#libamdhip64.so
        info = Dl_info()
        sym_addr = ctypes.cast(lib.hipGetDeviceCount, ctypes.c_void_p)#查找hipGetDeviceCount符号来自哪个库
        
        if dladdr(sym_addr, ctypes.byref(info)) != 0 and info.dli_fname:
            path = info.dli_fname.decode("utf-8", errors="ignore")
            if os.path.isabs(path):
                return path
    except Exception:
        pass

    # Method 2: /proc/self/maps
    try:
        with open("/proc/self/maps", "r") as f:#找linux下每个进程/proc/self/maps映射了哪些共享库文件
            for line in f:
                if "amdhip64" not in line:
                    continue
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cand = parts[-1]
                if cand == "(deleted)" and len(parts) >= 7:
                    cand = parts[-2]
                if os.path.isabs(cand) and "amdhip64" in cand:
                    return cand
    except Exception:
        pass

    # Method 3: Return the short name
    return "libgalaxyhip.so"#libamdhip64.so
