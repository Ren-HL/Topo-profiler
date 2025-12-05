#!/usr/bin/env python3
"""
The implementation of InfiniApi on the MetaX platform
Load libhcruntime.so via ctypes.CDLL
Bind the infiniXxx function to the hcXxx function
"""

from __future__ import annotations
from typing import Any
import ctypes
import os

from infiniAPI import InfiniApi


class HpccApi(InfiniApi):
    """METAX HPCC Platform API Implementation"""
    
    smi = "ht-smi"
    
    def __init__(self):
        try:
            self._libcudart = ctypes.CDLL("libmcruntime.so")
        except OSError:
            raise RuntimeError(
                "Failed to load libmcruntime.so. "
                "Make sure HPCC is installed and LD_LIBRARY_PATH is set."
            )

    # ------------ Device Management ------------
    def infiniGetDeviceCount(self, count: Any) -> int:
        return self._libcudart.hcGetDeviceCount(count)

    def infiniSetDevice(self, device: int) -> int:
        return self._libcudart.hcSetDevice(device)

    def infiniDeviceReset(self) -> int:
        return self._libcudart.hcDeviceReset()

    def infiniGetErrorString(self, error: int) -> bytes:
        self._libcudart.hcGetErrorString.restype = ctypes.c_char_p
        self._libcudart.hcGetErrorString.argtypes = [ctypes.c_int]
        return self._libcudart.hcGetErrorString(error)

    # ------------ P2P capability ------------
    def infiniDeviceCanAccessPeer(self, can_access: Any, device: int, peer_device: int) -> int:
        return self._libcudart.hcDeviceCanAccessPeer(can_access, device, peer_device)

    def infiniDeviceEnablePeerAccess(self, peer_device: int, flags: int) -> int:
        return self._libcudart.hcDeviceEnablePeerAccess(peer_device, flags)

    def infiniDeviceDisablePeerAccess(self, peer_device: int) -> int:
        return self._libcudart.hcDeviceDisablePeerAccess(peer_device)

    # ------------ Device memory management ------------
    def infiniMalloc(self, dev_ptr: Any, size: int) -> int:
        return self._libcudart.hcMalloc(dev_ptr, size)

    def infiniFree(self, dev_ptr: Any) -> int:
        return self._libcudart.hcFree(dev_ptr)

    # ------------ Pinned page memory ------------
    def infiniHostAlloc(self, ptr: Any, size: int, flags: int) -> int:
        return self._libcudart.hcMallocHost(ptr, size, flags)

    def infiniFreeHost(self, ptr: Any) -> int:
        return self._libcudart.hcFreeHost(ptr)

    # ------------ Stream management ------------
    def infiniStreamCreate(self, p_stream: Any) -> int:
        return self._libcudart.hcStreamCreate(p_stream)

    def infiniStreamDestroy(self, stream: Any) -> int:
        return self._libcudart.hcStreamDestroy(stream)

    def infiniStreamWaitEvent(self, stream: Any, event: Any, flags: int) -> int:
        return self._libcudart.hcStreamWaitEvent(stream, event, flags)

    def infiniStreamSynchronize(self, stream: Any) -> int:
        return self._libcudart.hcStreamSynchronize(stream)

    # ------------ Memory copy ------------
    def infiniMemcpyPeerAsync(
        self, dst: Any, dst_device: int, src: Any, src_device: int, count: int, stream: Any
    ) -> int:
        return self._libcudart.hcMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream)

    def infiniMemcpyAsync(self, dst: Any, src: Any, count: int, kind: int, stream: Any) -> int:
        return self._libcudart.hcMemcpyAsync(dst, src, count, kind, stream)

    # ------------ Event Management ------------
    def infiniEventCreate(self, event: Any) -> int:
        return self._libcudart.hcEventCreate(event)

    def infiniEventRecord(self, event: Any, stream: Any) -> int:
        return self._libcudart.hcEventRecord(event, stream)

    def infiniEventSynchronize(self, event: Any) -> int:
        return self._libcudart.hcEventSynchronize(event)

    def infiniEventElapsedTime(self, ms: Any, start: Any, end: Any) -> int:
        return self._libcudart.hcEventElapsedTime(ms, start, end)

    def infiniEventDestroy(self, event: Any) -> int:
        return self._libcudart.hcEventDestroy(event)



