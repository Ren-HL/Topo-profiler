#!/usr/bin/env python3
"""
The implementation of InfiniApi on the MetaX platform
Load libhcruntime.so via ctypes.CDLL
Bind the infiniXxx function to the mcXxx function
"""

from __future__ import annotations
from typing import Any
import ctypes
import os

from infiniAPI import InfiniApi


class MacaApi(InfiniApi):
    """METAX Maca Platform API Implementation"""
    
    smi = "mx-smi"
    
    def __init__(self):
        try:
            self._libcudart = ctypes.CDLL("libmcruntime.so")
        except OSError:
            raise RuntimeError(
                "Failed to load libmcruntime.so. "
                "Make sure Maca is installed and LD_LIBRARY_PATH is set."
            )

    # ------------ Device Management ------------
    def infiniGetDeviceCount(self, count: Any) -> int:
        return self._libcudart.mcGetDeviceCount(count)

    def infiniSetDevice(self, device: int) -> int:
        return self._libcudart.mcSetDevice(device)

    def infiniDeviceReset(self) -> int:
        return self._libcudart.mcDeviceReset()

    def infiniGetErrorString(self, error: int) -> bytes:
        self._libcudart.mcGetErrorString.restype = ctypes.c_char_p
        self._libcudart.mcGetErrorString.argtypes = [ctypes.c_int]
        return self._libcudart.mcGetErrorString(error)

    # ------------ P2P capability ------------
    def infiniDeviceCanAccessPeer(self, can_access: Any, device: int, peer_device: int) -> int:
        return self._libcudart.mcDeviceCanAccessPeer(can_access, device, peer_device)

    def infiniDeviceEnablePeerAccess(self, peer_device: int, flags: int) -> int:
        return self._libcudart.mcDeviceEnablePeerAccess(peer_device, flags)

    def infiniDeviceDisablePeerAccess(self, peer_device: int) -> int:
        return self._libcudart.mcDeviceDisablePeerAccess(peer_device)

    # ------------ Device memory management ------------
    def infiniMalloc(self, dev_ptr: Any, size: int) -> int:
        return self._libcudart.mcMalloc(dev_ptr, size)

    def infiniFree(self, dev_ptr: Any) -> int:
        return self._libcudart.mcFree(dev_ptr)

    # ------------ Pinned page memory ------------
    def infiniHostAlloc(self, ptr: Any, size: int, flags: int) -> int:
        return self._libcudart.mcMallocHost(ptr, size, flags)

    def infiniFreeHost(self, ptr: Any) -> int:
        return self._libcudart.mcFreeHost(ptr)

    # ------------ Stream management ------------
    def infiniStreamCreate(self, p_stream: Any) -> int:
        return self._libcudart.mcStreamCreate(p_stream)

    def infiniStreamDestroy(self, stream: Any) -> int:
        return self._libcudart.mcStreamDestroy(stream)

    def infiniStreamWaitEvent(self, stream: Any, event: Any, flags: int) -> int:
        return self._libcudart.mcStreamWaitEvent(stream, event, flags)

    def infiniStreamSynchronize(self, stream: Any) -> int:
        return self._libcudart.mcStreamSynchronize(stream)

    # ------------ Memory copy ------------
    def infiniMemcpyPeerAsync(
        self, dst: Any, dst_device: int, src: Any, src_device: int, count: int, stream: Any
    ) -> int:
        return self._libcudart.mcMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream)

    def infiniMemcpyAsync(self, dst: Any, src: Any, count: int, kind: int, stream: Any) -> int:
        return self._libcudart.mcMemcpyAsync(dst, src, count, kind, stream)

    # ------------ Event Management ------------
    def infiniEventCreate(self, event: Any) -> int:
        return self._libcudart.mcEventCreate(event)

    def infiniEventRecord(self, event: Any, stream: Any) -> int:
        return self._libcudart.mcEventRecord(event, stream)

    def infiniEventSynchronize(self, event: Any) -> int:
        return self._libcudart.mcEventSynchronize(event)

    def infiniEventElapsedTime(self, ms: Any, start: Any, end: Any) -> int:
        return self._libcudart.mcEventElapsedTime(ms, start, end)

    def infiniEventDestroy(self, event: Any) -> int:
        return self._libcudart.mcEventDestroy(event)



