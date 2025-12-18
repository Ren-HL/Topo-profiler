#!/usr/bin/env python3
"""
The implementation of InfiniApi on the Moore platform
Load libmusart.so via ctypes.CDLL
Bind the infiniXxx function to the libmusart.so function
"""

from __future__ import annotations
from typing import Any
import ctypes
import os

from infiniAPI import InfiniApi


class MusaApi(InfiniApi):
    """Musa Platform API Implementation"""
    
    smi = "mthreads-gmi"
    
    def __init__(self):
        try:
            self._libcudart = ctypes.CDLL("libmusart.so")
        except OSError:
            raise RuntimeError(
                "Failed to load libmusart.so. "
                "Make sure libmusart.so is installed and LD_LIBRARY_PATH is set."
            )

    # ------------ Device Management ------------
    def infiniGetDeviceCount(self, count: Any) -> int:
        return self._libcudart.musaGetDeviceCount(count)

    def infiniSetDevice(self, device: int) -> int:
        return self._libcudart.musaSetDevice(device)

    def infiniDeviceReset(self) -> int:
        return self._libcudart.musaDeviceReset()

    def infiniGetErrorString(self, error: int) -> bytes:
        self._libcudart.musaGetErrorString.restype = ctypes.c_char_p
        self._libcudart.musaGetErrorString.argtypes = [ctypes.c_int]
        return self._libcudart.musaGetErrorString(error)

    # ------------ P2P capability ------------
    def infiniDeviceCanAccessPeer(self, can_access: Any, device: int, peer_device: int) -> int:
        return self._libcudart.musaDeviceCanAccessPeer(can_access, device, peer_device)

    def infiniDeviceEnablePeerAccess(self, peer_device: int, flags: int) -> int:
        return self._libcudart.musaDeviceEnablePeerAccess(peer_device, flags)

    def infiniDeviceDisablePeerAccess(self, peer_device: int) -> int:
        return self._libcudart.musaDeviceDisablePeerAccess(peer_device)

    # ------------ Device memory management ------------
    def infiniMalloc(self, dev_ptr: Any, size: int) -> int:
        return self._libcudart.musaMalloc(dev_ptr, size)

    def infiniFree(self, dev_ptr: Any) -> int:
        return self._libcudart.musaFree(dev_ptr)

    # ------------ Pinned page memory ------------
    def infiniHostAlloc(self, ptr: Any, size: int, flags: int) -> int:
        return self._libcudart.musaMallocHost(ptr, size, flags)

    def infiniFreeHost(self, ptr: Any) -> int:
        return self._libcudart.musaFreeHost(ptr)

    # ------------ Stream management ------------
    def infiniStreamCreate(self, p_stream: Any) -> int:
        return self._libcudart.musaStreamCreate(p_stream)

    def infiniStreamDestroy(self, stream: Any) -> int:
        return self._libcudart.musaStreamDestroy(stream)

    def infiniStreamWaitEvent(self, stream: Any, event: Any, flags: int) -> int:
        return self._libcudart.musaStreamWaitEvent(stream, event, flags)

    def infiniStreamSynchronize(self, stream: Any) -> int:
        return self._libcudart.musaStreamSynchronize(stream)

    # ------------ Memory copy ------------
    def infiniMemcpyPeerAsync(
        self, dst: Any, dst_device: int, src: Any, src_device: int, count: int, stream: Any
    ) -> int:
        return self._libcudart.musaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream)

    def infiniMemcpyAsync(self, dst: Any, src: Any, count: int, kind: int, stream: Any) -> int:
        return self._libcudart.musaMemcpyAsync(dst, src, count, kind, stream)

    # ------------ Event Management ------------
    def infiniEventCreate(self, event: Any) -> int:
        return self._libcudart.musaEventCreate(event)

    def infiniEventRecord(self, event: Any, stream: Any) -> int:
        return self._libcudart.musaEventRecord(event, stream)

    def infiniEventSynchronize(self, event: Any) -> int:
        return self._libcudart.musaEventSynchronize(event)

    def infiniEventElapsedTime(self, ms: Any, start: Any, end: Any) -> int:
        return self._libcudart.musaEventElapsedTime(ms, start, end)

    def infiniEventDestroy(self, event: Any) -> int:
        return self._libcudart.musaEventDestroy(event)



