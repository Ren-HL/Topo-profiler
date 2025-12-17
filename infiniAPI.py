#!/usr/bin/env python3
"""
Infini API Abstract Base Class:
- Not related to specific manufacturers
- Define a unified GPU operation interface
- Subclasses implement function bindings for specific platforms
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class InfiniApi(ABC):
    """
A unified Infini API abstraction for all GPU platforms

Agreement
-smi: The SMI command name corresponding to the platform (such as "nvidia-smi", "ixsmi")
-_libcudart: Runtime library handle (ctypes.CDLL)
    """
    
    smi: str = ""
    _libcudart: Any = None

    # ------------ Equipment Management ------------
    @abstractmethod
    def infiniGetDeviceCount(self) -> int:
        """Return the number of Gpus"""
        ...

    @abstractmethod
    def infiniSetDevice(self, index: int) -> int:
        """Set the current GPU device and return an error code"""
        ...

    @abstractmethod
    def infiniDeviceReset(self) -> int:
        """Reset the current GPU device and return an error code"""
        ...

    @abstractmethod
    def infiniGetErrorString(self, err: int) -> bytes:
        """Obtain the error message corresponding to the error code"""
        ...

    # ------------ P2P capability ------------
    @abstractmethod
    def infiniDeviceCanAccessPeer(self, can_access_ptr: Any, device: int, peer_device: int) -> int:
        """Query P2P access capability and return error code"""
        ...

    @abstractmethod
    def infiniDeviceEnablePeerAccess(self, peer_device: int, flags: int) -> int:
        """Enable P2P access and return an error code"""
        ...

    @abstractmethod
    def infiniDeviceDisablePeerAccess(self, peer_device: int) -> int:
        """Disable P2P access and return an error code"""
        ...

    # ------------ Device memory management ------------
    @abstractmethod
    def infiniMalloc(self, dev_ptr: Any, size: int) -> int:
        """Allocate device memory and return error codes"""
        ...

    @abstractmethod
    def infiniFree(self, dev_ptr: Any) -> int:
        """Release the device's memory and return an error code"""
        ...

    # ------------ Pinned page memory management ------------
    @abstractmethod
    def infiniHostAlloc(self, ptr: Any, size: int, flags: int) -> int:
        """Allocate pinned host memory and return error code"""
        ...

    @abstractmethod
    def infiniFreeHost(self, ptr: Any) -> int:
        """Release pinned host memory and return error code"""
        ...

    # ------------ Flow management ------------
    @abstractmethod
    def infiniStreamCreate(self, stream_ptr: Any) -> int:
        """Create a stream and return an error code"""
        ...

    @abstractmethod
    def infiniStreamDestroy(self, stream: Any) -> int:
        """Destroy the stream and return the error code"""
        ...

    @abstractmethod
    def infiniStreamWaitEvent(self, stream: Any, event: Any, flags: int) -> int:
        """Let the stream wait for the event and return the error code"""
        ...

    @abstractmethod
    def infiniStreamSynchronize(self, stream: Any) -> int:
        """Synchronous stream, return error code"""
        ...

    # ------------ Memory copy ------------
    @abstractmethod
    def infiniMemcpyPeerAsync(
        self,
        dst: Any,
        dst_device: int,
        src: Any,
        src_device: int,
        count: int,
        stream: Any,
    ) -> int:
        """Asynchronous P2P memory copy returns an error code"""
        ...

    @abstractmethod
    def infiniMemcpyAsync(
        self,
        dst: Any,
        src: Any,
        count: int,
        kind: int,
        stream: Any,
    ) -> int:
        """Asynchronous memory copy returns an error code"""
        ...

    # ------------ Event Management ------------
    @abstractmethod
    def infiniEventCreate(self, event_ptr: Any) -> int:
        """Create an event and return an error code"""
        ...

    @abstractmethod
    def infiniEventRecord(self, event: Any, stream: Any) -> int:
        """Record the event and return the error code"""
        ...

    @abstractmethod
    def infiniEventSynchronize(self, event: Any) -> int:
        """Synchronize events and return error codes"""
        ...

    @abstractmethod
    def infiniEventElapsedTime(self, ms: Any, start: Any, end: Any) -> int:
        """Calculate the event interval time and return the error code"""
        ...

    @abstractmethod
    def infiniEventDestroy(self, event: Any) -> int:
        """Destroy the event and return an error code"""
        ...
