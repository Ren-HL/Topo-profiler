#!/usr/bin/env python3
"""
The implementation of InfiniApi on the Iluvatar platform
- Inherit the NvidiaApi and reuse the loading logic of libcudart.so
- Only modify the smi command to "ixsmi"
- infiniXxx directly calls cudaXxx(because the days are compatible with the CUDA API)
"""

from __future__ import annotations
from nvidiaApi import NvidiaApi


class IluvatarApi(NvidiaApi):
    """API implementation of the Tianshu GPU platform"""
    
    smi = "ixsmi"
    
# All other methods inherit from the NvidiaApi and do not need to be rewritten
# Because the CUDA API of the Tianshu platform is fully compatible with NVIDIA
