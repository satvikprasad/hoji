"""ctypes bindings for the hoji backend kernels.

The folding factor is a compile-time macro, so CMake emits one shared library
per value. Each Backend instance wraps one of them.
"""
import ctypes
import glob
import os
import sys

import numpy as np

_F32 = ctypes.POINTER(ctypes.c_float)


def _as_f32(arr):
    """Raw pointer to a C-contiguous float32 array, no copy."""
    if arr.dtype != np.float32:
        raise TypeError(f"expected float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous")
    return arr.ctypes.data_as(_F32)


class Backend:
    """One compiled backend, i.e. one folding factor."""

    def __init__(self, path):
        self.path = path
        self._lib = ctypes.CDLL(path)

        self._lib.hoji_backend_folding_factor.restype = ctypes.c_int
        self._lib.hoji_backend_folding_factor.argtypes = []

        self._lib.hoji_matmul_f32.restype = None
        self._lib.hoji_matmul_f32.argtypes = [
            _F32, _F32, _F32,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ]
        self.folding_factor = self._lib.hoji_backend_folding_factor()

    def matmul(self, a, b_trans, out=None):
        """out(m, n) = a(m, k) @ b_trans(n, k).T"""
        m, k = a.shape
        n, k2 = b_trans.shape
        if k != k2:
            raise ValueError(f"inner dims disagree: {k} vs {k2}")
        if out is None:
            out = np.zeros((m, n), dtype=np.float32)
        self._lib.hoji_matmul_f32(_as_f32(a), _as_f32(b_trans), _as_f32(out),
                                  m, k, n)
        return out

    def __repr__(self):
        return f"Backend(f={self.folding_factor}, {os.path.basename(self.path)})"


def load_all(build_dir="build"):
    """Every backend library in build_dir, sorted by folding factor."""
    libs = sorted(glob.glob(os.path.join(build_dir, "libhoji_backend_f*.dylib")))
    if not libs:
        libs = sorted(glob.glob(os.path.join(build_dir, "libhoji_backend_f*.so")))
    if not libs:
        sys.exit(f"no backend libraries in {build_dir!r}; "
                 f"run: cmake --build {build_dir}")
    return sorted((Backend(p) for p in libs), key=lambda b: b.folding_factor)
