import ctypes
import os
import sys
from pathlib import Path
from ctypes import c_int, c_double, POINTER

c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)

_DENSE_ARGTYPES = [c_double_p, c_double_p, c_int, c_int, c_int]
_DENSE_PW_ARGTYPES = [c_double_p, c_double_p, c_double_p, c_int, c_int, c_int, c_int]
_SPARSE_ARGTYPES = [c_int_p, c_int_p, c_double_p, c_double_p, c_int, c_int, c_int, c_int]
_SPARSE_PW_ARGTYPES = [
    c_int_p, c_int_p, c_double_p,
    c_int_p, c_int_p, c_double_p,
    c_double_p, c_int, c_int, c_int, c_int, c_int, c_int,
]


def _lib_paths():
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir,
        pkg_dir / "lib",
        pkg_dir.parent / "lib",
        pkg_dir.parent / "build",
    ]
    env_path = os.environ.get("GADES_LIB_DIR")
    if env_path:
        candidates.insert(0, Path(env_path))
    # Also check the installed package location in site-packages
    for sp in sys.path:
        sp_pkg = Path(sp) / "gades"
        if sp_pkg.is_dir() and sp_pkg != pkg_dir:
            candidates.append(sp_pkg)
    return candidates


def _find_lib(name):
    suffix = ".so"
    if sys.platform == "darwin":
        suffix = ".dylib"
    for d in _lib_paths():
        p = d / f"{name}{suffix}"
        if p.exists():
            return str(p)
    return None


def _load_gpu():
    path = _find_lib("_gades_gpu")
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None

    lib.gades_gpu_available.argtypes = []
    lib.gades_gpu_available.restype = c_int

    lib.gades_dense_gpu.argtypes = _DENSE_ARGTYPES
    lib.gades_dense_gpu.restype = c_int

    lib.gades_dense_pairwise_gpu.argtypes = _DENSE_PW_ARGTYPES
    lib.gades_dense_pairwise_gpu.restype = c_int

    lib.gades_sparse_gpu.argtypes = _SPARSE_ARGTYPES
    lib.gades_sparse_gpu.restype = c_int

    lib.gades_sparse_pairwise_gpu.argtypes = _SPARSE_PW_ARGTYPES
    lib.gades_sparse_pairwise_gpu.restype = c_int

    return lib


def _load_cpu():
    path = _find_lib("_gades_cpu")
    if path is None:
        raise RuntimeError(
            "Could not find _gades_cpu shared library. "
            "Make sure gades is installed: pip install ./python"
        )
    lib = ctypes.CDLL(path)

    lib.gades_dense_cpu.argtypes = _DENSE_ARGTYPES
    lib.gades_dense_cpu.restype = c_int

    lib.gades_dense_pairwise_cpu.argtypes = _DENSE_PW_ARGTYPES
    lib.gades_dense_pairwise_cpu.restype = c_int

    lib.gades_sparse_cpu.argtypes = _SPARSE_ARGTYPES
    lib.gades_sparse_cpu.restype = c_int

    lib.gades_sparse_pairwise_cpu.argtypes = _SPARSE_PW_ARGTYPES
    lib.gades_sparse_pairwise_cpu.restype = c_int

    return lib


class _Backend:
    def __init__(self):
        self._gpu = None
        self._cpu = None
        self._gpu_loaded = False
        self._cpu_loaded = False

    @property
    def gpu(self):
        if not self._gpu_loaded:
            self._gpu = _load_gpu()
            self._gpu_loaded = True
        return self._gpu

    @property
    def cpu(self):
        if not self._cpu_loaded:
            self._cpu = _load_cpu()
            self._cpu_loaded = True
        return self._cpu

    def has_gpu(self):
        lib = self.gpu
        if lib is None:
            return False
        try:
            return lib.gades_gpu_available() == 1
        except Exception:
            return False


backend = _Backend()
