# Python wrapper code to use the exported c++ mvsr functions

import ctypes
import platform
import typing
from enum import IntEnum
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    ndptr = type[np.ctypeslib._ndptr[typing.Any]]
    from ctypes import _NamedFuncPointer as NamedFuncPointer
else:
    ndptr = object
    NamedFuncPointer = object


def ndarray_or_null(*args: typing.Any, **kwargs: typing.Any):
    ndtype = typing.cast(ndptr, np.ctypeslib.ndpointer(*args, **kwargs))

    def from_param(cls: type, obj: np.ndarray | None):
        return ndtype.from_param(obj) if obj is not None else None

    return type(ndtype.__name__, (ndtype,), {"from_param": classmethod(from_param)})


####################
# Type definitions #
####################

_size_t = ctypes.c_size_t
_voidp = ctypes.c_void_p
_int = ctypes.c_int
_uint = ctypes.c_uint
# _sizeptr = ctypes.POINTER(ctypes.c_size_t)  # (dtype=np.uintp, ndim=1, flags="C")
# _f64ptr = ctypes.POINTER(ctypes.c_double)  # might be NULL
# _f32ptr = ctypes.POINTER(ctypes.c_float)  # might be NULL
_sizeptr = ndarray_or_null(dtype=np.uintp, ndim=1, flags="C")
_f64ptr = ndarray_or_null(dtype=np.float64, ndim=1, flags="C")
_f32ptr = ndarray_or_null(dtype=np.float32, ndim=1, flags="C")
_f64ptr2d = ndarray_or_null(dtype=np.float64, ndim=2, flags="C")
_f32ptr2d = ndarray_or_null(dtype=np.float32, ndim=2, flags="C")
_f64ptr3d = ndarray_or_null(dtype=np.float64, ndim=3, flags="C")
_f32ptr3d = ndarray_or_null(dtype=np.float32, ndim=3, flags="C")

########################
# Load dynamic library #
########################

LIBRARY_EXTENSION = {"Windows": "dll", "Darwin": "dylib"}.get(platform.system(), "so")
LIBRARY_PATH = Path(__file__).parent / "lib" / f"libmvsr.{LIBRARY_EXTENSION}"
_libmvsr = ctypes.CDLL(LIBRARY_PATH)

########################
# Function definitions #
########################

# F64 Functions
_libmvsr.mvsr_init_f64.restype = _voidp
_libmvsr.mvsr_init_f64.argtypes = [_size_t, _size_t, _size_t, _f64ptr2d, _size_t, _int]
_libmvsr.mvsr_reduce_f64.restype = _size_t
_libmvsr.mvsr_reduce_f64.argtypes = [_voidp, _size_t, _size_t, _int, _int, _int]
_libmvsr.mvsr_optimize_f64.restype = _size_t
_libmvsr.mvsr_optimize_f64.argtypes = [_voidp, _f64ptr2d, _uint, _int]
_libmvsr.mvsr_get_data_f64.restype = _size_t
_libmvsr.mvsr_get_data_f64.argtypes = [_voidp, _sizeptr, _f64ptr3d, _f64ptr]
_libmvsr.mvsr_copy_f64.restype = _voidp
_libmvsr.mvsr_copy_f64.argtypes = [_voidp]
_libmvsr.mvsr_release_f64.restype = None
_libmvsr.mvsr_release_f64.argtypes = [_voidp]

# F32 Functions
_libmvsr.mvsr_init_f32.restype = _voidp
_libmvsr.mvsr_init_f32.argtypes = [_size_t, _size_t, _size_t, _f32ptr2d, _size_t, _int]
_libmvsr.mvsr_reduce_f32.restype = _size_t
_libmvsr.mvsr_reduce_f32.argtypes = [_voidp, _size_t, _size_t, _int, _int, _int]
_libmvsr.mvsr_optimize_f32.restype = _size_t
_libmvsr.mvsr_optimize_f32.argtypes = [_voidp, _f32ptr2d, _uint, _int]
_libmvsr.mvsr_get_data_f32.restype = _size_t
_libmvsr.mvsr_get_data_f32.argtypes = [_voidp, _sizeptr, _f32ptr3d, _f32ptr]
_libmvsr.mvsr_copy_f32.restype = _voidp
_libmvsr.mvsr_copy_f32.argtypes = [_voidp]
_libmvsr.mvsr_release_f32.restype = None
_libmvsr.mvsr_release_f32.argtypes = [_voidp]

##################################################
# Function dictionary (direct usage discouraged) #
##################################################

funcs = {
    np.float64: {
        "init": _libmvsr.mvsr_init_f64,
        "reduce": _libmvsr.mvsr_reduce_f64,
        "optimize": _libmvsr.mvsr_optimize_f64,
        "get_data": _libmvsr.mvsr_get_data_f64,
        "copy": _libmvsr.mvsr_copy_f64,
        "release": _libmvsr.mvsr_release_f64,
    },
    np.float32: {
        "init": _libmvsr.mvsr_init_f32,
        "reduce": _libmvsr.mvsr_reduce_f32,
        "optimize": _libmvsr.mvsr_optimize_f32,
        "get_data": _libmvsr.mvsr_get_data_f32,
        "copy": _libmvsr.mvsr_copy_f32,
        "release": _libmvsr.mvsr_release_f32,
    },
}

#######################
# Low-Level Interface #
#######################

valid_dtypes = type[np.float32] | type[np.float64]
MvsrArray = npt.NDArray[np.float32 | np.float64]


class InternalError(Exception):
    _Unset = object()

    def __init__(self, function: NamedFuncPointer, return_value: typing.Any = _Unset):
        super().__init__(
            f"internal error in '{function.__name__}'"
            + (f" (returned '{return_value}')" if return_value is not self._Unset else "")
        )


class Placement(IntEnum):
    """Method used to place initial segments."""

    ALL = 0


class Algorithm(IntEnum):
    """Algorithm used to reduce the number of segments."""

    GREEDY = 0
    """Fast Greedy Algorithm ( :math:`O(n \\log n)` )."""
    DP = 1
    """Dynamic Program ( :math:`O(n^2)` )."""


class Metric(IntEnum):
    """Metric used to calculate the error."""

    MSE = 0
    """Mean Squared Error."""


class Score(IntEnum):
    """Scoring method used to optimize the number of segments."""

    EXACT = 0
    CHI = 1
    """Calinski-Harabasz index."""


class Mvsr:
    _reg = None

    def __init__(
        self,
        x: MvsrArray,
        y: MvsrArray,
        minsegsize: int | None = None,
        placement: Placement = Placement.ALL,
        dtype: valid_dtypes = np.float64,
    ):
        if len(x.shape) != 2:
            raise ValueError(f"unsupported input shape 'len({x.shape}) != 2'")
        if len(y.shape) != 2:
            raise ValueError(f"unsupported input shape 'len({y.shape}) != 2'")
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"incompatible input shapes '{x.shape}, {y.shape}' ({x.shape[1]} != {y.shape[1]})"
            )
        if dtype not in funcs:
            raise TypeError(f"unsupported dtype '{dtype}' (valid: {valid_dtypes})")

        self._dimensions = x.shape[0]
        self._variants = y.shape[0]
        self._dtype = dtype
        self._funcs = funcs[dtype]
        self._data = np.ascontiguousarray(
            np.concatenate((x, y), dtype=dtype).transpose(), dtype=dtype
        )
        self._num_pieces = None
        minsegsize = self._dimensions if minsegsize is None else minsegsize

        self._reg = self._funcs["init"](
            self._data.shape[0],
            self._dimensions,
            self._variants,
            self._data,
            minsegsize,
            placement,
        )
        if self._reg is None:
            raise InternalError(self._funcs["init"], self._reg)

    def reduce(
        self,
        min: int,
        max: int = 0,
        alg: Algorithm = Algorithm.GREEDY,
        score: Score = Score.EXACT,
        metric: Metric = Metric.MSE,
    ):
        res = self._funcs["reduce"](self._reg, min, max, alg, metric, score)
        if res == 0:
            raise InternalError(self._funcs["reduce"], res)
        self._num_pieces = res

    def optimize(self, range: int = ctypes.c_uint(-1).value + 1 // 4, metric: Metric = Metric.MSE):
        res = self._funcs["optimize"](self._reg, self._data, range, metric)
        if res == 0:
            raise InternalError(self._funcs["optimize"], res)
        self._num_pieces = res

    def get_data(self):
        if self._num_pieces is None or self._num_pieces == 0:  # pragma: no cover
            res = self._funcs["get_data"](self._reg, None, None, None)
            if res == 0:
                raise InternalError(self._funcs["get_data"], res)
            self._num_pieces = res

        starts = np.empty((self._num_pieces), dtype=np.uintp)
        models = np.empty((self._num_pieces, self._dimensions, self._variants), dtype=self._dtype)
        errors = np.empty((self._num_pieces), dtype=self._dtype)

        res = self._funcs["get_data"](self._reg, starts, models, errors)
        if res == 0:
            raise InternalError(self._funcs["get_data"], res)

        return (starts, models, errors)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        copy = self.__new__(self.__class__)

        copy._dimensions = self._dimensions
        copy._variants = self._variants
        copy._dtype = self._dtype
        copy._funcs = self._funcs
        copy._data = self._data
        copy._num_pieces = self._num_pieces
        copy._reg = self._funcs["copy"](self._reg)
        if self._reg is None:
            raise InternalError(self._funcs["copy"], self._reg)

        return copy

    def close(self):
        if self._reg is not None:
            self._funcs["release"](self._reg)
            self._reg = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[Exception] | None,
        exc_value: Exception | None,
        traceback: TracebackType | None,
    ):
        self.close()
