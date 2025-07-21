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
        if obj is None:
            return obj
        return ndtype.from_param(obj)

    return type(ndtype.__name__, (ndtype,), {"from_param": classmethod(from_param)})


####################
# Type definitions #
####################

__size_t = ctypes.c_size_t
__voidp = ctypes.c_void_p
__int = ctypes.c_int
__uint = ctypes.c_uint
# __sizeptr = ctypes.POINTER(ctypes.c_size_t)  # (dtype=np.uintp, ndim=1, flags="C")
# __f64ptr = ctypes.POINTER(ctypes.c_double)  # might be NULL
# __f32ptr = ctypes.POINTER(ctypes.c_float)  # might be NULL
__sizeptr = ndarray_or_null(dtype=np.uintp, ndim=1, flags="C")
__f64ptr = ndarray_or_null(dtype=np.float64, ndim=1, flags="C")
__f32ptr = ndarray_or_null(dtype=np.float32, ndim=1, flags="C")
__f64ptr2d = ndarray_or_null(dtype=np.float64, ndim=2, flags="C")
__f32ptr2d = ndarray_or_null(dtype=np.float32, ndim=2, flags="C")
__f64ptr3d = ndarray_or_null(dtype=np.float64, ndim=3, flags="C")
__f32ptr3d = ndarray_or_null(dtype=np.float32, ndim=3, flags="C")

########################
# Load dynamic library #
########################

LIBRARY_EXTENSION = {"Windows": "dll", "Darwin": "dylib"}.get(platform.system(), "so")
LIBRARY_PATH = Path(__file__).parent / "lib" / f"libmvsr.{LIBRARY_EXTENSION}"
__libmvsr = ctypes.CDLL(LIBRARY_PATH)

########################
# Function definitions #
########################

# F64 Functions
__libmvsr.mvsr_init_f64.restype = __voidp
__libmvsr.mvsr_init_f64.argtypes = [__size_t, __size_t, __size_t, __f64ptr2d, __size_t, __int]
__libmvsr.mvsr_reduce_f64.restype = __size_t
__libmvsr.mvsr_reduce_f64.argtypes = [__voidp, __size_t, __size_t, __int, __int, __int]
__libmvsr.mvsr_optimize_f64.restype = __size_t
__libmvsr.mvsr_optimize_f64.argtypes = [__voidp, __f64ptr2d, __uint, __int]
__libmvsr.mvsr_get_data_f64.restype = __size_t
__libmvsr.mvsr_get_data_f64.argtypes = [__voidp, __sizeptr, __f64ptr3d, __f64ptr]
__libmvsr.mvsr_copy_f64.restype = __voidp
__libmvsr.mvsr_copy_f64.argtypes = [__voidp]
__libmvsr.mvsr_release_f64.restype = None
__libmvsr.mvsr_release_f64.argtypes = [__voidp]

# F32 Functions
__libmvsr.mvsr_init_f32.restype = __voidp
__libmvsr.mvsr_init_f32.argtypes = [__size_t, __size_t, __size_t, __f32ptr2d, __size_t, __int]
__libmvsr.mvsr_reduce_f32.restype = __size_t
__libmvsr.mvsr_reduce_f32.argtypes = [__voidp, __size_t, __size_t, __int, __int, __int]
__libmvsr.mvsr_optimize_f32.restype = __size_t
__libmvsr.mvsr_optimize_f32.argtypes = [__voidp, __f32ptr2d, __uint, __int]
__libmvsr.mvsr_get_data_f32.restype = __size_t
__libmvsr.mvsr_get_data_f32.argtypes = [__voidp, __sizeptr, __f32ptr3d, __f32ptr]
__libmvsr.mvsr_copy_f32.restype = __voidp
__libmvsr.mvsr_copy_f32.argtypes = [__voidp]
__libmvsr.mvsr_release_f32.restype = None
__libmvsr.mvsr_release_f32.argtypes = [__voidp]

##################################################
# Function dictionary (direct usage discouraged) #
##################################################

funcs = {
    np.float64: {
        "init": __libmvsr.mvsr_init_f64,
        "reduce": __libmvsr.mvsr_reduce_f64,
        "optimize": __libmvsr.mvsr_optimize_f64,
        "get_data": __libmvsr.mvsr_get_data_f64,
        "copy": __libmvsr.mvsr_copy_f64,
        "release": __libmvsr.mvsr_release_f64,
    },
    np.float32: {
        "init": __libmvsr.mvsr_init_f32,
        "reduce": __libmvsr.mvsr_reduce_f32,
        "optimize": __libmvsr.mvsr_optimize_f32,
        "get_data": __libmvsr.mvsr_get_data_f32,
        "copy": __libmvsr.mvsr_copy_f32,
        "release": __libmvsr.mvsr_release_f32,
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
    ALL = 0


class Algorithm(IntEnum):
    GREEDY = 0
    DP = 1


class Metric(IntEnum):
    MSE = 0


class Score(IntEnum):
    EXACT = 0
    CHI = 1


class Mvsr:
    __reg = None

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

        self.__dimensions = x.shape[0]
        self.__variants = y.shape[0]
        self.__dtype = dtype
        self.__funcs = funcs[dtype]
        self.__data = np.ascontiguousarray(
            np.concatenate((x, y), dtype=dtype).transpose(), dtype=dtype
        )
        self.__num_pieces = None
        minsegsize = self.__dimensions if minsegsize is None else minsegsize

        self.__reg = self.__funcs["init"](
            self.__data.shape[0],
            self.__dimensions,
            self.__variants,
            self.__data,
            minsegsize,
            placement,
        )
        if self.__reg is None:
            raise InternalError(self.__funcs["init"], self.__reg)

    def reduce(
        self,
        min: int,
        max: int = 0,
        alg: Algorithm = Algorithm.GREEDY,
        score: Score = Score.EXACT,
        metric: Metric = Metric.MSE,
    ):
        res = self.__funcs["reduce"](self.__reg, min, max, alg, metric, score)
        if res == 0:
            raise InternalError(self.__funcs["reduce"], res)
        self.__num_pieces = res

    def optimize(self, range: int = ctypes.c_uint(-1).value + 1 // 4, metric: Metric = Metric.MSE):
        res = self.__funcs["optimize"](self.__reg, self.__data, range, metric)
        if res == 0:
            raise InternalError(self.__funcs["optimize"], res)
        self.__num_pieces = res

    def get_data(self):
        if self.__num_pieces is None or self.__num_pieces == 0:
            res = self.__funcs["get_data"](self.__reg, None, None, None)
            if res == 0:
                raise InternalError(self.__funcs["get_data"], res)
            self.__num_pieces = res
        starts = np.empty((self.__num_pieces), dtype=np.uintp)
        models = np.empty(
            (self.__num_pieces, self.__dimensions, self.__variants), dtype=self.__dtype
        )
        errors = np.empty((self.__num_pieces), dtype=self.__dtype)

        res = self.__funcs["get_data"](self.__reg, starts, models, errors)
        if res == 0:
            raise InternalError(self.__funcs["get_data"], res)

        return (starts, models, errors)

    def __copy__(self) -> typing.Self:
        copy = self.__new__(self.__class__)

        copy.__dimensions = self.__dimensions
        copy.__variants = self.__variants
        copy.__dtype = self.__dtype
        copy.__funcs = self.__funcs
        copy.__data = self.__data
        copy.__num_pieces = self.__num_pieces
        copy.__reg = self.__funcs["copy"](self.__reg)
        if self.__reg is None:
            raise InternalError(self.__funcs["copy"], self.__reg)

        return copy

    def close(self):
        if self.__reg is not None:
            self.__funcs["release"](self.__reg)
            self.__reg = None

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
