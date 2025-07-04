# Python wrapper code to use the exported c++ mvsr functions

import ctypes
import platform
from enum import IntEnum
from pathlib import Path

import numpy as np


def ndarray_or_null(*args, **kwargs):
    ndtype = np.ctypeslib.ndpointer(*args, **kwargs)

    def from_param(cls, obj):
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

#################################################
# Function dictionary (direct usage discouraged) #
#################################################

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


class Mvsr:
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

    __reg = None

    def __init__(self, x, y=None, minsegsize=None, placement=Placement.ALL, dtype=np.float64):
        if isinstance(x, Mvsr):
            self.__dimensions = x.__dimensions
            self.__variants = x.__variants
            self.__dtype = x.__dtype
            self.__funcs = x.__funcs
            self.__data = x.__data
            self.__num_pieces = x.__num_pieces
            self.__reg = x.__funcs["copy"](x.__reg)
            if self.__reg is None:
                raise Exception("Error copying regression.")
            return

        if y is None:
            raise Exception("Missing y values")
        x = np.array(x, dtype=dtype)
        y = np.array(y, dtype=dtype)
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise Exception("Unsupported input dymensions.")
        if x.shape[1] != y.shape[1]:
            raise Exception("Mismatch in sample count of x and y.")
        if dtype not in funcs:
            raise Exception("Unsupported dtype.")

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
            raise Exception("Error initializing segments.")  # todo: more info?

    def reduce(self, min, max=0, alg=Algorithm.GREEDY, score=Score.EXACT, metric=Metric.MSE):
        res = self.__funcs["reduce"](self.__reg, min, max, alg, metric, score)
        if res == 0:
            raise Exception("Error reducing segments.")  # todo: more info?
        self.__num_pieces = res

    def optimize(self, range=ctypes.c_uint(-1).value + 1 // 4, metric=Metric.MSE):
        res = self.__funcs["optimize"](self.__reg, self.__data, range, metric)
        if res == 0:
            raise Exception("Error optimizing segments.")  # todo: more info?
        self.__num_pieces = res

    def get_data(self):
        if self.__num_pieces is None or self.__num_pieces == 0:
            res = self.__funcs["get_data"](self.__reg, None, None, None)
            if res == 0:
                raise Exception("Error getting segments.")  # todo: more info?
            self.__num_pieces = res
        starts = np.empty((self.__num_pieces), dtype=np.uintp)
        models = np.empty(
            (self.__num_pieces, self.__dimensions, self.__variants), dtype=self.__dtype
        )
        errors = np.empty((self.__num_pieces), dtype=self.__dtype)

        res = self.__funcs["get_data"](self.__reg, starts, models, errors)
        if res == 0:
            raise Exception("Error getting segments.")  # todo: more info?

        return (starts, models, errors)

    def copy(self):
        return Mvsr(self)

    def close(self):
        if self.__reg is not None:
            self.__funcs["release"](self.__reg)
            self.__reg = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
