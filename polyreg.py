# Python wrapper code to use the exported c++ tolyreg functions

import ctypes
from pathlib import Path

import numpy as np

libpwreg = ctypes.CDLL(Path(__file__).parent / "build" / "libpwreg.so")
f32ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C")
sizeptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
f32ptr2d = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")
f64ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
f64ptr2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")

libpwreg.pwreg_f32d2_init.argtypes = [ctypes.c_size_t, f32ptr2d]
libpwreg.pwreg_f32d2_init.restype = ctypes.c_void_p
libpwreg.pwreg_f32d2_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f32d2_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f32d2_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f32d2_delete.restype = None
libpwreg.pwreg_f32d2_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f32ptr2d, f32ptr]
libpwreg.pwreg_f32d2_reduce.restype = None
libpwreg.pwreg_f32d2_optimize.argtypes = [ctypes.c_void_p, f32ptr2d]
libpwreg.pwreg_f32d2_optimize.restype = ctypes.c_size_t

libpwreg.pwreg_f64d1_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d1_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d1_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d1_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d1_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d1_delete.restype = None
libpwreg.pwreg_f64d1_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d1_reduce.restype = None
libpwreg.pwreg_f64d1_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d1_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d2_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d2_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d2_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d2_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d2_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d2_delete.restype = None
libpwreg.pwreg_f64d2_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d2_reduce.restype = None
libpwreg.pwreg_f64d2_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d2_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d3_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d3_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d3_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d3_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d3_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d3_delete.restype = None
libpwreg.pwreg_f64d3_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d3_reduce.restype = None
libpwreg.pwreg_f64d3_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d3_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d4_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d4_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d4_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d4_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d4_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d4_delete.restype = None
libpwreg.pwreg_f64d4_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d4_reduce.restype = None
libpwreg.pwreg_f64d4_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d4_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d5_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d5_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d5_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d5_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d5_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d5_delete.restype = None
libpwreg.pwreg_f64d5_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d5_reduce.restype = None
libpwreg.pwreg_f64d5_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d5_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d6_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d6_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d6_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d6_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d6_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d6_delete.restype = None
libpwreg.pwreg_f64d6_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d6_reduce.restype = None
libpwreg.pwreg_f64d6_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d6_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d7_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d7_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d7_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d7_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d7_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d7_delete.restype = None
libpwreg.pwreg_f64d7_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d7_reduce.restype = None
libpwreg.pwreg_f64d7_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d7_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d8_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d8_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d8_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d8_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d8_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d8_delete.restype = None
libpwreg.pwreg_f64d8_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d8_reduce.restype = None
libpwreg.pwreg_f64d8_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d8_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d9_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d9_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d9_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d9_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d9_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d9_delete.restype = None
libpwreg.pwreg_f64d9_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d9_reduce.restype = None
libpwreg.pwreg_f64d9_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d9_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d16_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d16_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d16_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d16_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d16_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d16_delete.restype = None
libpwreg.pwreg_f64d16_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d16_reduce.restype = None
libpwreg.pwreg_f64d16_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d16_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d32_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d32_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d32_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d32_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d32_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d32_delete.restype = None
libpwreg.pwreg_f64d32_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d32_reduce.restype = None
libpwreg.pwreg_f64d32_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d32_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d64_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d64_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d64_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d64_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d64_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d64_delete.restype = None
libpwreg.pwreg_f64d64_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d64_reduce.restype = None
libpwreg.pwreg_f64d64_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d64_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d128_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d128_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d128_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d128_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d128_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d128_delete.restype = None
libpwreg.pwreg_f64d128_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d128_reduce.restype = None
libpwreg.pwreg_f64d128_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d128_optimize.restype = ctypes.c_size_t
libpwreg.pwreg_f64d256_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d256_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d256_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d256_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d256_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d256_delete.restype = None
libpwreg.pwreg_f64d256_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d256_reduce.restype = None
libpwreg.pwreg_f64d256_optimize.argtypes = [ctypes.c_void_p, f64ptr2d]
libpwreg.pwreg_f64d256_optimize.restype = ctypes.c_size_t

pwrf64_init = {
               1: libpwreg.pwreg_f64d1_init,
               2: libpwreg.pwreg_f64d2_init,
               3: libpwreg.pwreg_f64d3_init,
               4: libpwreg.pwreg_f64d4_init,
               5: libpwreg.pwreg_f64d5_init,
               6: libpwreg.pwreg_f64d6_init,
               7: libpwreg.pwreg_f64d7_init,
               8: libpwreg.pwreg_f64d8_init,
               9: libpwreg.pwreg_f64d9_init,
               16: libpwreg.pwreg_f64d16_init,
               32: libpwreg.pwreg_f64d32_init,
               64: libpwreg.pwreg_f64d64_init,
               128: libpwreg.pwreg_f64d128_init,
               256: libpwreg.pwreg_f64d256_init,
}
pwrf64_copy = {
               1: libpwreg.pwreg_f64d1_copy,
               2: libpwreg.pwreg_f64d2_copy,
               3: libpwreg.pwreg_f64d3_copy,
               4: libpwreg.pwreg_f64d4_copy,
               5: libpwreg.pwreg_f64d5_copy,
               6: libpwreg.pwreg_f64d6_copy,
               7: libpwreg.pwreg_f64d7_copy,
               8: libpwreg.pwreg_f64d8_copy,
               9: libpwreg.pwreg_f64d9_copy,
               16: libpwreg.pwreg_f64d16_copy,
               32: libpwreg.pwreg_f64d32_copy,
               64: libpwreg.pwreg_f64d64_copy,
               128: libpwreg.pwreg_f64d128_copy,
               256: libpwreg.pwreg_f64d256_copy,
}
pwrf64_delete = {
               1: libpwreg.pwreg_f64d1_delete,
               2: libpwreg.pwreg_f64d2_delete,
               3: libpwreg.pwreg_f64d3_delete,
               4: libpwreg.pwreg_f64d4_delete,
               5: libpwreg.pwreg_f64d5_delete,
               6: libpwreg.pwreg_f64d6_delete,
               7: libpwreg.pwreg_f64d7_delete,
               8: libpwreg.pwreg_f64d8_delete,
               9: libpwreg.pwreg_f64d9_delete,
               16: libpwreg.pwreg_f64d16_delete,
               32: libpwreg.pwreg_f64d32_delete,
               64: libpwreg.pwreg_f64d64_delete,
               128: libpwreg.pwreg_f64d128_delete,
               256: libpwreg.pwreg_f64d256_delete,
}
pwrf64_reduce = {
               1: libpwreg.pwreg_f64d1_reduce,
               2: libpwreg.pwreg_f64d2_reduce,
               3: libpwreg.pwreg_f64d3_reduce,
               4: libpwreg.pwreg_f64d4_reduce,
               5: libpwreg.pwreg_f64d5_reduce,
               6: libpwreg.pwreg_f64d6_reduce,
               7: libpwreg.pwreg_f64d7_reduce,
               8: libpwreg.pwreg_f64d8_reduce,
               9: libpwreg.pwreg_f64d9_reduce,
               16: libpwreg.pwreg_f64d16_reduce,
               32: libpwreg.pwreg_f64d32_reduce,
               64: libpwreg.pwreg_f64d64_reduce,
               128: libpwreg.pwreg_f64d128_reduce,
               256: libpwreg.pwreg_f64d256_reduce,
}
pwrf64_optimize = {
               1: libpwreg.pwreg_f64d1_optimize,
               2: libpwreg.pwreg_f64d2_optimize,
               3: libpwreg.pwreg_f64d3_optimize,
               4: libpwreg.pwreg_f64d4_optimize,
               5: libpwreg.pwreg_f64d5_optimize,
               6: libpwreg.pwreg_f64d6_optimize,
               7: libpwreg.pwreg_f64d7_optimize,
               8: libpwreg.pwreg_f64d8_optimize,
               9: libpwreg.pwreg_f64d9_optimize,
               16: libpwreg.pwreg_f64d16_optimize,
               32: libpwreg.pwreg_f64d32_optimize,
               64: libpwreg.pwreg_f64d64_optimize,
               128: libpwreg.pwreg_f64d128_optimize,
               256: libpwreg.pwreg_f64d256_optimize,
}

class PwReg:
    def __init__(self, x, y=None, funcs=None):
        if isinstance(x, PwReg):
            self.reg = self.reg = pwrf64_copy[self.data.shape[1]-1](x.reg)
            self.num_pieces = x.num_pieces
            self.data = x.data
        elif len(x) != len(y):
            raise("Every x needs an y")
        else:
            self.data = np.ascontiguousarray(np.concatenate((x,np.array(y,ndmin=2).T), axis=1))
            self.reg = pwrf64_init[self.data.shape[1]-1](self.data.shape[0], self.data)
            self.num_pieces = self.data.shape[0]//(self.data.shape[1]-1)

    def __copy__(self):
        return PwReg(self)

    def __del__(self):
        pwrf64_copy[self.data.shape[1]-1](self.reg)

    def reduce(self, num_pieces):
        if (num_pieces < self.num_pieces):
            self.num_pieces = num_pieces
        self.starts = np.zeros((self.num_pieces), dtype=np.uintp)
        self.models = np.zeros((self.num_pieces,self.data.shape[1]-1), dtype=np.float64)
        self.errors = np.zeros((self.num_pieces), dtype=np.float64)
        pwrf64_reduce[self.data.shape[1]-1](self.reg, self.num_pieces, self.starts, self.models, self.errors)
        return (self.starts, self.models, self.errors)

    def optimize(self):
        self.num_pieces = pwrf64_optimize[self.data.shape[1]-1](self.reg, self.data)

    def getData(self):
        return self.reduce(np.iinfo(np.uintp).max)

def linfit(model, x):
    return model[0] + model[1]*x

def get_breakpoints(xs, starts, models):
    starts = starts[1:].tolist()+[len(xs[:-1])]
    curstart = 0
    x = []
    y = []
    for i in range(0,len(starts)):
        curend = starts[i]
        x = x+[xs[curstart], xs[curend]]
        y = y+[linfit(models[i], val) for val in [xs[curstart], xs[curend]]]
        curstart = curend
    return (x,y)

def get_ys(X, starts, models):
    #print(starts)
    #print(models)
    X_per_model = np.split(X, starts[1:])
    ys_per_model = []
    for model, model_xs in zip(models, X_per_model):
        ys_per_model.append(model_xs @ model)
    return np.concatenate(ys_per_model)

if __name__ == "__main__":
    reg = PwReg(range(20), [1,2, 3,4, 5,7, 6,5, 4,3, 2,11, 12,13, 14,15, 16,2, 2,2])
    r = PwReg(reg)
    print(r.reduce(4))
    r = PwReg(reg)
    print(r.dp(4))
    r = PwReg(reg)
    r.reduce(2*4-1)
    r.optimize()
    print(r.dp(4))
    r = PwReg(reg)
    r.optimize()
    print(r.dp(4))
