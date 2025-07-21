from bisect import bisect
from enum import Enum

import numpy as np

from .libmvsr import Algorithm as Algorithm
from .libmvsr import Metric as Metric
from .libmvsr import Mvsr
from .libmvsr import Placement as Placement
from .libmvsr import Score as Score


class Kernel:
    class Raw:
        __tidx = None
        __offsets = None
        __factors = None

        def __init__(self, translation_index=None):
            if translation_index is not None:
                self.__tidx = translation_index

        def normalize(self, y):
            if self.__tidx is None:
                raise NotImplementedError(
                    "Normalization is not possible by default on Raw kernel. Either specify the translation dimension index or consider turnign off normalization."
                )
            self.__offsets = np.min(y, axis=1)
            y -= self.__offsets[:, np.newaxis]
            self.__factors = np.max(y, axis=1)
            return y / self.__factors[:, np.newaxis]

        def denormalize(self, models):
            if self.__tidx is None:
                raise NotImplementedError(
                    "Normalization is not possible by default on Raw kernel. Either specify the translation dimension index or consider turnign off normalization."
                )
            if self.__offsets is None or self.__factors is None:
                raise NotImplementedError("Denormalization can not happen before normalization.")
            res = models * self.__factors[np.newaxis, :]
            res[self.__tidx] += self.__offsets
            return res

        def __call__(self, x):
            return np.array(x, ndmin=2).T

        def interpolate(self, s1, s2, x1, x2):
            raise NotImplementedError("Interpolation is not possible by default on Raw Kernel.")

    class Poly(Raw):
        def __init__(self, degree=1, combinations=True):
            super().__init__(translation_index=0)
            self.__degree = degree
            self.__conbinations = combinations

        def __call__(self, x):  # [1,2,3] or [[1,1],[2,2],[3,3]]
            # @TODO: handle combinations!
            x = np.array(x)
            x = x if len(x.shape) > 1 else np.array(x, ndmin=2).T
            return np.concatenate(
                (
                    np.ones((1, len(x))),
                    *([np.power(val, i)] for val in x.T for i in range(1, self.__degree + 1)),
                )
            )

        def interpolate(self, s1, s2, x1, x2):
            xstart = self(np.array(x1[-1], ndmin=2))
            xend = self(np.array(x2[0], ndmin=2))
            ystart = np.matmul(s1, xstart).T[0]
            yend = np.matmul(s2, xend).T[0]
            if xstart.shape[1] > self.__degree + 1:
                NotImplementedError(
                    "Interpolation is nor possible on Poly Kernel for multidimensional data."
                )
            slopes = yend - ystart
            offsets = ystart - xstart[1] * slopes
            res = np.zeros((s1.shape))
            res[:, 0] = offsets
            res[:, 1] = slopes
            return res


class Segment:
    def __init__(self, x, y, models, errors, kernel, flatten):
        self.__x = x
        self.__y = y
        self.__models = models
        self.__errors = errors
        self.__kernel = kernel
        self.__flatten = flatten

    def __call__(self, x):
        res = np.matmul(
            self.__models, np.array(self.__kernel(np.array([x])), dtype=self.__model.dtype)
        ).T[0]
        return res[0] if self.__flatten else res

    def get_rss(self):
        return self.__errors

    def get_mse(self):
        sc = self.get_samplecount()
        return 0 if sc == 0 else self.get_rss() / sc

    def get_samplecount(self):
        return len(self.__x)

    def get_model(self, variant=None):
        if variant is not None:
            if variant < 0 or variant >= len(self.__models):
                raise IndexError("Index out of range.")
            return self.__models[variant]
        return self.__models

    def get_range(self):
        return (self.__x[0], self.__x[-1])

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def plot(self):
        # TODO!
        pass


class Interpolate(Enum):
    INTERPOLATE = 0
    LEFT = 1
    RIGHT = 2
    CLOSEST = 3


class Regression:
    def __init__(self, x, y, kernel, starts, models, errors, flatten, interpolate):
        starts = np.append(starts, len(x))
        self.__x = x
        self.__y = y
        self.__kernel = kernel
        self.__flatten = flatten
        self.__starts = starts[:-1]
        self.__ends = [i - 1 for i in starts[1:]]
        self.__models = models
        self.__errors = errors  # todo: recalculate?
        self.__samplecount = [e - s for s, e in zip(starts[:-1], starts[1:])]
        self.__endvals = [x[idx] for idx in self.__ends]
        self.__startvals = [x[idx] for idx in self.__starts]
        self.__interpolate = interpolate

    def get_range(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")
        return (self.__startvals[idx], self.__endvals[idx])

    def get_segment_idx(self, x):
        idx = bisect(self.__startvals[1:], x)
        if self.__endvals[idx] < x:
            return (idx, idx + 1)
        return (idx,)

    def get_segment(self, x):
        idx = self.get_segment_idx(x)
        if len(idx) == 1:
            return self[idx[0]]

        match self.__interpolate:
            case Interpolate.INTERPOLATE:
                return Segment(
                    [],
                    [],
                    self.__kernel.interpolate(
                        self.__models[idx[0]],
                        self.__models[idx[1]],
                        self.__x[self.__starts[idx[0]] : self.__ends[idx[0]] + 1],
                        self.__x[self.__starts[idx[1]] : self.__ends[idx[1]] + 1],
                    ),
                    [],
                    self.__kernel,
                    self.__flatten,
                )
            case Interpolate.CLOSEST:
                left_distance = np.sum(np.pow(x - self.__x[self.__starts[idx[0]]], 2))
                right_distance = np.sum(np.pow(x - self.__x[self.__starts[idx[1]]], 2))
                return self[idx[0] if left_distance < right_distance else idx[1]]
            case Interpolate.LEFT:
                return self[idx[0]]
            case Interpolate.RIGHT:
                return self[idx[1]]

    def get_variant(self, variant):
        if variant < 0 or variant >= self.__y.shape[1]:
            raise IndexError("Variant index out of range")
        return Regression(
            self.__x,
            self.__y[:, variant],
            self.__kernel,
            self.__starts,
            self.__models[:, variant, :],
            self.__errors,
            True,
            self.__interpolate,
        )

    def plot(self, axs, styles={}, istyles=None):
        # TODO
        istyles = styles if istyles is None else istyles
        try:
            _ = iter(axs)
        except TypeError:
            [axs] * self.__y.shape[1]
        try:
            _ = iter(styles)
        except TypeError:
            [styles] * self.__y.shape[1]
        try:
            _ = iter(istyles)
        except TypeError:
            [istyles] * self.__y.shape[1]

        return [
            reg.plot(ax, style, istyle)
            for reg, ax, style, istyle in zip(
                [self.get_variant(v) for v in range(self.__y.shape[1])], axs, styles, istyles
            )
        ]

    def __call__(self, x):
        return self.get_segment(x)(x)

    def __len__(self):
        return len(self.__endvals) + 1

    def __getitem__(self, idx):
        if idx < 0 or idx > len(self):
            raise IndexError("Index out of range.")
        return Segment(
            self.__x[self.__starts[idx] : self.__ends[idx] + 1],
            self.__y[self.__starts[idx] : self.__ends[idx] + 1],
            self.__models[idx],
            self.__errors[idx],
            self.__kernel,
            self.__flatten,
        )


def segreg(
    x,
    y,
    k=None,
    *,  # Following arguments must be explicitly specified via names.
    kernel=Kernel.Poly(1),
    alg=Algorithm.GREEDY,
    score=None,
    metric=Metric.MSE,
    normalize=None,
    weighting=None,
    dtype=np.float64,
    donotflattenvariants=False,
    interpolate=None,
):
    x_dat = np.array(kernel(x), dtype=dtype)

    y = np.array(y, ndmin=2, dtype=dtype)
    normalize = True if y.shape[0] != 1 or weighting is not None else normalize
    y_norm = np.array(kernel.normalize(y), dtype=dtype) if normalize else y

    if weighting is not None:
        y_norm *= np.array(weighting, dtype=dtype)[:, np.newaxis]

    dimensions, num_samplesx = x_dat.shape
    variants, num_samplesy = y_norm.shape
    samples_per_seg = dimensions if alg == Algorithm.GREEDY else 1
    flatten = False if variants != 1 else not donotflattenvariants
    if interpolate is True:
        interpolate = Interpolate.INTERPOLATE
    elif interpolate is False:
        interpolate = Interpolate.CLOSEST

    with Mvsr(x_dat, y_norm, samples_per_seg, Placement.ALL, dtype) as reg:
        reg.reduce(k, alg=alg)
        if alg == Algorithm.GREEDY and dimensions > 1:
            reg.optimize()

        (starts, models, errors) = reg.get_data()
        if weighting is not None:
            models /= weighting
        if normalize:
            models = np.array([kernel.denormalize(model).T for model in models])
        else:
            models = np.transpose(models, (0, 2, 1))

        return Regression(
            x,
            y,
            kernel,
            np.array(starts, dtype=int),
            models,
            errors,  # TODO: recalculate
            flatten,
            interpolate,
        )
