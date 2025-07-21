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
        __translation_dimension = None
        __offsets = None
        __factors = None

        def __init__(self, translation_dimension=None):
            self.__translation_dimension = translation_dimension

        def normalize(self, y):
            self._ensure_translation_dimension()

            self.__offsets = np.min(y, axis=1)
            y -= self.__offsets[:, np.newaxis]
            self.__factors = np.max(y, axis=1)
            return y / self.__factors[:, np.newaxis]

        def denormalize(self, models):
            self._ensure_translation_dimension()

            if self.__offsets is None or self.__factors is None:
                raise RuntimeError("'normalize' was not called before 'denormalize'")

            res = models * self.__factors[np.newaxis, :]
            res[self.__translation_dimension] += self.__offsets
            return res

        def __call__(self, x):
            return np.array(x, ndmin=2).T

        def interpolate(self, s1, s2, x1, x2):
            raise RuntimeError(
                f"interpolation is not possible with '{self.__class__.__name__}' kernel"
            )

        def _ensure_translation_dimension(self):
            if self.__translation_dimension is None:
                raise RuntimeError(
                    f"normalization without specifying 'translation_dimension' is not possible with"
                    f" '{self.__class__.__name__}' kernel"
                )

    class Poly(Raw):
        def __init__(self, degree=1, combinations=True):
            super().__init__(translation_dimension=0)
            self.__degree = degree
            self.__combinations = combinations

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
            x_start = self(np.array(x1[-1], ndmin=2))
            x_end = self(np.array(x2[0], ndmin=2))
            y_start = np.matmul(s1, x_start).T[0]
            y_end = np.matmul(s2, x_end).T[0]

            if x_start.shape[1] > self.__degree + 1:
                RuntimeError(
                    f"interpolation of multidimensional data is not possible with "
                    f"'{self.__class__.__name__}' kernel"
                )

            slopes = y_end - y_start
            offsets = y_start - x_start[1] * slopes
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
        res = np.matmul(self.__models, self.__kernel(np.array([x], dtype=self.__models.dtype))).T[0]
        return res[0] if self.__flatten else res

    @property
    def rss(self):
        return self.__errors.copy()

    @property
    def mse(self):
        return 0 if self.samplecount == 0 else self.rss / self.samplecount

    @property
    def samplecount(self):
        return len(self.__x)

    @property
    def models(self):
        return self.__models.copy()

    @property
    def range(self):
        return (self.__x[0], self.__x[-1])

    @property
    def x(self):
        return self.__x.copy()

    @property
    def y(self):
        return self.__y.copy()

    def plot(self):
        # TODO
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

    @property
    def variants(self):
        return [
            Regression(
                self.__x,
                self.__y[:, variant],
                self.__kernel,
                self.__starts,
                self.__models[:, variant, :],
                self.__errors,
                True,
                self.__interpolate,
            )
            for variant in range(self.__y.shape[1])
        ]

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
            for reg, ax, style, istyle in zip(self.variants, axs, styles, istyles)
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
