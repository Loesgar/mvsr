from bisect import bisect
from enum import Enum
from typing import Any, Iterator, Sequence, cast

import numpy as np
import numpy.typing as npt

from .libmvsr import Algorithm as Algorithm
from .libmvsr import Metric as Metric
from .libmvsr import Mvsr, MvsrArray, valid_dtypes
from .libmvsr import Placement as Placement
from .libmvsr import Score as Score


class Kernel:
    class Raw:
        __translation_dimension: int | None = None
        __offsets: MvsrArray | None = None
        __factors: MvsrArray | None = None

        def __init__(self, translation_dimension: int | None = None):
            self.__translation_dimension = translation_dimension

        def normalize(self, y: MvsrArray):
            self._ensure_translation_dimension()

            self.__offsets = cast(MvsrArray, np.min(y, axis=1))
            y = y - self.__offsets[:, np.newaxis]
            self.__factors = cast(MvsrArray, np.max(y, axis=1))
            return y / self.__factors[:, np.newaxis]

        def denormalize(self, models: MvsrArray):
            self._ensure_translation_dimension()

            if self.__offsets is None or self.__factors is None:
                raise RuntimeError("'normalize' was not called before 'denormalize'")

            result = models * self.__factors[np.newaxis, :]
            result[self.__translation_dimension] += self.__offsets
            return result

        def __call__(self, x: npt.ArrayLike) -> npt.NDArray[Any]:
            return np.array(x, ndmin=2)

        def interpolate(
            self, s1: MvsrArray, s2: MvsrArray, x1: npt.NDArray[Any], x2: npt.NDArray[Any]
        ) -> MvsrArray:
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
        def __init__(self, degree: int = 1, combinations: bool = True):
            super().__init__(translation_dimension=0)
            self.__degree = degree
            self.__combinations = combinations

        def __call__(self, x: npt.ArrayLike):  # [1,2,3] or [[1,1],[2,2],[3,3]]
            # TODO: handle combinations!
            x = np.array(x)
            x = x if len(x.shape) > 1 else np.array(x, ndmin=2).T
            return np.concatenate(
                (
                    np.ones((1, len(x))),
                    *([np.power(val, i)] for val in x.T for i in range(1, self.__degree + 1)),
                )
            )

        def interpolate(
            self, s1: MvsrArray, s2: MvsrArray, x1: npt.NDArray[Any], x2: npt.NDArray[Any]
        ):
            x_start = self(x1[-1])
            x_end = self(x2[0])
            y_start = np.matmul(s1, x_start).T[0]
            y_end = np.matmul(s2, x_end).T[0]

            if x_start.shape[1] > self.__degree + 1:
                RuntimeError(
                    f"interpolation of multidimensional data is not possible with "
                    f"'{self.__class__.__name__}' kernel"
                )

            slopes = y_end - y_start
            offsets = y_start - x_start[1] * slopes
            result = np.zeros((s1.shape))
            result[:, 0] = offsets
            result[:, 1] = slopes
            return result


class Segment:
    def __init__(
        self,
        x: MvsrArray,
        y: MvsrArray,
        model: MvsrArray,
        errors: MvsrArray,
        kernel: Kernel.Raw,
        keep_y_dims: bool,
    ):
        self.__x = x
        self.__y = y
        self.__model = model
        self.__errors = errors
        self.__kernel = kernel
        self.__keep_y_dims = keep_y_dims

    def __call__(self, x: Any):
        result = np.matmul(self.__model, self.__kernel([x])).T[0]
        return result if self.__keep_y_dims else result[0]

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
    def model(self):
        return self.__model.copy()

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
    def __init__(
        self,
        x: npt.ArrayLike,
        y: MvsrArray,
        kernel: Kernel.Raw,
        starts: npt.NDArray[np.uintp],
        models: MvsrArray,
        errors: MvsrArray,
        keep_y_dims: bool,
        interpolate: Interpolate,
    ):
        self.__x = x = np.array(x, dtype=object)
        self.__y = y
        self.__kernel = kernel
        self.__starts = starts
        self.__models = models
        self.__errors = errors  # TODO: recalculate?
        self.__keep_y_dims = keep_y_dims
        self.__interpolate = interpolate

        self.__ends = np.concatenate((starts[1:], np.array([x.shape[0]], dtype=np.uintp))) - 1
        self.__samplecounts = self.__ends - self.__starts
        self.__start_values = x[self.__starts]
        self.__end_values = x[self.__ends]

    def get_segment_index(self, x: Any):
        index = bisect(self.__start_values[1:], x)
        if self.__end_values[index] < x:
            return (index, index + 1)
        return (index,)

    def get_segment(self, x: Any):
        index = self.get_segment_index(x)
        if len(index) == 1:
            return self[index[0]]

        match self.__interpolate:
            case Interpolate.INTERPOLATE:
                return Segment(
                    np.empty(0),
                    np.empty(0),
                    self.__kernel.interpolate(
                        self.__models[index[0]],
                        self.__models[index[1]],
                        self.__x[self.__starts[index[0]] : int(self.__ends[index[0]]) + 1],
                        self.__x[self.__starts[index[1]] : int(self.__ends[index[1]]) + 1],
                    ),
                    np.empty(0),
                    self.__kernel,
                    self.__keep_y_dims,
                )
            case Interpolate.CLOSEST:
                left_distance = np.sum(np.power(x - self.__x[self.__ends[index[0]]], 2))
                right_distance = np.sum(np.power(x - self.__x[self.__starts[index[1]]], 2))
                return self[index[0] if left_distance < right_distance else index[1]]
            case Interpolate.LEFT:
                return self[index[0]]
            case Interpolate.RIGHT:
                return self[index[1]]

    @property
    def starts(self):
        return self.__starts.copy()

    @property
    def segments(self) -> Sequence[Segment]:
        return [segment for segment in self]

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
            for variant in range(self.__y.shape[0])
        ]

    """
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
    """

    def __call__(self, x: Any):
        return self.get_segment(x)(x)

    def __len__(self):
        return len(self.__end_values)

    def __getitem__(self, index: int):
        if index < -len(self) or index >= len(self):
            raise IndexError(f"segment index '{index}' is out of range [{-len(self)}, {len(self)})")
        return Segment(
            self.__x[self.__starts[index] : int(self.__ends[index]) + 1],
            self.__y[:, self.__starts[index] : int(self.__ends[index]) + 1],
            self.__models[index],
            self.__errors[index],
            self.__kernel,
            self.__keep_y_dims,
        )

    def __iter__(self) -> Iterator[Segment]:
        return (self[i] for i in range(len(self)))


def segreg(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    k: int,
    *,  # Following arguments must be explicitly specified via names.
    kernel: Kernel.Raw = Kernel.Poly(1),
    algorithm: Algorithm = Algorithm.GREEDY,
    score: Score | None = None,  # TODO: unused atm
    metric: Metric = Metric.MSE,  # TODO: unused atm
    normalize: bool | None = None,
    weighting: npt.ArrayLike | None = None,
    dtype: valid_dtypes = np.float64,
    keep_y_dims: bool = False,
    interpolate: Interpolate | bool = False,
) -> Regression:
    x_data = kernel(x)
    y = np.array(y, ndmin=2, dtype=dtype)

    normalize = normalize or y.shape[0] != 1 or weighting is not None
    y_data = np.array(kernel.normalize(y), dtype=dtype) if normalize else y.copy()

    if weighting is not None:
        weighting = np.array(weighting, dtype=dtype)
        y_data *= weighting[:, np.newaxis]

    dimensions, _n_samples_x = x_data.shape
    samples_per_segment = dimensions if algorithm == Algorithm.GREEDY else 1
    n_variants, _n_samples_y = y_data.shape
    keep_y_dims = n_variants > 1 or keep_y_dims

    if interpolate is True:
        interpolate = Interpolate.INTERPOLATE
    elif interpolate is False:
        interpolate = Interpolate.CLOSEST

    with Mvsr(x_data, y_data, samples_per_segment, Placement.ALL, dtype) as regression:
        regression.reduce(k, alg=algorithm)
        if algorithm == Algorithm.GREEDY and dimensions > 1:
            regression.optimize()

        (starts, models, errors) = regression.get_data()
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
            np.array(starts, dtype=np.uintp),
            models,
            errors,  # TODO: recalculate
            keep_y_dims,
            interpolate,
        )
