from bisect import bisect
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, cast, overload

import numpy as np
import numpy.typing as npt

from .libmvsr import Algorithm as Algorithm
from .libmvsr import Metric as Metric
from .libmvsr import Mvsr, MvsrArray, valid_dtypes
from .libmvsr import Placement as Placement
from .libmvsr import Score as Score

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison
    from matplotlib.axes import Axes
else:
    SupportsRichComparison = Any

_ModelInterpolation = Callable[[npt.ArrayLike, list["Segment"]], list[float]]


class Interpolate:
    @staticmethod
    def left(_x: npt.ArrayLike, segments: list["Segment"]):
        """Always use the leftmost (first) Segment for interpolating."""
        return [1.0] + [0.0] * (len(segments) - 1)

    @staticmethod
    def right(_x: npt.ArrayLike, segments: list["Segment"]):
        """Always use the rightmost (last) Segment for interpolating."""
        return [0.0] * (len(segments) - 1) + [1.0]

    @staticmethod
    def closest(x: npt.ArrayLike, segments: list["Segment"]):
        """Use the Segment that is closest to :obj:`x` for interpolating."""
        index = np.argmin(
            [
                min([sum((np.array(x, ndmin=1) - np.array(sx, ndmin=1)) ** 2) for sx in segment.xs])
                for segment in segments
            ]
        )
        result = np.zeros((len(segments)))
        result[index] = 1.0
        return result.tolist()

    @staticmethod
    def linear(x: npt.ArrayLike, segments: list["Segment"]):
        """Interpolate linearly between Segments based on :obj:`x`."""
        distance = segments[1].xs[0] - segments[0].xs[-1]
        x_normalized: float = (x - segments[0].xs[-1]) / distance
        return [1 - x_normalized] + [0.0] * (len(segments) - 2) + [x_normalized]

    @staticmethod
    def smooth(x: npt.ArrayLike, segments: list["Segment"]):
        """Interpolate smoothly (using a cubic function) between Segments based on :obj:`x`."""
        distance = segments[1].xs[0] - segments[0].xs[-1]
        x_normalized = (x - segments[0].xs[-1]) / distance
        result: float = 3 * x_normalized**2 - 2 * x_normalized**3
        return [1 - result] + [0.0] * (len(segments) - 2) + [result]


class Kernel:
    class Raw:
        """Raw Kernel to be used as a base class for other Kernel types.

        Implements pass-through transformation of x values, normalization of y values and
        interpolation between segments.

        Args:
            translation_dimension: Index of the model dimension that translates the regression along
                the y axis (required for normalization). Defaults to :obj:`None`.
            model_interpolation (typing.Callable[[numpy.typing.ArrayLike, list[Segment]], list[float]] | None):
                Function to interpolate between neighbouring segments. Defaults to
                :func:`Interpolate.closest`.
        """

        _translation_dimension: int | None = None
        _offsets: MvsrArray | None = None
        _factors: MvsrArray | None = None

        def __init__(
            self,
            translation_dimension: int | None = None,
            model_interpolation: _ModelInterpolation | None = Interpolate.closest,
        ):
            self._translation_dimension = translation_dimension
            self._model_interpolation = model_interpolation

        def normalize(self, y: MvsrArray):
            """Normalize each y variant to a range of [0,1].

            Args:
                y (numpy.ndarray): Input y values. Shape :code:`(n_variants, n_samples)`

            Raises:
                RuntimeError: If :code:`translation_dimension` has not been specified.

            Returns:
                numpy.ndarray: Normalized y values.
            """
            self._ensure_translation_dimension()

            self._offsets = cast(MvsrArray, np.min(y, axis=1))
            y = y - self._offsets[:, np.newaxis]
            self._factors = cast(MvsrArray, np.max(y, axis=1))
            return y / self._factors[:, np.newaxis]

        def denormalize(self, models: MvsrArray):
            """Denormalize models derived from values previously normalized with :meth:`normalize`.

            Args:
                models (numpy.ndarray): Models for regression segments.

            Raises:
                RuntimeError: If :meth:`normalize` has not been called on this kernel before.

            Returns:
                numpy.ndarray: Denormalized segment models.
            """
            self._ensure_translation_dimension()

            if self._offsets is None or self._factors is None:
                raise RuntimeError("'normalize' was not called before 'denormalize'")

            result = models * self._factors[np.newaxis, :]
            result[self._translation_dimension] += self._offsets
            return result

        def __call__(self, x: npt.ArrayLike) -> npt.NDArray[Any]:
            """Convert input array of x values to numpy array of dimensions.

            Args:
                x (numpy.typing.ArrayLike_): Input x values.

            Returns:
                numpy.ndarray: Internal X matrix to use with :class:`libmvsr.Mvsr`.
            """
            x = np.array(x)
            return x.T if len(x.shape) > 1 else np.array(x, ndmin=2)

        def interpolate(self, segments: list["Segment"]) -> "Segment":
            """Create interpolated :class:`Segment` using the provided :obj:`model_interpolation`.

            Args:
                segments: List of segments to be interpolated between.

            Raises:
                RuntimeError: If :obj:`model_interpolation` has not been specified.

            Returns:
                Segment: Interpolated segment.
            """
            if self._model_interpolation:
                interpolator = Kernel.ModelInterpolator(self, self._model_interpolation, segments)
                return interpolator.interpolate(segments)

            raise RuntimeError(
                f"native interpolation is not possible with '{self.__class__.__name__}' kernel"
            )

        def _ensure_translation_dimension(self):
            if self._translation_dimension is None:
                raise RuntimeError(
                    f"normalization without specifying 'translation_dimension' is not possible with"
                    f" '{self.__class__.__name__}' kernel"
                )

    class Poly(Raw):
        """Kernel for polynomial regression segments.

        Bases: :class:`Kernel.Raw`

        Inherited Methods: :meth:`normalize`, :meth:`denormalize`

        Args:
            degree: Polynomial degree.
            model_interpolation: (typing.Callable[[numpy.typing.ArrayLike, list[Segment]], list[float]] | None):
                Function to interpolate between neighbouring segments. If :obj:`None` interpolate
                linearly between segment endpoints. Defaults to :obj:`None`.
        """

        def __init__(self, degree: int = 1, model_interpolation: _ModelInterpolation | None = None):
            super().__init__(translation_dimension=0, model_interpolation=model_interpolation)
            self._degree = degree

        def __call__(self, x: npt.ArrayLike):
            x = super().__call__(x)
            return np.concatenate(
                (
                    np.ones((1, x.shape[1])),
                    *([np.power(val, i)] for val in x for i in range(1, self._degree + 1)),
                )
            )

        def interpolate(self, segments: list["Segment"]):
            """Create interpolated :class:`Segment`.

            Uses :obj:`model_interpolation` if provided else linearly interpolates between segment
            endpoints.

            Args:
                segments: List of segments to be interpolated between.

            Raises:
                RuntimeError: If :obj:`model_interpolation` is set to :obj:`None` and more than 2
                segments were provided or segments were constructed from multidimensional x values.

            Returns:
                Segment: Interpolated segment.
            """
            try:
                return super().interpolate(segments)
            except RuntimeError:
                pass

            if len(segments) > 2:  # pragma: no cover
                raise RuntimeError(
                    "interpolation of more than 2 segments is not possible with "
                    f"'{self.__class__.__name__}' kernel"
                )

            x_start = self([segments[0].range[1]])
            x_end = self([segments[1].range[0]])

            if x_start.shape[0] > self._degree + 1 or x_end.shape[0] > self._degree + 1:
                raise RuntimeError(
                    f"interpolation of multidimensional data without 'model_interpolation' "
                    f"is not possible with '{self.__class__.__name__}' kernel"
                )

            y_start = segments[0](segments[0].range[1])
            y_end = segments[1](segments[1].range[0])

            slopes = (y_end - y_start) / (x_end - x_start)[1]
            offsets = y_start - x_start[1] * slopes
            model = np.zeros(segments[0].get_model(True).shape)
            model[:, 0] = offsets
            model[:, 1] = slopes

            return Segment(
                np.empty(0), np.empty(0), self, model, np.empty(0), segments[0]._keepdims
            )

    class ModelInterpolator:
        """Helper to support interpolating between multiple models.

        Only for internal use, should not be used as input kernel.
        """

        def __init__(
            self,
            kernel: "Kernel.Raw",
            model_interpolation: _ModelInterpolation,
            segments: list["Segment"],
        ):
            self._kernel = kernel
            self._model_interpolation = model_interpolation
            self._segments = segments

        def __call__(self, x: npt.ArrayLike):
            xs = cast(Iterable[Any], x)
            kernel_xs = self._kernel(x)
            interpolation_weights = np.array(
                [self._model_interpolation(x, self._segments) for x in xs]
            )
            return np.concatenate([kernel_xs * weight for weight in interpolation_weights.T])

        def interpolate(self, segments: list["Segment"]):
            return Segment(
                np.empty(0),
                np.empty(0),
                Kernel.ModelInterpolator(self._kernel, self._model_interpolation, segments),
                np.concatenate([segment.get_model(True) for segment in segments], axis=1),
                np.empty(0),
                segments[0]._keepdims,
            )


class Segment:
    """Regression segment.

    Args:
        xs (numpy.ndarray): X input values.
        ys (numpy.ndarray): Y input values.
        kernel (:class:`Kernel.Raw`): Kernel used to transform x values.
        model (numpy.ndarray): Model matrix describing the segment.
        errors (numpy.ndarray):  Residual sum of squares for each segment sample.
        keepdims: If set to False, return scalar values when evaluating single-variant segments.
    """

    def __init__(
        self,
        xs: MvsrArray,
        ys: MvsrArray,
        kernel: Kernel.Raw | Kernel.ModelInterpolator,
        model: MvsrArray,
        errors: MvsrArray,
        keepdims: bool,
    ):
        self._xs = xs
        self._ys = ys
        self._model = model
        self._errors = errors
        self._kernel = kernel
        self._keepdims = keepdims

    def __call__(self, x: Any, keepdims: bool | None = None) -> MvsrArray:
        """Evaluate the segment for a given x value.

        Args:
            x: Input x value.
            keepdims: If set to False, return scalar values when the segment only has one variant.
                If None, use value provided from segment initialization. Defaults to None.

        Returns:
            numpy.ndarray: Predicted y value.
        """
        return self.predict([x], keepdims=keepdims)[0]

    def predict(self, xs: npt.ArrayLike, keepdims: bool | None = None):
        """Evaluate the regression for the given x values.

        Args:
            xs (numpy.typing.ArrayLike_): Input x values.
            keepdims: If set to False, return scalar values when the segment only has one variant.
                If None, use value provided from segment initialization. Defaults to None.

        Returns:
            numpy.ndarray: Predicted y values.
        """
        result = (self._model @ self._kernel(xs)).T
        keepdims = self._keepdims if keepdims is None else keepdims
        return result if keepdims else result[:, 0]

    @property
    def model(self):
        """numpy.ndarray: Model matrix describing the segment."""
        return self.get_model()

    def get_model(self, keepdims: bool | None = None):
        """Get the model matrix describing the segment.

        Args:
            keepdims: If set to False, return scalar values when the segment only has one variant.
                If None, use value provided from segment initialization. Defaults to None.

        Returns:
            numpy.ndarray: Model matrix.
        """
        keepdims = self._keepdims if keepdims is None else keepdims
        result = self._model.copy()
        return result if len(result) > 1 or keepdims else result[0]

    @property
    def range(self):
        """tuple[typing.Any, typing.Any]: Input x value range."""
        return (self._xs[0], self._xs[-1])

    @property
    def samplecount(self):
        """int: Number of samples."""
        return len(self._xs)

    @property
    def xs(self):
        """numpy.ndarray: Input x values."""
        return self._xs.copy()

    @property
    def ys(self):
        """numpy.ndarray: Input y values."""
        return self._ys.copy()

    @property
    def rss(self):
        """numpy.ndarray: Residual sum of squares, per sample."""
        result = self._errors.copy()
        return result if self._keepdims else result[0]

    @property
    def mse(self):
        """numpy.ndarray: Mean squared error, per sample."""
        result = self._errors * 0 if self.samplecount == 0 else self._errors / self.samplecount
        return result if self._keepdims else result[0]

    def plot(
        self,
        ax: "Axes" | Iterable["Axes"],
        xs: int | npt.ArrayLike = 1000,
        style: dict[str, Any] | Iterable[dict[str, Any] | None] = {},
    ):
        if not _is_iter(ax):
            ax = [ax] * self._ys.shape[0]
        axes = cast(Iterable["Axes"], ax)

        if _is_mapping(style):
            styles = [style] * self._ys.shape[0]
        else:
            style = cast(list[dict[str, Any] | None], style)
            styles = [{} if s is None else s for s in style]

        if not _is_iter(xs):
            xs = cast(int, xs)
            xs = [
                (self._xs[0] + (self._xs[-1] * i - self._xs[0] * i) / (xs - 1)) for i in range(xs)
            ]
        xs = cast(npt.ArrayLike, xs)

        ys = np.matmul(self._model, self._kernel(xs))
        return [ax.plot(xs, y, **style) for ax, y, style in zip(axes, ys, styles)]  # pyright: ignore


class Regression:
    """Regression consisting of multiple segments.

    Args:
        xs (numpy.typing.ArrayLike_): X input values.
        ys (numpy.ndarray): Y input values.
        kernel (:class:`Kernel.Raw`): Kernel used to transform x values.
        starts (numpy.ndarray): Segment start indices.
        models (numpy.ndarray): Segment models.
        errors (numpy.ndarray): Residual sum of squares for each sample.
        keepdims: If set to False, return scalar values when evaluating single-variant segments.
        sortkey: Function to extract a comparison key from x values. Defaults to :obj:`None`.
    """

    def __init__(
        self,
        xs: npt.ArrayLike,
        ys: MvsrArray,
        kernel: Kernel.Raw,
        starts: npt.NDArray[np.uintp],
        models: MvsrArray,
        errors: MvsrArray,
        keepdims: bool,
        sortkey: Callable[[Any], SupportsRichComparison] | None = None,
    ):
        self._xs = xs = np.array(xs, dtype=object)
        self._ys = ys
        self._kernel = kernel
        self._starts = starts
        self._models = models
        self._errors = errors
        self._keepdims = keepdims
        self._sortkey: Callable[[Any], Any] = (lambda x: x) if sortkey is None else sortkey

        self._ends = np.concatenate((starts[1:], np.array([xs.shape[0]], dtype=np.uintp))) - 1
        self._samplecounts = self._ends - self._starts
        self._start_values = xs[self._starts]
        self._end_values = xs[self._ends]

    @property
    def segments(self):
        """list[Segment]: List of :class:`Segment` objects."""
        return [segment for segment in self]

    @property
    def starts(self):
        """numpy.ndarray: Input sample indices of segment starts."""
        return self._starts.copy()

    @property
    def variants(self):
        """list[Regression]: List of :class:`Regression` objects for each variant."""
        return [
            Regression(
                self._xs,
                self._ys[variant : variant + 1],
                self._kernel,
                self._starts,
                self._models[:, variant : variant + 1, :],
                self._errors[:, variant : variant + 1],
                False,
                self._sortkey,
            )
            for variant in range(self._ys.shape[0])
        ]

    def get_segment(self, x: Any):
        """Get :class:`Segment` object for a given x value.

        Returns an interpolated :class:`Segment` if x is inbetween segments.

        Args:
            x: Input x value.

        Returns:
            Segment: Segment corresponding to x.
        """
        return self.get_segment_by_index(self.get_segment_index(x))

    def get_segment_index(self, x: Any) -> tuple[int, ...]:
        """Get segment indices for a given x value.

        Returns multiple indices if x is inbetween segments.

        Args:
            x: Input x value.

        Returns:
            tuple[int, ...]: Tuple of segment indices.
        """
        index = bisect(self._start_values[1:], self._sortkey(x), key=self._sortkey)
        if self._sortkey(self._end_values[index]) < self._sortkey(x):
            return (index, index + 1)
        return (index,)

    def get_segment_by_index(self, index: tuple[int, ...]):
        """Get :class:`Segment` object for the given indices.

        Returns an interpolated :class:`Segment` if multiple indices are provided.

        Args:
            index: Tuple of segment indices.

        Returns:
            Segment: Segment at the given index or interpolated segment.
        """
        return (
            self[index[0]]
            if len(index) == 1
            else self._kernel.interpolate([self[i] for i in index])
        )

    def plot(
        self,
        ax: "Axes" | Iterable["Axes"],
        xs: int | npt.ArrayLike | Iterable[Any] = 1000,
        style: dict[str, Any] | Iterable[dict[str, Any] | None] = {},
        istyle: dict[str, Any] | Iterable[dict[str, Any] | None] | None = None,
    ):
        from matplotlib.cbook import normalize_kwargs
        from matplotlib.lines import Line2D

        if not _is_iter(ax):
            ax = [ax] * self._ys.shape[0]
        axes = cast(Iterable["Axes"], ax)

        if _is_mapping(style):
            styles = [style] * self._ys.shape[0]
        else:
            style = cast(list[dict[str, Any] | None], style)
            styles = [{} if s is None else s for s in style]

        default_istyle = {"linestyle": "dotted", "alpha": 0.5}
        if istyle is None:
            istyles = [{**style, **default_istyle} for style in styles]
        else:
            if _is_mapping(istyle):
                istyles = [istyle] * self._ys.shape[0]
            else:
                istyle = cast(list[dict[str, Any] | None], style)
                istyles = [
                    {**style, **default_istyle} if i is None else i
                    for i, style in zip(istyle, styles)
                ]

        # instantiate styles
        for ax, style, istyle in zip(axes, styles, istyles):
            snorm = normalize_kwargs(style, Line2D)
            inorm = normalize_kwargs(istyle, Line2D)
            changing_props: dict[str, Any] = _ax_get_defaults(
                ax, {k: v if v is not None else inorm[k] for k, v in snorm.items() if k in inorm}
            )
            style.clear()
            istyle.clear()
            style.update(changing_props | snorm)
            istyle.update(changing_props | inorm)

        # find desired xvals
        if not _is_iter(xs):
            xs = cast(int, xs)
            xs = [
                (self._xs[0] + (self._xs[-1] * i - self._xs[0] * i) / (xs - 1)) for i in range(xs)
            ]
        xs = cast(Iterable[Any], xs)

        # plot segments
        segments: dict[tuple[int, ...], list[Any]] = {}
        for x in xs:
            segments.setdefault(self.get_segment_index(x), []).append(x)

        results: list[list[list[Line2D]]] = [[]] * len(segments)
        for segment_index, segment_xs in segments.items():
            segment = self.get_segment_by_index(segment_index)
            ys = segment.predict(segment_xs, keepdims=True)

            if is_interpolated := len(segment_index) != 1:
                prev_segment = self[segment_index[0]]
                next_segment = self[segment_index[-1]]
                segment_xs = [prev_segment.xs[-1], *segment_xs, next_segment.xs[0]]
                ys = np.array(
                    [
                        prev_segment(segment_xs[0], keepdims=True),
                        *ys,
                        next_segment(segment_xs[-1], keepdims=True),
                    ]
                )

            plot_styles = istyles if is_interpolated else styles
            for result, ax, variant_ys, style in zip(results, axes, ys.T, plot_styles):
                result.append(ax.plot(segment_xs, variant_ys, **style))  # pyright: ignore

        return results

    def __call__(self, x: Any):
        """Evaluate the regression for a given x value.

        Args:
            x: Input x value.

        Returns:
            numpy.ndarray: Predicted y value.
        """
        return self.get_segment(x)(x)

    def __len__(self):
        """Get the number of segments.

        Returns:
            int: Number of segments.
        """
        return len(self._end_values)

    @overload
    def __getitem__(self, index: int) -> Segment: ...
    @overload
    def __getitem__(self, index: slice) -> list[Segment]: ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < -len(self) or index >= len(self):
            raise IndexError(f"segment index '{index}' is out of range [{-len(self)}, {len(self)})")
        return Segment(
            self._xs[self._starts[index] : int(self._ends[index]) + 1],
            self._ys[:, self._starts[index] : int(self._ends[index]) + 1],
            self._kernel,
            self._models[index],
            self._errors[index],
            self._keepdims,
        )

    def __iter__(self) -> Iterator[Segment]:
        return (self[i] for i in range(len(self)))


def _ax_get_defaults(ax: "Axes", kw: dict[str, Any]):
    return cast(dict[str, Any], ax._get_lines._getdefaults(kw=kw, ignore=frozenset()))  # pyright: ignore


def _is_iter(value: Any):
    try:
        _ = iter(value)
        return True
    except TypeError:
        return False


def _is_mapping(value: Any):
    try:
        _ = {**value}
        return True
    except TypeError:
        return False


def mvsr(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    k: int,
    *,  # Following arguments must be explicitly specified via names.
    kernel: Kernel.Raw = Kernel.Poly(1),
    algorithm: Algorithm | None = None,
    score: Score | None = None,
    normalize: bool | None = None,
    weighting: npt.ArrayLike | None = None,
    dtype: valid_dtypes = np.float64,
    keepdims: bool = False,
    sortkey: Callable[[Any], SupportsRichComparison] | None = None,
) -> Regression:
    """Run multi-variant segmented regression on input data, reducing it to k piecewise segments.

    Args:
        x (numpy.typing.ArrayLike_): Array-like containing the x input values. This gets transformed
            into the internal X matrix by the selected kernel. Values may be of any type.
        y (numpy.typing.ArrayLike_): Array-like containing the y input values. Shape
            :code:`(n_samples,)` or :code:`(n_variants, n_samples)`.
        k: Target number of segments for the Regression.
        kernel (:class:`Kernel.Raw`): Kernel used to transform x values into the internal X matrix,
            as well as normalize and interpolate y values. Defaults to :obj:`Kernel.Poly()`.
        algorithm: Algorithm used to reduce the number of segments. If :obj:`None`, the algorithm
            will be selected automatically based on the number of samples, number of x dimensions
            and set :obj:`k`. Defaults to :obj:`None`.
        score: Placeholder for k scoring method (not implemented yet).
        normalize: Normalize y input values. If :obj:`None`, auto-enabled for multi-variant input
            data. Defaults to :obj:`None`.
        weighting (numpy.typing.ArrayLike_): Optional per-variant weights. Defaults to :obj:`None`.
        dtype (numpy.float32_ | numpy.float64_): Internally used :obj:`numpy` data type.
            Defaults to `numpy.float64`_.
        keepdims: If set to False, return scalar values when evaluating single-variant segments.
            Defaults to :obj:`False`.
        sortkey: If the x values are not comparable, this function is used to extract a comparison
            key for each of them. Defaults to :obj:`None`.

    Returns:
        :class:`Regression` object containing k segments.

    Raises:
        ValueError: If input dimensions of x, y, weighting are incompatible.
        RuntimeError: If normalization is enabled but the selected kernel does not support it.
    """

    x_data = kernel(x)
    y = np.array(y, ndmin=2, dtype=dtype)

    normalize = normalize or y.shape[0] != 1 or weighting is not None
    y_data = np.array(kernel.normalize(y), dtype=dtype) if normalize else y.copy()

    if weighting is not None:
        weighting = np.array(weighting, dtype=dtype)
        y_data *= weighting[:, np.newaxis]

    dimensions, n_samples_x = x_data.shape
    if algorithm is None:
        algorithm = Algorithm.DP if dimensions * k * 10 > n_samples_x else Algorithm.GREEDY

    samples_per_segment = dimensions if algorithm == Algorithm.GREEDY else 1
    n_variants, _n_samples_y = y_data.shape
    keepdims = n_variants > 1 or keepdims

    with Mvsr(x_data, y_data, samples_per_segment, Placement.ALL, dtype) as regression:
        regression.reduce(k, alg=algorithm, score=score or Score.EXACT)
        if algorithm == Algorithm.GREEDY and dimensions > 1:
            regression.optimize()

        (starts, models, _errors) = regression.get_data()
        if weighting is not None:
            models /= weighting
        if normalize:
            models = np.array([kernel.denormalize(model).T for model in models])
        else:
            models = np.transpose(models, (0, 2, 1))

        # Need to recalculate error in order to get errors per variant
        errors = np.array(
            [
                [
                    np.sum((np.matmul(model[i], x_data[:, start:end]) - variant_ys[start:end]) ** 2)
                    for i, variant_ys in enumerate(y)
                ]
                for start, end, model in zip(starts, [*starts[1:], n_samples_x], models)
            ]
        )

        return Regression(
            x, y, kernel, np.array(starts, dtype=np.uintp), models, errors, keepdims, sortkey
        )
