import sys
from bisect import bisect
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, cast

import numpy as np
import numpy.typing as npt

from .libmvsr import Algorithm as Algorithm
from .libmvsr import Metric as Metric
from .libmvsr import Mvsr, MvsrArray, valid_dtypes
from .libmvsr import Placement as Placement
from .libmvsr import Score as Score

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison
else:
    SupportsRichComparison = object


class Interpolate:
    @staticmethod
    def left(_x, segments):
        return [1.0] + [0.0] * (len(segments) - 1)

    @staticmethod
    def right(_x, segments):
        return [0.0] * (len(segments) - 1) + [1.0]

    @staticmethod
    def closest(x, segments):
        index = np.argmin(
            [
                min([sum((np.array(x, ndmin=1) - np.array(sx, ndmin=1)) ** 2) for sx in segment.x])
                for segment in segments
            ]
        )
        result = np.zeros((len(segments)))
        result[index] = 1.0
        return result

    @staticmethod
    def smooth(x, segments):
        distance = segments[1].x[0] - segments[0].x[-1]
        x_normalized = (x - segments[0].x[-1]) / distance
        result = 3 * x_normalized**2 - 2 * x_normalized**3
        return [1 - result] + [0] * (len(segments) - 2) + [result]


class Kernel:
    class Raw:
        """Raw Kernel to be used as a base class for other Kernel types.

        Implements pass-through transformation of x values and normalization of y values.
        Does not implement interpolation.

        Args:
            translation_dimension: Index of the model dimension that translates the regression along
                the y axis (required for normalization). Defaults to :obj:`None`.
            model_interpolation: Function to interpolate between neighbouring segments.
        """

        _translation_dimension: int | None = None
        _offsets: MvsrArray | None = None
        _factors: MvsrArray | None = None

        def __init__(
            self, translation_dimension: int | None = None, model_interpolation=Interpolate.closest
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
            if self._model_interpolation:
                interpolator = Kernel.ModelInterpolater(self, self._model_interpolation, segments)
                return interpolator.interpolate(segments)

            raise RuntimeError(
                f"interpolation is not possible with '{self.__class__.__name__}' kernel"
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
            degree: Degree.
            model_interpolation: Function to interpolate between neighbouring segments.
        """

        def __init__(self, degree: int = 1, model_interpolation=None):
            super().__init__(translation_dimension=0, model_interpolation=model_interpolation)
            self._degree = degree

        def __call__(self, x: npt.ArrayLike):  # [1,2,3] or [[1,1],[2,2],[3,3]]
            x = super().__call__(x)
            return np.concatenate(
                (
                    np.ones((1, x.shape[1])),
                    *([np.power(val, i)] for val in x for i in range(1, self._degree + 1)),
                )
            )

        def interpolate(self, segments: list["Segment"]):
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
                    f"interpolation of multidimensional data without lerp "
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
                np.empty(0), np.empty(0), model, np.empty(0), self, segments[0]._keepdims
            )

    # Helper  to support lerping between multiple models, should not be used as input kernel
    class ModelInterpolater:
        def __init__(self, kernel, lerp, segments):
            self._kernel = kernel
            self._lerp = lerp
            self._segments = segments

        def __call__(self, x: npt.ArrayLike):
            kx = self._kernel(x)
            lx = np.array([self._lerp(x, self._segments) for x in x]).T
            return np.concatenate([kx * l for l in lx])

        def interpolate(self, segments: list["Segment"]):
            return Segment(
                np.empty(0),
                np.empty(0),
                np.concatenate([segment.get_model(True) for segment in segments], axis=1),
                np.empty(0),
                Kernel.ModelInterpolater(self._kernel, self._lerp, segments),
                segments[0]._keepdims,
            )


class Segment:
    def __init__(
        self,
        x: MvsrArray,
        y: MvsrArray,
        model: MvsrArray,
        errors: MvsrArray,
        kernel: Kernel.Raw | Kernel.ModelInterpolater,
        keepdims: bool,
    ):
        self._x = x
        self._y = y
        self._model = model
        self._errors = errors
        self._kernel = kernel
        self._keepdims = keepdims

    def __call__(self, x: Any, *, keepdims=None):
        result = np.matmul(self._model, self._kernel([x])).T[0]
        keepdims = self._keepdims if keepdims is None else keepdims
        return result if keepdims else result[0]

    @property
    def rss(self):
        result = self._errors.copy()
        return result if self._keepdims else result[0]

    @property
    def mse(self):
        result = self._errors * 0 if self.samplecount == 0 else self._errors / self.samplecount
        return result if self._keepdims else result[0]

    @property
    def samplecount(self):
        return len(self._x)

    def get_model(self, keepdims=None):
        keepdims = self._keepdims if keepdims is None else keepdims
        result = self._model.copy()
        return result if len(result) > 1 or keepdims else result[0]

    model = property(get_model)

    @property
    def range(self):
        return (self._x[0], self._x[-1])

    @property
    def x(self):
        return self._x.copy()

    @property
    def y(self):
        return self._y.copy()

    def plot(self, ax, xvals=1000, style={}):
        def testmapping(**a):
            pass

        # get one axis object per variant
        try:
            _ = iter(ax)
        except TypeError:
            ax = [ax] * self._y.shape[0]

        # get one style per variant
        try:
            testmapping(**style)
            style = [style] * self._y.shape[0]
        except:
            pass

        try:
            _ = iter(xvals)
        except TypeError:
            xvals = [(self._x[0] + (self._x[-1] * i - self._x[0] * i) / (xvals - 1)) for i in range(xvals)]

        yvals = np.matmul(self._model, self._kernel(xvals))
        return [ax.plot(xvals, y, **sty) for ax,y,sty in zip(ax,yvals,style)]

class Regression:
    def __init__(
        self,
        x: npt.ArrayLike,
        y: MvsrArray,
        kernel: Kernel.Raw,
        starts: npt.NDArray[np.uintp],
        models: MvsrArray,
        errors: MvsrArray,
        keepdims: bool,
        sortkey = None
    ):
        self._x = x = np.array(x, dtype=object)
        self._y = y
        self._kernel = kernel
        self._starts = starts
        self._models = models
        self._errors = errors
        self._keepdims = keepdims
        self._sortkey = (lambda x: x) if sortkey is None else sortkey

        self._ends = np.concatenate((starts[1:], np.array([x.shape[0]], dtype=np.uintp))) - 1
        self._samplecounts = self._ends - self._starts
        self._start_values = x[self._starts]
        self._end_values = x[self._ends]

    def get_segment_index(self, x: Any):
        index = bisect(self._start_values[1:], self._sortkey(x), key=self._sortkey)
        if self._sortkey(self._end_values[index]) < self._sortkey(x):
            return (index, index + 1)
        return (index,)

    def get_segment_by_index(self, index):
        return self[index[0]] if len(index) == 1 else self._kernel.interpolate([self[i] for i in index])

    def get_segment(self, x: Any):
        index = self.get_segment_index(x)
        return self.get_segment_by_index(index)

    @property
    def starts(self):
        return self._starts.copy()

    @property
    def segments(self) -> Sequence[Segment]:
        return [segment for segment in self]

    @property
    def variants(self):
        return [
            Regression(
                self._x,
                self._y[variant:variant+1],
                self._kernel,
                self._starts,
                self._models[:, variant:variant+1, :],
                self._errors[:, variant:variant+1],
                False,
                self._sortkey
            )
            for variant in range(self._y.shape[0])
        ]

    def plot(self, ax, xvals=1000, *, style={}, istyle=None):
        def testmapping(**a):
            pass

        # get one axis object per variant
        try:
            _ = iter(ax)
        except TypeError:
            ax = [ax] * self._y.shape[0]

        # get one style per variant
        try:
            testmapping(**style)
            style = [style] * self._y.shape[0]
        except:
            pass

        # get one istyle per variant
        if istyle is None:
            istyle = [{**s, 'linestyle':'dotted', 'alpha':0.5} for s in style]
        else:
            try:
                testmapping(**istyle)
                istyle = [istyle] * self._y.shape[0]
            except:
                pass
        
        # instantiate styles
        mpl = sys.modules.get('matplotlib')
        norm_kwargs = mpl.cbook.normalize_kwargs
        l2d = mpl.lines.Line2D
        for a,s,i in zip(ax,style,istyle):
            snorm = norm_kwargs(s, l2d)
            inorm = norm_kwargs(i, l2d)
            changing_props = a._get_lines._getdefaults(
                {k:v if v is not None else inorm[k] for k,v in snorm.items() if k in inorm}
            )
            s.clear()
            i.clear()
            s.update(changing_props | snorm)
            i.update(changing_props | inorm)

        # find desired xvals
        try:
            _ = iter(xvals)
        except TypeError:
            xvals = [(self._x[0] + (self._x[-1] * i - self._x[0] * i) / (xvals - 1)) for i in range(xvals)]

        # plot segments
        idx = [self.get_segment_index(x) for x in xvals]
        segs = {k:i+1 for i,k in enumerate(idx)}
        results = [[]] * len(segs)
        for seg,idxend in segs.items():
            if len(seg) == 1:
                seg_x = np.array([self._x[self._starts[seg[0]]]] + [x for i,x in zip(idx, xvals[:idxend]) if i == seg] + [self._x[self._ends[seg[0]]]])
                yvals = np.array([self[seg[0]](x, keepdims=True) for x in seg_x]).T
            else:
                seg_x = [x for i,x in zip(idx, xvals[:idxend]) if i == seg]
                cur_seg = self.get_segment_by_index(seg)
                yvals = [cur_seg(x, keepdims=True) for x in seg_x]
                
                # pre- and append neighbouring segment values
                seg_x = np.array([self[seg[0]]._x[-1]] + seg_x + [self[seg[-1]]._x[0]])
                yvals = np.array([self[seg[0]](seg_x[0], keepdims=True)] + yvals + [self[seg[1]](seg_x[-1], keepdims=True)])

                yvals = yvals.T

            for res, a, var_y, sty in zip(results, ax, yvals, (style if len(seg) == 1 else istyle)):
                res.append(a.plot(seg_x, var_y, **sty))
        return results

    def __call__(self, x: Any):
        return self.get_segment(x)(x)

    def __len__(self):
        return len(self._end_values)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < -len(self) or index >= len(self):
            raise IndexError(f"segment index '{index}' is out of range [{-len(self)}, {len(self)})")
        return Segment(
            self._x[self._starts[index] : int(self._ends[index]) + 1],
            self._y[:, self._starts[index] : int(self._ends[index]) + 1],
            self._models[index],
            self._errors[index],
            self._kernel,
            self._keepdims,
        )

    def __iter__(self) -> Iterator[Segment]:
        return (self[i] for i in range(len(self)))


def mvsr(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    k: int,
    *,  # Following arguments must be explicitly specified via names.
    kernel: Kernel.Raw = Kernel.Poly(1),
    algorithm: Algorithm | None = None,
    score: Score | None = None,
    metric: Metric = Metric.MSE,
    normalize: bool | None = None,
    weighting: npt.ArrayLike | None = None,
    dtype: valid_dtypes = np.float64,
    keepdims: bool = False,
    sortkey = None
) -> Regression:
    """Run multi-variant segmented regression on input data, reducing it to k piecewise segments.

    Args:
        x (numpy.typing.ArrayLike_): Array-like containing the x input values. This gets transformed
            into the internal X matrix by the selected kernel. Values may be of any type.
        y (numpy.typing.ArrayLike_): Array-like containing the y input values. Shape
            :code:`(n_samples,)` or :code:`(n_variants, n_samples)`.
        k: Target number of segments for the Regression.
        kernel (:class:`Kernel.Raw`): Kernel used to transform x values into the internal X matrix,
            as well as normalize and interpolate y values. Defaults to :obj:`Kernel.Poly()` with
            :obj:`degree=1` and :obj:`lerp=None`.
        algorithm: Algorithm used to reduce the number of segments. Defaults to
            :obj:`Algorithm.GREEDY`.
        score: Placeholder for k scoring method (not implemented yet).
        metric: Placeholder for error metric (not implemented yet). Defaults to :obj:`Metric.MSE`.
        normalize: Normalize y input values. If :obj:`None`, auto-enabled for multi-variant input
            data. Defaults to :obj:`None`.
        weighting (numpy.typing.ArrayLike_): Optional per-variant weights. Defaults to :obj:`None`.
        dtype (numpy.float32_ | numpy.float64_): Internally used :obj:`numpy` data type. Defaults to
            `numpy.float64`_.
        keepdims: If set to False, return scalar values when evaluating single-variant segments.
            Defaults to :obj:`False`.
        sortkey: If the x values are not compareable, this function returns a key to sort.
            Defaults to :obj:`None`.

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

    dimensions, _n_samples_x = x_data.shape
    samples_per_segment = dimensions if algorithm == Algorithm.GREEDY else 1
    n_variants, _n_samples_y = y_data.shape
    keepdims = n_variants > 1 or keepdims

    if algorithm is None:
        algorithm = Algorithm.DP if dimensions * k * 10 > _n_samples_x else Algorithm.GREEDY

    with Mvsr(x_data, y_data, samples_per_segment, Placement.ALL, dtype) as regression:
        regression.reduce(k, alg=algorithm, score=score or Score.EXACT, metric=metric)
        if algorithm == Algorithm.GREEDY and dimensions > 1:
            regression.optimize()

        (starts, models, errors) = regression.get_data()
        if weighting is not None:
            models /= weighting
        if normalize:
            models = np.array([kernel.denormalize(model).T for model in models])
        else:
            models = np.transpose(models, (0, 2, 1))

        # Need to recalculate error in order to get errors per variant
        errors = np.array([[np.sum((np.matmul(model[vi], x_data[:,s:e])-vy[s:e])**2) for vi,vy in enumerate(y)] for s,e,model in zip(starts, (list(starts)+[len(x)])[1:], models)])
        
        return Regression(
            x,
            y,
            kernel,
            np.array(starts, dtype=np.uintp),
            models,
            errors,
            keepdims,
            sortkey
        )
