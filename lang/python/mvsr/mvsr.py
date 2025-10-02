from bisect import bisect
from enum import Enum
from typing import Any, Iterator, Sequence, cast

import numpy as np
import numpy.typing as npt

import sys

from .libmvsr import Algorithm as Algorithm
from .libmvsr import Metric as Metric
from .libmvsr import Mvsr, MvsrArray, valid_dtypes
from .libmvsr import Placement as Placement
from .libmvsr import Score as Score


class Lerp:
    def Left(x, segs):
        return [1.0] + [0.0]*(len(segs)-1)
    def Right(x, segs):
        return [0.0]*(len(segs)-1) + [1.0]
    def Closest(x, segs):
        idx = np.argmin([min([sum((np.array(x,ndmin=1)-np.array(sx,ndmin=1))**2) for sx in s.x]) for s in segs])
        res = np.zeros((len(segs)))
        res[idx] = 1.0
        return res
    def Smooth(x, segs):
        cur = x - segs[0].x[-1]
        dist = segs[1].x[0] - segs[0].x[-1]
        t = cur/dist
        res = 3*t*t-2*t*t*t
        return [1-res]+[0]*(len(segs)-2)+[res]


class Kernel:
    class Raw:
        _translation_dimension: int | None = None
        _offsets: MvsrArray | None = None
        _factors: MvsrArray | None = None

        def __init__(self, translation_dimension: int | None = None, *, lerp = Lerp.Closest):
            self._translation_dimension = translation_dimension
            self._lerp = lerp

        def normalize(self, y: MvsrArray):
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
            x = np.array(x)
            return x.T if len(x.shape) > 1 else np.array(x, ndmin=2)

        def interpolate(self, segments: list["Segment"]) -> "Segment":
            if self._lerp is None:
                raise RuntimeError(
                    f"interpolation is not possible with '{self.__class__.__name__}' kernel"
                )
            return Kernel.Lerper(self, self._lerp, segments).interpolate(segments)

        def _ensure_translation_dimension(self):
            if self._translation_dimension is None:
                raise RuntimeError(
                    f"normalization without specifying 'translation_dimension' is not possible with"
                    f" '{self.__class__.__name__}' kernel"
                )

    class Poly(Raw):
        def __init__(self, degree: int = 1, *, lerp=None):
            super().__init__(translation_dimension=0, lerp=lerp)
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
    class Lerper:
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
                np.empty(0), np.empty(0),
                np.concatenate([s.get_model(True) for s in segments],axis=1),
                np.empty(0),
                Kernel.Lerper(self._kernel, self._lerp, segments),
                segments[0]._keepdims
            )

class Segment:
    def __init__(
        self,
        x: MvsrArray,
        y: MvsrArray,
        model: MvsrArray,
        errors: MvsrArray,
        kernel: Kernel.Raw,
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
    algorithm: Algorithm = Algorithm.GREEDY,
    score: Score | None = None,  # TODO: unused atm
    metric: Metric = Metric.MSE,  # TODO: unused atm
    normalize: bool | None = None,
    weighting: npt.ArrayLike | None = None,
    dtype: valid_dtypes = np.float64,
    keepdims: bool = False,
    sortkey = None
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
    keepdims = n_variants > 1 or keepdims

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
