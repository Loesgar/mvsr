from contextlib import nullcontext as does_not_raise
from itertools import chain, product

import numpy as np
import pytest
from pytest import approx, raises

from mvsr import Algorithm, Interpolate, Kernel, mvsr
from mvsr.libmvsr import Metric, Mvsr, Score

# pyright: basic

Y = [1, 2, 3, 4, 5, 6, 7, 8, 2, 2, 2, 2, 2, 2, 1, 0, -1, -2, -3, -4]
X = list(range(len(Y)))
Y2 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
K = 3
STARTS = [0, 8, 13]
WEIGHTING = [0.1, 10.0]
STARTS_WEIGHTED = [0, 8, 10]


def test_simple_dp():
    assert mvsr(X, Y, K, algorithm=Algorithm.DP).starts.tolist() == STARTS


def test_simple_greedy():
    assert mvsr(X, Y, K, algorithm=Algorithm.GREEDY).starts.tolist() == STARTS


def test_simple_normalize():
    assert mvsr(X, Y, K, normalize=True).starts.tolist() == STARTS


def test_simple_weighting():
    assert mvsr(X, [Y, Y2], K, weighting=WEIGHTING).starts.tolist() == STARTS_WEIGHTED


def test_simple_poly2():
    assert mvsr(X, Y, K, kernel=Kernel.Poly(2)).starts.tolist() == STARTS


def test_interpolate():
    c = STARTS[1] - 0.5
    cl = c - 0.25
    cr = c + 0.25
    interp_results = [
        (Interpolate.INTERPOLATE, interpolate := [(cl, 6.5), (c, 5.0), (cr, 3.5)]),
        (True, interpolate),
        (Interpolate.LEFT, [(cl, 8.25), (c, 8.5), (cr, 8.75)]),
        (Interpolate.RIGHT, [(cl, 2.0), (c, 2.0), (cr, 2.0)]),
        (Interpolate.CLOSEST, closest := [(cl, 8.25), (c - 1e-9, 8.5), (c, 2.0), (cr, 2.0)]),
        (False, closest),
    ]

    for interpolate, results in interp_results:
        for x, y in results:
            assert mvsr(X, Y, K, interpolate=interpolate)(x) == approx(y)


def test_regression_and_segment():
    regression = mvsr(X, Y, K)
    assert len(regression) == len(regression.segments) == K
    assert regression[-K]
    for index in (-(K + 1), K):
        with raises(IndexError, match="segment index"):
            assert regression[index]

    for segment1, segment2 in zip(regression, regression.segments):
        assert (segment1.model == segment2.model).all()
        assert segment1.mse == segment2.mse
        assert segment1.range == segment2.range
        assert segment1.rss == segment2.rss
        assert segment1.samplecount == segment2.samplecount
        assert (segment1.x == segment2.x).all()
        assert (segment1.y == segment2.y).all()

    assert regression.starts.tolist() == STARTS
    assert len(regression.variants) == 1
    assert regression(3.5) == approx(4.5)


def test_keep_y_dims():
    assert isinstance(mvsr(X, Y, K)(0.0), float)
    assert len(y := mvsr(X, Y, K, keep_y_dims=True)(0.0)) == 1 and isinstance(y[0], float)


def test_kernels():
    models = mvsr(X, Y, K)._models

    for kernel in (Kernel.Raw(0), Kernel.Poly(1)):
        assert kernel(X).shape[1] == len(X)

        with raises(RuntimeError, match="'normalize' was not called before 'denormalize'"):
            kernel.denormalize(models)

    raw_kernel = Kernel.Raw()
    with raises(RuntimeError, match="normalization without specifying .* is not possible"):
        raw_kernel.denormalize(models)

    assert len(mvsr(X, Y, K, kernel=Kernel.Raw()).segments) == 3

    with raises(RuntimeError, match="interpolation is not possible"):
        regression = mvsr(X, Y, K, kernel=Kernel.Raw(), interpolate=True)
        regression(regression.starts[1] - 0.5)


TESTDATA_MVSR = chain(
    product(
        [Y],
        [Kernel.Raw(), Kernel.Raw(0), *(Kernel.Poly(d) for d in range(1, 3))],
        Algorithm,
        Score,
        Metric,
        [None, False, True],
        [None],
        [np.float32, np.float64],
        [False, True],
        [False, True],
    ),
    product(
        [[Y, Y2]],
        [Kernel.Raw(), Kernel.Raw(0), *(Kernel.Poly(d) for d in range(1, 3))],
        Algorithm,
        Score,
        Metric,
        [None, False, True],
        [None, WEIGHTING],
        [np.float32, np.float64],
        [False, True],
        [False, True],
    ),
)


@pytest.mark.parametrize(
    "y,kernel,algorithm,score,metric,normalize,weighting,dtype,keep_y_dims,interpolate",
    TESTDATA_MVSR,
)
def test_mvsr(
    y, kernel, algorithm, score, metric, normalize, weighting, dtype, keep_y_dims, interpolate
):
    match (len(y), kernel, normalize, bool(weighting), interpolate):
        case (_, _, True, *_) | (2, _, _, *_) | (_, _, _, True, _) if (
            type(kernel) is Kernel.Raw and kernel._translation_dimension is None
        ):
            expectation = raises(RuntimeError, match="normalization .* is not possible")
        case (_, _, _, _, True) if type(kernel) is Kernel.Raw:
            expectation = raises(RuntimeError, match="interpolation is not possible")
        case _:
            expectation = does_not_raise()

    with expectation:
        regression = mvsr(
            X,
            y,
            K,
            kernel=kernel,
            algorithm=algorithm,
            score=score,
            metric=metric,
            normalize=normalize,
            weighting=weighting,
            dtype=dtype,
            keep_y_dims=keep_y_dims,
            interpolate=interpolate,
        )
        for start in regression.starts[1:]:
            for offset in [-0.5, 0.5]:
                regression(start + offset)


def test_libmvsr():
    dtype = np.float64

    kernel = Kernel.Poly()
    x_data = kernel(X)
    y_data = np.array(Y, ndmin=2, dtype=dtype)

    dimensions, _n_samples_x = x_data.shape

    with raises(ValueError, match="unsupported input shape"):
        Mvsr(np.array(X), y_data, dimensions)

    with raises(ValueError, match="unsupported input shape"):
        Mvsr(x_data, np.array(Y), dimensions)

    with raises(ValueError, match="incompatible input shapes"):
        Mvsr(x_data[:, :-2], y_data, dimensions)

    with raises(TypeError, match="unsupported dtype"):
        Mvsr(x_data, y_data, dimensions, dtype=float)  # pyright: ignore

    with Mvsr(x_data, y_data, dimensions) as regression:
        regression.reduce(K, alg=Algorithm.GREEDY)
        regression.optimize()

        regression.copy()

        (starts, _models, _errors) = regression.get_data()
        assert starts.tolist() == STARTS
