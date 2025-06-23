import csv
import itertools
import math
from datetime import datetime
from pathlib import Path
from timeit import default_timer

import numpy as np

# disable rank warnings by numpy
import warnings

warnings.simplefilter("ignore", np.exceptions.RankWarning)

from juliacall import Main as jl
from juliacall import convert as jlconvert
from matplotlib import pyplot as plt

from polyreg import PwReg, get_ys

UINT32_MAX = np.iinfo(np.uint32).max


def with_noise(data, sigma, rng, rng_method="standard_normal"):
    if rng_method == "standard_normal":
        return data + sigma * rng.standard_normal(len(data))
    elif rng_method == "uniform":
        return data + sigma * rng.uniform(-1, 1, len(data))
    else:
        return data


def generate_continuous(
    n, d=2, segments=6, seed=1337, sigma=0, rng_method="standard_normal"
):
    rng = np.random.default_rng(seed)

    data = np.empty(n)
    half_segment_size = n // (segments * 2)

    segment_indices = np.linspace(0, n, segments + 1, dtype=np.int64)
    x_offsets = np.rint(
        rng.uniform(-1, 1, segments - 1) * 0.5 * half_segment_size
    ).astype(np.int64)

    # use non-equal-sized segments
    segment_indices[1:-1] += x_offsets

    start = rng.random()
    for seg_start, seg_end in zip(segment_indices[:-1], segment_indices[1:]):
        end = rng.random()
        data[seg_start:seg_end] = np.linspace(
            start, end, seg_end - seg_start, endpoint=False
        )
        start = end

    return np.array([np.ones(n), np.arange(n)]).T, data, with_noise(data, sigma, rng, rng_method)


def generate_continuous_md(
    n, d, segments=6, seed=1337, sigma=0, rng_method="standard_normal"
):
    rng = np.random.default_rng(seed)

    data = np.empty(n)
    half_segment_size = n // (segments * 2)

    segment_indices = np.linspace(0, n, segments + 1, dtype=np.int64)
    x_offsets = np.rint(
        rng.uniform(-1, 1, segments - 1) * 0.5 * half_segment_size
    ).astype(np.int64)

    # use non-equal-sized segments
    segment_indices[1:-1] += x_offsets
    X = np.concatenate(
        (
            np.array([np.ones(n), np.linspace(0, 1, n)]).T,
            [[rng.random() for _ in range(d - 2)] for _ in range(n)],
        ),
        axis=1,
        dtype=np.float64,
    )

    start = rng.random()
    for seg_start, seg_end in zip(segment_indices[:-1], segment_indices[1:]):
        beta = [0] + [(rng.random() - 0.5) * 20 for _ in range(d - 1)]
        y_start = sum([b * x for b, x in zip(beta, X[seg_start])])
        beta[0] = start - y_start

        data[seg_start:seg_end] = [
            sum([b * x for b, x in zip(beta, row)]) for row in X[seg_start:seg_end]
        ]
        if seg_end < n:
            start = sum([b * x for b, x in zip(beta, X[seg_end])])

    # normalize
    data -= np.min(data)
    data /= np.max(data)
    return X, data, with_noise(data, sigma, rng, rng_method)


def generate_non_continuous(
    n, segments=6, seed=1337, sigma=0, rng_method="standard_normal"
):
    rng = np.random.default_rng(seed)

    data = np.empty(n)
    half_segment_size = n // (segments * 2)

    segment_indices = np.linspace(0, n, segments + 1, dtype=np.int64)
    x_offsets = np.rint(
        rng.uniform(-1, 1, segments - 1) * 0.5 * half_segment_size
    ).astype(np.int64)

    # use non-equal-sized segments
    segment_indices[1:-1] += x_offsets

    # segment_indices[1:-1] = segment_indices[1:-1] // 2 * 2 # MOD 2 HACk
    # print(segment_indices) # DEBUG

    for seg_start, seg_end in zip(segment_indices[:-1], segment_indices[1:]):
        start = rng.random()
        end = rng.random()
        data[seg_start:seg_end] = np.linspace(
            start, end, seg_end - seg_start, endpoint=False
        )

    return np.array([np.ones(n), np.arange(n)]).T, data, with_noise(data, sigma, rng, rng_method)


def mean_squared_error(a: np.ndarray, b: np.ndarray):
    return ((a - b) ** 2).mean()


def max_error(a: np.ndarray, b: np.ndarray):
    return np.abs(a - b).max()


def models_from_starts(X, y, starts):
    split_X = np.split(X, starts[1:])
    split_y = np.split(y, starts[1:])
    return np.array(
        [np.linalg.lstsq(sec_X, sec_y)[0] for sec_X, sec_y in zip(split_X, split_y)]
    )


def time_fit_linear_dp(X, y, target_segments, sigma=None):
    jl_X = jlconvert(jl.Matrix[jl.Float64], X)
    jl_y = jlconvert(jl.Vector[jl.Float64], y)

    start = default_timer()
    results_linear_dp = jl.fit_linear_dp(jl_X, jl_y, target_segments)
    end = default_timer()

    # print([a.left_index - 1 for a in results_linear_dp]) # DEBUG
    starts = np.array([result.left_index - 1 for result in results_linear_dp])
    models = models_from_starts(X, y, starts)

    return end - start, get_ys(X, starts, models), (starts, models)


def time_fit_linear_merging_k(X, y, target_segments, sigma):
    jl_X = jlconvert(jl.Matrix[jl.Float64], X)
    jl_y = jlconvert(jl.Vector[jl.Float64], y)

    start = default_timer()
    results_linear_merging = jl.fit_linear_merging(
        jl_X,
        jl_y,
        sigma,
        target_segments,
        int(target_segments / 2),
        initial_merging_size=1,
    )
    end = default_timer()

    starts = np.array([result.left_index - 1 for result in results_linear_merging])
    models = models_from_starts(X, y, starts)

    return end - start, get_ys(X, starts, models), (starts, models)


def time_fit_linear_merging_2k(X, y, target_segments, sigma):
    jl_X = jlconvert(jl.Matrix[jl.Float64], X)
    jl_y = jlconvert(jl.Vector[jl.Float64], y)

    start = default_timer()
    results_linear_merging = jl.fit_linear_merging(
        jl_X, jl_y, sigma, target_segments * 2, target_segments, initial_merging_size=1
    )
    end = default_timer()

    starts = np.array([result.left_index - 1 for result in results_linear_merging])
    models = models_from_starts(X, y, starts)

    return end - start, get_ys(X, starts, models), (starts, models)


def time_fit_linear_merging_4k(X, y, target_segments, sigma):
    jl_X = jlconvert(jl.Matrix[jl.Float64], X)
    jl_y = jlconvert(jl.Vector[jl.Float64], y)

    start = default_timer()
    results_linear_merging = jl.fit_linear_merging(
        jl_X,
        jl_y,
        sigma,
        target_segments * 4,
        target_segments * 2,
        initial_merging_size=1,
    )
    end = default_timer()

    starts = np.array([result.left_index - 1 for result in results_linear_merging])
    models = models_from_starts(X, y, starts)

    return end - start, get_ys(X, starts, models), (starts, models)


def time_fit_polyreg(
    X, y, target_segments, sigma=None, optimize=True, optimize_segments=None
):
    start = default_timer()
    pwreg = PwReg(X, y)
    if not optimize_segments:
        optimize_segments = target_segments  # * 2
    if optimize:
        pwreg.reduce(optimize_segments)
        pwreg.optimize()
    if optimize_segments >= target_segments:
        pwreg.reduce(target_segments)
    end = default_timer()

    starts, _, _ = pwreg.getData()
    models = models_from_starts(X, y, starts)

    return end - start, get_ys(X, starts, models), (starts, models)


def benchmark_6(
    func,
    iters,
    start_iter,
    per_iter,
    sigmas,
    d,
    seed=1337,
    generate=generate_continuous_md,
    rng_method="standard_normal",
):
    rng = np.random.default_rng(seed)
    target_segments = 6
    times_per_sigma = [np.empty((per_iter, iters)) for _ in sigmas]
    errors_per_sigma = [np.empty((per_iter, iters)) for _ in sigmas]
    max_errors_per_sigma = [np.empty((per_iter, iters)) for _ in sigmas]
    for i, n in enumerate(2 ** (np.arange(iters) + start_iter)):
        print(n, end=" ", flush=True)
        for j in range(per_iter):
            for k, sigma in enumerate(sigmas):
                seed2 = rng.integers(UINT32_MAX)
                X, ys, ys_with_noise = generate(
                    n, d, target_segments, seed2, sigma, rng_method
                )
                time, results, _ = func(X, ys_with_noise, target_segments, sigma=sigma)
                times_per_sigma[k][j][i] = time
                errors_per_sigma[k][j][i] = mean_squared_error(ys, results)
                max_errors_per_sigma[k][j][i] = max_error(ys, results)
            print(".", end="", flush=True)
        print()

    return times_per_sigma, errors_per_sigma, max_errors_per_sigma


def pre_compile_julia():
    ### The following 3 lines can be commented out after the package is installed.
    import juliapkg

    juliapkg.add("StatsBase", "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91")
    juliapkg.resolve()

    jl.include(
        str(
            Path(__file__).parent
            / "related_work/fast_segmented_regression/src/linear_merging.jl"
        )
    )

    benchmark_6(time_fit_linear_dp, 1, 8, 1, (0.05,), 2)
    benchmark_6(time_fit_linear_merging_k, 1, 8, 1, (0.05,), 2)


ALGO_MAP = {
    "DP": time_fit_linear_dp,
    "M1K": time_fit_linear_merging_k,
    "M2K": time_fit_linear_merging_2k,
    "M4K": time_fit_linear_merging_4k,
    "PREG": time_fit_polyreg,
}
GENERATOR_MAP = {"CONT": generate_continuous_md, "NONCONT": generate_non_continuous}

INT32_MAX = np.iinfo(np.int32).max


def array2string_csv(array):
    string = np.array2string(array, max_line_width=INT32_MAX, threshold=INT32_MAX)
    return " ".join(string.replace(",", " ").split())


def benchmark_synthetic(
    algorithms, generators, sigmas, ns, repeats, seed, target_segments, ds, nd
):
    if ds is None:
        ds = [2]
        GENERATOR_MAP["CONT"] = generate_continuous
    for algo, time_fit in zip(algorithms, map(ALGO_MAP.get, algorithms)):
        rng = np.random.default_rng(seed)
        for generator, generate in zip(generators, map(GENERATOR_MAP.get, generators)):
            for sigma in sigmas:
                timestamp = datetime.now().isoformat(timespec="seconds")
                filename = (
                    "_".join(
                        (algo, generator, "S" + str(sigma).split(".")[-1], timestamp)
                    )
                    + ".csv"
                )
                print(algo, generator, sigma, end=" ", flush=True)
                with open(filename, "w", newline="") as file:
                    csvwriter = csv.writer(
                        file, dialect=csv.unix_dialect, quoting=csv.QUOTE_MINIMAL
                    )
                    for n, k, d in itertools.product(ns, target_segments, ds):
                        n = n * d if nd else n
                        print(f"[{n} {k} {d}]", end=" ", flush=True)
                        for _ in range(repeats):
                            seed2 = rng.integers(UINT32_MAX)
                            X, ys, ys_with_noise = generate(
                                n, d, k, seed2, sigma, "standard_normal"
                            )

                            time, results, (starts, models) = time_fit(
                                X, ys_with_noise, k, sigma=sigma
                            )
                            error = mean_squared_error(ys, results)
                            error2 = mean_squared_error(ys_with_noise, results)

                            csvwriter.writerow(
                                (
                                    n,
                                    k,
                                    d,
                                    algo,
                                    generator,
                                    sigma,
                                    seed2,
                                    f"{time:.9f}",
                                    f"{error:.6e}",
                                    f"{error2:.6e}",
                                    array2string_csv(starts),
                                    array2string_csv(models),
                                )
                            )
                            file.flush()
                print()


def benchmark_from_file(filepath: Path, algorithms, repeats, target_segments):
    if not filepath.is_file():
        print(f"error: failed to read '{args.txt}' (not a file)")
        return

    if len(target_segments) == 1:
        target_segments = np.arange(2, target_segments[0] + 1, dtype=np.int64)

    data = np.loadtxt(filepath)
    n = len(data)
    if len(data.shape) == 1:
        xs = np.linspace(0, 1, n)
        ys = data
    elif len(data.shape) > 2 or data.shape[1] != 2:
        print(
            f"error: unsupported data shape '{data.shape}' (supported: (n,) or (2, n))"
        )
        return
    else:
        xs, ys = data.T

    X = np.array([np.ones(n), xs]).T

    for algo, time_fit in zip(algorithms, map(ALGO_MAP.get, algorithms)):
        timestamp = datetime.now().isoformat(timespec="seconds")
        filename = "_".join((algo, "CSV", "NONE", timestamp)) + ".csv"
        print(algo, end=" ", flush=True)
        with open(filename, "w", newline="") as file:
            csvwriter = csv.writer(
                file, dialect=csv.unix_dialect, quoting=csv.QUOTE_MINIMAL
            )
            for k in target_segments:
                print(k, end=" ", flush=True)
                for _ in range(repeats):
                    time, results, (starts, models) = time_fit(
                        X, ys, k, sigma=0.0
                    )

                    error = math.nan
                    error2 = mean_squared_error(ys, results)

                    csvwriter.writerow(
                        (
                            n,
                            k,
                            2,
                            algo,
                            "TXT_FILE",
                            "NONE",
                            "NONE",
                            f"{time:.9f}",
                            f"{error:.6e}",
                            f"{error2:.6e}",
                            array2string_csv(starts),
                            array2string_csv(models),
                        )
                    )
                    file.flush()
        print()


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser()
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("DP", "M1K", "M2K", "M4K", "PREG"),
        default=["PREG"],
    )
    parser.add_argument(
        "--generators", nargs="+", choices=("CONT", "NONCONT"), default=["CONT"]
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.005])
    parser.add_argument("--ns", nargs="+", type=int, default=[])
    parser.add_argument("--nd", action=BooleanOptionalAction, default=False)
    parser.add_argument(
        "--range", nargs=2, type=int, metavar=("from", "to"), default=(6, 14)
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--target-segments", nargs="+", default=[6], type=int)
    parser.add_argument("--dimensions", nargs="+", default=None, type=int)
    parser.add_argument("--txt", type=Path, default=None, metavar="TXT_FILE")

    args = parser.parse_args()
    non_default_args = {
        arg.dest
        for arg in parser._option_string_actions.values()
        if hasattr(args, arg.dest) and getattr(args, arg.dest) != arg.default
    }
    pre_compile_julia()

    if args.txt:
        if ignored_args := non_default_args - {
            "txt",
            "target_segments",
            "algorithms",
            "repeats",
        }:
            print(
                f"warning: '{', '.join(('--' + arg for arg in ignored_args))}' "
                "is not compatible with '--txt' and will be ignored"
            )
        benchmark_from_file(
            args.txt, args.algorithms, args.repeats, args.target_segments
        )

    else:
        benchmark_synthetic(
            args.algorithms,
            args.generators,
            args.sigmas,
            args.ns if args.ns else 2 ** np.arange(args.range[0], args.range[1] + 1),
            args.repeats,
            args.seed,
            args.target_segments,
            args.dimensions,
            args.nd,
        )
else:
    pre_compile_julia()
