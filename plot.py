import csv
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# Read runtimes from CSV files
def getTimes(alg, sigma, dir=sys.argv[1]):
    times = {}
    for path in Path(dir).glob(f"{alg}_CONT_S{sigma}*"):
        with open(path, newline='') as file:
            for (n, k, d, alg, gen, sigma, seed, time, err, noisy_err, starts, models) in csv.reader(file):
                times.setdefault(int(n), []).append(float(time))
    return dict(sorted(times.items()))

# Read errors from CSV files
def getErrs(alg, sigma, dir=sys.argv[1]):
    times = {}
    for path in Path(dir).glob(f"{alg}_CONT_S{sigma}*"):
        with open(path, newline='') as file:
            for (n, k, d, alg, gen, sigma, seed, time, err, noisy_err, starts, models) in csv.reader(file):
                times.setdefault(int(n), []).append(float(err))
    return dict(sorted(times.items()))

algs = ["DP", "PREG", "M1K", "M2K", "M4K"]
legend_names = ["Exact (DP)", "Our Approach", "Acharya", "Acharya 2k", "Acharya 4k"]
#algcolor = ["dodgerblue", "gold", "chocolate", "lightpink", "peachpuff"]
#algcolor = ["mediumblue", "darkgreen", "darkred", "chocolate", "goldenrod"]
algcolor = ["black", "red", "blue", "dodgerblue", "cyan"]
algalpha = [None, None, None, 0.5, 0.5]
algmarker = ["o", "^", "s", "D", "d"]
markersize = 5
plot_order = [3, 4, 2, 0, 1]
sigmas = ["01", "1"]
times = dict(zip(sigmas, [dict(zip(algs, [getTimes(alg, sigma) for alg in algs])) for sigma in sigmas]))
errors = dict(zip(sigmas, [dict(zip(algs, [getErrs(alg, sigma) for alg in algs])) for sigma in sigmas]))

# Definition for calculating our error bars
def get_percentiles(data, q=0.5):
    mean = np.mean(data)
    upper = np.quantile([d for d in data if d >= mean], q)
    lower = np.quantile([d for d in data if d <= mean], 1.0-q)
    return (mean, upper, lower)

# Generate eval plot (Fig. 3)
def plot():
    fig, axs = plt.subplots(len(sigmas), 4, figsize=np.array((8.5, 4.4)))# * 1.25)

    for ax, text in zip(axs[0], ["Runtime (s)", "MSE", "Relative Speedup", "Relative MSE"]):
        ax.annotate(
            text, xy=(0.5, 1), xytext=(0, 5),
            xycoords='axes fraction', textcoords='offset points',
            size='large', ha='center', va='baseline'
        )

    for ax in axs[len(sigmas) - 1]:
        ax.set_xlabel("number of samples (n)")

    for ax, s in zip(axs[:, 0], sigmas):
        ax.annotate(
            f"$\\sigma=0.{s}$", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center', rotation=90
        )

    # speed
    for s, axss in zip(sigmas, axs):
        ax = axss[0]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(.00003,1000)
        for alg, color, marker, alpha in np.array((*zip(algs, algcolor, algmarker, algalpha),), dtype=object)[plot_order]:
            means, upper_errs, lower_errs = np.array([
                get_percentiles(data) for data in times[s][alg].values()
            ]).T
            upper_errs = upper_errs - means
            lower_errs = means - lower_errs
            ax.errorbar(
                [i for i in times[s][alg].keys()][:len(means)], means, yerr=[lower_errs, upper_errs],
                capsize=3, color=color, marker=marker, markersize=markersize, alpha=alpha,
            )
        ax.relim()

    # error
    for s, axss in zip(sigmas, axs):
        ax = axss[1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        for alg, color, marker, alpha in np.array((*zip(algs, algcolor, algmarker, algalpha),), dtype=object)[plot_order]:
            means, upper_errs, lower_errs = np.array([
                get_percentiles(data) for n,data in errors[s][alg].items() if n < 300000
            ]).T
            upper_errs = upper_errs - means
            lower_errs = means - lower_errs
            ax.errorbar(
                [i for i in errors[s][alg].keys()][:len(means)], means, yerr=[lower_errs, upper_errs],
                capsize=3, color=color, marker=marker, markersize=markersize, alpha=alpha,
            )
        ax.relim()

    # relative speed
    for s, axss in zip(sigmas, axs):
        ax = axss[2]
        ax.set_xscale('log')
        ax.set_yscale('log')
        dp_means = [np.mean(data) for data in times[s]["DP"].values()]
        for alg, color, marker, alpha in np.array((*zip(algs, algcolor, algmarker, algalpha),), dtype=object)[plot_order]:
            means, upper_errs, lower_errs = np.array([
                get_percentiles([rel / t for rel,t in zip(reldata, data)])
                for reldata, data in zip(times[s]["DP"].values(), times[s][alg].values())
                #get_percentiles([rel / t for t in data])
                #for rel, data in zip(dp_means, times[s][alg].values())
            ]).T
            upper_errs = upper_errs - means
            lower_errs = means - lower_errs
            ax.errorbar(
                [i for i in times[s][alg].keys()][:len(means)], means, yerr=[lower_errs, upper_errs],
                capsize=3, color=color, marker=marker, markersize=markersize, alpha=alpha,
            )
        ax.relim()

    # relative error
    for s, axss in zip(sigmas, axs):
        ax = axss[3]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(.5, 100)
        dp_means = [np.mean(data) for data in errors[s]["DP"].values()]
        for alg, color, marker, alpha in np.array((*zip(algs, algcolor, algmarker, algalpha),), dtype=object)[plot_order]:
            #if alg == "M1K" and s == sigmas[0]:
            #    continue
            means, upper_errs, lower_errs = np.array([
                get_percentiles([e / rel for rel,e in zip(reldata, data)])
                for reldata, data in zip(errors[s]["DP"].values(), errors[s][alg].values())
                #get_percentiles([e / rel for e in data])
                #for rel, data in zip(dp_means, errors[s][alg].values())
            ]).T
            upper_errs = upper_errs - means
            lower_errs = means - lower_errs
            ax.errorbar(
                [i for i in errors[s][alg].keys()][:len(means)], means, yerr=[lower_errs, upper_errs],
                capsize=3, color=color, marker=marker, markersize=markersize, alpha=alpha,
            )
        ax.relim()
    
    for alg in errors["01"].keys():
        err_rel_vals = 0
        err_rel_num = 0
        for (n,dp), pa in zip(errors["01"]["DP"].items(), errors["01"][alg].values()):
            #if n < 10000:
            for dperr, alerr in zip(dp,pa):
                #for dperr, prerr in zip(dperrs, prerrs):
                err_rel_vals += alerr/dperr
                err_rel_num += 1
        print(f"Average Error of {alg} relativ to dp at 0.01 sigma: ", err_rel_vals / err_rel_num)

    handles = [
        mlines.Line2D([], [], color=color, label=legend_name, marker=marker, markersize=markersize)
        for legend_name, color, marker in zip(legend_names, algcolor, algmarker)
    ]
    fig.legend(handles=handles, ncol=len(handles), loc="lower center")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig("fig-eval.pdf")
    plt.show()

# Generate eval plot (Fig. 4)
def plot_time_vs_error():
    fig = plt.figure(figsize=np.array((4, 4.35)))# * 1.25)
    ax = fig.gca()

    sigma = "1" # tradeoff for signa = 0.1. Use "01" for 0.01

    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("MSE")

    ax.set_xscale("log")
    ax.set_yscale("log")
    for alg, color, marker, alpha in np.array((*zip(algs, algcolor, algmarker, algalpha),), dtype=object)[plot_order]:
        x_means, x_upper_errs, x_lower_errs = np.array([
            get_percentiles(data) for data in times[sigma][alg].values()
        ]).T
        x_upper_errs = x_upper_errs - x_means
        x_lower_errs = x_means - x_lower_errs

        y_means, y_upper_errs, y_lower_errs = np.array([
            get_percentiles(data) for data in errors[sigma][alg].values()
        ]).T
        y_upper_errs = y_upper_errs - y_means
        y_lower_errs = y_means - y_lower_errs

        ax.errorbar(
            x_means, y_means, xerr=[x_lower_errs, x_upper_errs], yerr=[y_lower_errs, y_upper_errs],
            capsize=3, color=(color,alpha), marker=marker, markersize=markersize, ecolor=(color,0.2)
        )

        annotation_indices = np.round(np.linspace(0, len(times[sigma]["DP"]) - 1, 4)).astype(int)
        if alg != "DP":
            annotation_indices = np.append(annotation_indices, [len(times[sigma]["M1K"]) - 1])
        if alg == "PREG":
            annotation_indices = np.append(annotation_indices, [len(times[sigma]["PREG"]) - 1])

        for x, y, n in np.array((*zip(x_means, y_means, times[sigma][alg].keys()),))[annotation_indices]:
            if alg in ("M2K", "M4K"):
                continue
            offset = (4, 4)
            halign = "left"
            valign = "baseline"

            flip_m1k = alg == "M1K" and n in np.array((*times[sigma][alg].keys(),))[annotation_indices[1:-1]]
            flip_dp = alg == "DP" and n in np.array((*times[sigma][alg].keys(),))[annotation_indices[-2:]]
            if flip_m1k or flip_dp:
                offset = (-4, -4)
                halign = "right"
                valign = "top"

            ax.annotate(
                f"n={int(n)}", (x, y), xytext=offset, textcoords="offset points",
                color=color, alpha=0.75, size="x-small", ha=halign, va=valign,
            )

    arrowcolor = "0.93"
    arrow_props = dict(color=arrowcolor, zorder=-100, transform=ax.transAxes)

    ax.plot((0.15, 0.85), (0.15, 0.85), linewidth=20, **arrow_props)
    ax.add_patch(plt.Polygon(((0.1, 0.1), (0.1, 0.25), (0.25, 0.1)), linewidth=5, joinstyle="round", **arrow_props))
    ax.add_patch(plt.Polygon(((0.9, 0.9), (0.9, 0.75), (0.75, 0.9)), linewidth=5, joinstyle="round", **arrow_props))

    ax.text(
        0.125, 0.125, "BETTER", rotation=45,
        fontsize="small", color="gray", zorder=-95, transform=ax.transAxes
    )
    ax.text(
        0.875, 0.875, "WORSE", ha="right", va="top", rotation=45,
        fontsize="small", color="gray", zorder=-95, transform=ax.transAxes
    )

    reorder_legend_fix = [0, 3, 1, 4, 2]
    handles = [
        mlines.Line2D([], [], color=color, label=legend_name, marker=marker, markersize=markersize)
        for legend_name, color, marker in np.array((*zip(legend_names, algcolor, algmarker),))[reorder_legend_fix]
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center")

    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig("fig-eval-tradeoff.pdf")
    plt.show()

# Execution of plot generation
plot()
plot_time_vs_error()
