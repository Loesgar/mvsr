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
                times.setdefault(int(d), []).append(float(time))
    return dict(sorted(times.items()))

# Read errors from CSV files
def getErrs(alg, sigma, dir=sys.argv[1]):
    times = {}
    for path in Path(dir).glob(f"{alg}_CONT_S{sigma}*"):
        with open(path, newline='') as file:
            for (n, k, d, alg, gen, sigma, seed, time, err, noisy_err, starts, models) in csv.reader(file):
                times.setdefault(int(d), []).append(float(err))
    return dict(sorted(times.items()))

sigmas = ["01", "1"]
algs = ["DP", "DPSM", "PREG", "PREGSM", "M1K", "M2K", "M4K"]
legend_names = ["Exact (DP)", "Exact (DP,R1U)", "Our Approach", "Our Approach (R1U)", "Acharya", "Acharya 2k", "Acharya 4k"]
#algcolor = ["dodgerblue", "gold", "chocolate", "lightpink", "peachpuff"]
#algcolor = ["mediumblue", "darkgreen", "darkred", "chocolate", "goldenrod"]
algcolor = ["black", "grey", "red", "orange", "blue", "dodgerblue", "cyan"]
algalpha = [None, 1.0, None, 1.0, None, 0.5, 0.5]
algmarker = ["o", ".", "^", "v", "s", "D", "d"]
algls = ["solid","dotted","solid","dotted","solid","solid","solid"]
markersize = 5
plot_order = [2, 3, 5, 6, 4, 0, 1]

# Definition for calculating our error bars
def get_percentiles(data, q=0.5):
    mean = np.mean(data)
    upper = np.quantile([d for d in data if d >= mean], q)
    lower = np.quantile([d for d in data if d <= mean], 1.0-q)
    return (mean, upper, lower)

# Generate eval plot (Fig. 3)
def plot(axs, times, errors):
    for ax, text in zip(axs[0], ["Runtime (s)", "MSE", "Relative Speedup", "Relative MSE"]):
        ax.annotate(
            text, xy=(0.5, 1), xytext=(0, 5),
            xycoords='axes fraction', textcoords='offset points',
            size='large', ha='center', va='baseline'
        )

    for ax in axs[len(sigmas) - 1]:
        ax.set_xlabel("dimension (d)")

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
        ax.set_xticks([i for i in times[s]["PREG"].keys()], [str(i) if i != 128 else "" for i in times[s]["PREG"].keys()])
        ax.set_xticks([], minor=True)
        for alg, color, marker, alpha, ls in np.array((*zip(algs, algcolor, algmarker, algalpha, algls),), dtype=object)[plot_order]:
            means, upper_errs, lower_errs = np.array([
                get_percentiles(data) for data in times[s][alg].values()
            ]).T
            upper_errs = upper_errs - means
            lower_errs = means - lower_errs
            ax.errorbar(
                [i for i in times[s][alg].keys()][:len(means)], means, yerr=[lower_errs, upper_errs],
                capsize=3, color=color, marker=marker, markersize=markersize, alpha=alpha, ls=ls,
            )
        ax.relim()

    # error
    for s, axss in zip(sigmas, axs):
        ax = axss[1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([i for i in times[s]["PREG"].keys()], [str(i) if i != 128 else "" for i in times[s]["PREG"].keys()])
        ax.set_xticks([], minor=True)
        for alg, color, marker, alpha, ls in np.array((*zip(algs, algcolor, algmarker, algalpha, algls),), dtype=object)[plot_order]:
            means, upper_errs, lower_errs = np.array([
                get_percentiles(data) for n,data in errors[s][alg].items() if n < 300000
            ]).T
            upper_errs = upper_errs - means
            lower_errs = means - lower_errs
            ax.errorbar(
                [i for i in errors[s][alg].keys()][:len(means)], means, yerr=[lower_errs, upper_errs],
                capsize=3, color=color, marker=marker, markersize=markersize, alpha=alpha, ls=ls,
            )
        ax.relim()

    # relative speed
    for s, axss in zip(sigmas, axs):
        ax = axss[2]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([i for i in times[s]["DP"].keys()], [str(i) for i in times[s]["DP"].keys()])
        ax.set_xticks([], minor=True)
        #dp_means = [np.mean(data) for data in times[s]["DP"].values()]
        for alg, color, marker, alpha, ls in np.array((*zip(algs, algcolor, algmarker, algalpha, algls),), dtype=object)[plot_order]:
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
                capsize=3, color=color, marker=marker, markersize=markersize, alpha=alpha, ls=ls,
            )
        ax.relim()

    # relative error
    for s, axss in zip(sigmas, axs):
        ax = axss[3]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(.5, 100)
        ax.set_xticks([i for i in times[s]["DP"].keys()], [str(i) for i in times[s]["DP"].keys()])
        ax.set_xticks([], minor=True)
        dp_means = [np.mean(data) for data in errors[s]["DP"].values()]
        for alg, color, marker, alpha, ls in np.array((*zip(algs, algcolor, algmarker, algalpha, algls),), dtype=object)[plot_order]:
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
                capsize=3, color=color, marker=marker, markersize=markersize, alpha=alpha, ls=ls,
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
        print(f"Average Error of {alg} relativ to preg at 0.01 sigma: ", err_rel_vals / err_rel_num)

    return [
        mlines.Line2D([], [], color=color, label=legend_name, marker=marker, markersize=markersize, ls=ls)
        for legend_name, color, marker, ls in zip(legend_names, algcolor, algmarker, algls)
    ]

def plotd(folder_path, title, output):
    fig, axs = plt.subplots(len(sigmas), 4, figsize=np.array((8.5, 4.4)))
    ax = axs[0][0]
    ax.annotate(
                title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 20, -70),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='xx-large', ha='right', va='center', rotation=90
            )
    times = dict(zip(sigmas, [dict(zip(algs, [getTimes(alg, sigma, folder_path) for alg in algs])) for sigma in sigmas]))
    errors = dict(zip(sigmas, [dict(zip(algs, [getErrs(alg, sigma, folder_path) for alg in algs])) for sigma in sigmas]))
    handles = plot(axs, times, errors)
    fig.legend(handles=handles, ncol=4, loc='lower center')
    plt.tight_layout(h_pad=-2, w_pad=0.5, rect=(0, 0.1, 1, 1))
    plt.savefig(output)#, bbox_inches="tight", pad_inches=1)
    #plt.show()

# Execution of plot generation
plotd(sys.argv[1], "$n=4096$", "fig-evald-n4096.pdf")
plotd(sys.argv[2], "$n=8192$", "fig-evald-n8192.pdf")
plotd(sys.argv[3], "$n=64 \\cdot d$", "fig-evald-n64d.pdf")
