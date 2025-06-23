from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
from matplotlib.patches import ConnectionPatch
import numpy as np
import sys
import matplotlib.style as mplstyle

mplstyle.use("fast")

k = 4
data = sys.argv[1] if len(sys.argv) >= 2 else "../data/task_948_bwa__DA45_CPU.txt"
eval = sys.argv[2] if len(sys.argv) >= 3 else "."

# load raw data to scatter points
data = np.loadtxt(data).T  # dtype=np.double
X = data[0]
Y = data[1]


# load benchmark results from folder
def getBps(alg, dir=eval):
    if alg == "DP":
        alg = "PREG"
    curk = -1
    bps = None
    for path in Path(dir).glob(f"{alg}_CSV_NONE_*"):
        with open(path, newline="") as file:
            *_, (n, k, d, alg, gen, sigma, seed, time, err, noisy_err, starts, models) = (
                csv.reader(file)
            )
            if int(k) >= curk:
                bps = np.fromstring(starts[1:-1], dtype=int, sep=" ")
    return bps


def plotseg(ax, x, y, starts, color, name, alpha, marker):
    for s, e in zip(starts, (list(starts) + [len(X)])[1:]):
        slope, offset = np.polyfit(x[s:e], y[s:e], 1)
        ax.plot(
            [x[s], x[e - 1]],
            [slope * x[s] + offset, slope * x[e - 1] + offset],
            color=color,
            label=name,
            alpha=alpha,
            marker=marker,
        )


algs = ["DP", "PREG", "M1K", "M2K", "M4K"]
algcolor = ["black", "red", "blue", "dodgerblue", "cyan"]
algmarker = ["o", "^", "s", "D", "d"]
algalpha = [None, None, None, 0.5, 0.5]
plot_order = [3, 4, 2, 0, 1]

fig, (axs_all, axs_zoom) = plt.subplots(1, 2, width_ratios=[2, 3])
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.subplots_adjust(wspace=0.025, bottom=0.13)
fig.set_size_inches(np.array((8.5, 3.4)))# * 1.25)  # 4.4 # 2.5

axs_all.scatter(X, Y, color="black", s=0.5, alpha=0.2, edgecolor="None").set_rasterized(
    True
)
axs_zoom.scatter(X, Y, color="black", s=0.5, alpha=1, edgecolor="None").set_rasterized(
    True
)

axs_zoom.get_yaxis().set_visible(False)
axs_all.set_xticks(np.arange(0, 160001, 40000))
for i, alg in np.array((*enumerate(algs),), dtype=object)[plot_order]:
    starts = getBps(alg)
    plotseg(axs_all, X, Y, starts, algcolor[i], alg, algalpha[i], algmarker[i])
    plotseg(axs_zoom, X, Y, starts, algcolor[i], alg, algalpha[i], algmarker[i])

axs_all.add_patch(
    plt.Rectangle(
        (147000, 50), 156200 - 147000, 800, ls="--", ec="black", alpha=0.4, fc="none"
    )
)

axs_all.set_ylabel("CPU (\\%)")
axs_zoom.set_xlabel("Time (s)", loc="left")
axs_zoom.set_xlim((147000, 156200))
axs_zoom.set_ylim(axs_all.get_ylim())
xrange = axs_zoom.get_xlim()
yrange = axs_zoom.get_ylim()
fig.add_artist(
    ConnectionPatch(
        xyA=(156200, 850),
        coordsA=axs_all.transData,
        xyB=(xrange[0], yrange[1]),
        coordsB=axs_zoom.transData,
        ls="--",
    )
)
fig.add_artist(
    ConnectionPatch(
        xyA=(156200, 50),
        coordsA=axs_all.transData,
        xyB=(xrange[0], yrange[0]),
        coordsB=axs_zoom.transData,
        ls="--",
    )
)

legend_names = ["Exact (DP)", "Our Approach", "Acharya", "Acharya 2k", "Acharya 4k"]
handles = [
    mlines.Line2D(
        [],
        [],
        color="black",
        alpha=0.5,
        ls="None",
        markeredgecolor="None",
        label="Samples",
        marker=mmarkers.MarkerStyle("."),
    )
] + [
    mlines.Line2D(
        [], [], color=color, label=legend_name, marker=marker
    )  # , markersize=markersize)
    for legend_name, color, marker in np.array(
        (*zip(legend_names, algcolor, algmarker),)
    )
]
axs_zoom.legend(handles=handles, ncol=3, loc="upper right", columnspacing=0.8)

plt.savefig("fig-eval-real.pdf", dpi=600)
plt.show()
