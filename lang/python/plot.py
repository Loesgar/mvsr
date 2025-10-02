import mvsr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys

########################################
#             GENERATE DATA            #
########################################

# Define start and end indices on x-axis.
bp = [0,851,950,1551,1650,2001]
# Define x values.
x = np.concat((
    np.arange(bp[0]/100.0,bp[1]/100.0, 0.01),
    np.arange(bp[2]/100.0,bp[3]/100.0, 0.01),
    np.arange(bp[4]/100.0,bp[5]/100.0, 0.01),
))
# Calculate true y values for two variants.
y_true = [
    [ 1 * i +  0 for i in x[bp[0]:bp[1]]] +
    [ 0 * i +  2 for i in x[bp[1]:bp[1]+bp[3]-bp[2]]] +
    [-2 * i + 40 for i in x[bp[1]+bp[3]-bp[2]:]],

    [-1 * i +  9 for i in x[bp[0]:bp[1]]] +
    [ 1 * i -  8 for i in x[bp[1]:bp[1]+bp[3]-bp[2]]] +
    [ 0 * i +  6 for i in x[bp[1]+bp[3]-bp[2]:]]
]
# Add gaussian noise to y values.
import random
seed = 1
sigma = 1.0
r = random.Random()
r.seed(seed)
y = [[r.gauss(y,sigma) for y in yv] for yv in y_true]

########################################
#         CALCULATE REGRESSION         #
########################################

# Define custom lerping between segments.
# This is not necessary and here only done for pretty plotting.
# It looks similar to using mvsr.Lerp.Smooth as input to 'lerp'.
def LerpSmooth(x, segs):
    import math
    cur = x - segs[0].x[-1]             # distance between x and left segment
    dist = segs[1].x[0] - segs[0].x[-1] # distance between segments
    t = cur/dist                        # progress (0.0 <= t <= 1.0)
    res = (math.erf((t-.5)*4.0)+1.0)/2  # smoothed value
    return [1-res,res]                  # weighting of the two models

# Calculate segmented regression
regression = mvsr.mvsr(x,y,3, kernel=mvsr.Kernel.Poly(1, lerp=LerpSmooth))

########################################
#            PLOT REGRESSION           #
########################################

# Setup figure with an own axes for each variant.
fig, axs = plt.subplots(2, sharex=True)
axs[0].set_xticks([],minor=False)
axs[1].set_xticks([i for i in range(0,21,2)])
axs[0].set_yticks([],minor=False)
axs[1].set_yticks([],minor=False)
axs[0].set_ylim((-2.75, 11.00))
axs[1].set_ylim((-2.75, 11.00))

# Plot samples.
axs[0].scatter(x,y[0], c='black', s=.1, alpha=.3)
axs[1].scatter(x,y[1], c='black', s=.1, alpha=.3)

# Plot result, remember lines of variants to use the same color later.
# Style and interpolation style parameters can be given as one dict or 
# one dict per variant. Undefined of plotting styles are defined by matplotlib
# configuration for lines (see rcParams). Below are the standard parameters.
(
    ((line_v1,*_),*_),
    ((line_v2,*_),*_),
*_) = regression.plot(axs) # style={}, istyle={'linestyle':'dotted', 'alpha':0.5}

# Plot 'likely' area between segments
def fill_area_between(ax, s1, s2, **kwargs):
    x_start = s1.range[1]  # get end of previous segment
    x_end = s2.range[0]    # get start of next segment

    # Get y = f(x) according to both segments. Sort them to get correct drawing
    # order for matplotlib.
    y_start = sorted([s1(x_start), s2(x_start)])
    y_end = sorted([s1(x_end), s2(x_end)])

    # Draw the Polygon, pass kwargs as styling options.
    ax.add_collection(
        mpl.collections.PolyCollection(
            (np.array([
                [x_start, y_start[0]],
                [x_start, y_start[1]],
                [x_end,   y_end[1]],
                [x_end,   y_end[0]]
            ]),),
            **kwargs
        )
    )

# Iterating over the variants
for vi,(ax,v,l) in enumerate(zip(axs, regression.variants, [line_v1, line_v2])):
    vi += 1
    # Iterate over noighbouring segments to plot area in same color, semitransparently
    for s1,s2 in zip(v, v[1:]):
        fill_area_between(ax, s1, s2, color=l.get_color(), alpha=.1)

    for i,s in enumerate(v):
        i += 1
        # Plot ranges
        ax.plot(s.range, [-1.75]*len(s.range), c='black', marker='|')
        ax.text(sum(s.range)/2, -1.5,
            f'${s.range[0]:.1f} \\leq x \\leq {s.range[1]:.1f}$',
            ha='center', size='small'
        )

        # Print function models
        ax.text(sum(s.range)/2, 9.5,
            f'$f_{i}(x)={s.model[1]:.2f}\\cdot x {s.model[0]:+.2f}$',
            ha='center', size='small'
        )
        # Print MSEs
        ax.text(sum(s.range)/2, -2.25,
            f'$\\mathit{{MSE}}_{i}={s.mse:.2f}$',
            ha='center', size='xx-small'
        )

    # Overall MSE
    ax.set_ylabel(f'Variant {vi} — ($\\mathit{{MSE}}={sum([s.rss for s in v])/sum([len(s.x) for s in v]):.2f}$)')
    #ax.text((v[0].range[0]+v[-1].range[-1])/2, 12,
    #    f'$\\mathit{{MSE}}={sum([s.rss for s in v])/sum([len(s.x) for s in v]):.2f}$',
    #    ha='center'
    #)

# export figure
fig.set_size_inches((7,7))
fig.tight_layout()
fig.savefig("example.pdf")

