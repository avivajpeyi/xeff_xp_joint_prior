import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Uniform
from bilby.core.prior import PriorDict
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from effective_spins.conversions import (
    calculate_xeff,
    calculate_xp,
    calculate_xp_given_xeff,
)


def get_traditional_prior():
    priors = PriorDict()
    priors["q"] = Uniform(minimum=0, maximum=1)
    priors["a1"] = Uniform(minimum=0, maximum=1)
    priors["a2"] = Uniform(minimum=0, maximum=1)
    priors["cos2"] = Uniform(minimum=-1, maximum=1)
    priors["cos1"] = Uniform(minimum=-1, maximum=1)
    return priors


def uniform_distribution(min=-1, max=1, N=10000):
    dist = Uniform(minimum=min, maximum=max)
    samples = dist.sample(N)
    pdf = dist.prob(samples)
    return samples, pdf, dist


def get_traditional_samples(num_samples=10 ** 5):
    priors = get_traditional_prior()
    s = pd.DataFrame(priors.sample(num_samples))
    s["sin1"] = np.sqrt(1 - s.cos1 ** 2)
    s["sin2"] = np.sqrt(1 - s.cos2 ** 2)
    s["tan1"] = s["sin1"] / s["cos1"]
    s["tan2"] = s["sin2"] / s["cos2"]
    s["xeff"] = calculate_xeff(s.a1, s.a2, s.cos1, s.cos2, s.q)
    s["xp"] = calculate_xp(s.a1, s.a2, s.q, s.sin1, s.sin2)
    s["xp|xeff"] = calculate_xp_given_xeff(
        s.xeff, s.a1, s.a2, s.q, s.cos1, s.cos2, s.tan1, s.tan2
    )
    return s


def plot_funct_and_samples(func, samples, limits, labels, func_kwargs={}, bins=100):
    xvals = np.linspace(limits[0], limits[1], 300)
    yvals = func(xvals, **func_kwargs)
    print(max(yvals))
    fig = plt.figure(figsize=(15, 4))
    ax1 = fig.add_subplot(111)
    ax1.hist(samples, density=True, bins=bins)
    ax1.plot(xvals, yvals, color="black")
    ax1.xaxis.grid(True, which="major", ls=":", color="grey")
    ax1.yaxis.grid(True, which="major", ls=":", color="grey")
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel(labels[0], fontsize=18)
    ax1.set_ylabel(labels[1], fontsize=18)
    ax1.set_xlim(limits[0], limits[1])
    plt.tight_layout()
    return fig


def gauss_smooth(args, smooth):
    x, y, z = args
    z = gaussian_filter(z, smooth)
    return x, y, z

def plot_heatmap(x, y, z, ax, cmap, add_cbar=False, smooth=None):
    x = np.unique(x)
    y = np.unique(y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = z.reshape(len(x), len(y))
    args = (X, Y, Z)
    if smooth:
        args = gauss_smooth(args, smooth)
    cbar = ax.pcolor(*args, cmap=cmap, vmin=np.nanmin(z), vmax=np.nanmax(z), zorder=-100)
    if add_cbar:
        fig = ax.get_figure()
        cbar = fig.colorbar(cbar, ax=ax)
        return cbar
    return None


def get_kde(x, y):
    return gaussian_kde(np.vstack([x, y]))


def evaluate_kde_on_grid(kde_x, kde_y, xrng, yrng, num_gridpoints=100j):
    kde = get_kde(kde_x, kde_y)
    xmin, xmax = np.min(xrng), np.max(xrng)
    ymin, ymax = np.min(yrng), np.max(yrng)
    x_grid, y_grid = np.mgrid[xmin:xmax:num_gridpoints, ymin:ymax:num_gridpoints]
    xy_array = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(xy_array).T
    return x_grid, y_grid, z.reshape(x_grid.shape)


