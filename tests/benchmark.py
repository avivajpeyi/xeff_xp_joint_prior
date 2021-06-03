"""Script to benchmark CuPy vs NumPy"""
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from effective_spins import priors_conditional_on_xeff
from utils import get_traditional_prior
from effective_spins.cupy_utils import CUPY_LOADED

p = get_traditional_prior()


def computation(num_samples):
    """Plots a scatter plot."""
    priors_conditional_on_xeff.MCMC_SAMPLES = int(num_samples)
    priors_conditional_on_xeff.p_param_and_xeff(
        param=0.2,
        xeff=0.2,
        init_a1a2qcos2_prior=p,
        param_key='a1',
    )


def benchmark():
    num_samp = [1e3, 1e4, 1e5, 1e6, 1e7]
    times = np.zeros(len(num_samp))
    for i in tqdm(range(len(num_samp))):
        start = time.time()
        computation(num_samp[i])
        times[i] = time.time() - start
    return pd.DataFrame(dict(n=num_samp, t=times))


def main():
    if CUPY_LOADED:
        print("Using CuPy")
    else:
        print("Using NumPy")
    print(benchmark())


if __name__ == "__main__":
    main()
