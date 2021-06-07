import os

import matplotlib.pyplot as plt
import numpy as np

from effective_spins import probability_cacher
from effective_spins.priors_conditional_on_xeff import p_param_given_xeff

plt.style.use(
    'https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle')

OUTDIR = "p_param_given_xeff"
PRIOR_P_AND_XEFF = "p_param_and_xeff/p_{param}_and_xeff.h5"


def compute_and_store_p_param_given_xeff(param_key, prior_xeff_fname):
    df = probability_cacher.load_probabilities(PRIOR_P_AND_XEFF.format(param=param_key))
    xeff_df = probability_cacher.load_probabilities(prior_xeff_fname)
    df = df.merge(xeff_df, on='xeff')
    p_param_and_xeff_key = f"p_{param_key}_and_xeff"
    p_param_given_xeff_key = f"p_{param_key}_given_xeff"
    p = p_param_given_xeff(_p_xeff=df.p_xeff, _p_param_and_xeff=df[p_param_and_xeff_key])
    p = np.nan_to_num(p)
    df[p_param_given_xeff_key] = p
    fname = os.path.join(OUTDIR, f"{p_param_given_xeff_key}.h5")
    probability_cacher.store_probabilities(df, fname)
    probability_cacher.plot_probs(
        x=df[param_key], y=df['xeff'], p=p,
        xlabel=param_key, ylabel="xeff", plabel=p_param_given_xeff_key.replace('_', ' '),
        fname=fname.replace('.h5', '.png')
    )
    print(f"Saved {fname} with {len(df)} datapoints")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for k in ['a1', 'a2', 'q', 'cos2']:
        compute_and_store_p_param_given_xeff(
            param_key=k,
            prior_xeff_fname='p_xeff/p_xeff.h5'
        )


if __name__ == '__main__':
    main()
