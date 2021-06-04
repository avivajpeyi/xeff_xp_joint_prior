import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

plt.style.use(
    'https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle')

from effective_spins import probability_cacher
from effective_spins.priors_conditional_on_xeff import p_xeff

OUTDIR = "p_xeff"


def compute_and_store_p_xeff(p_a1_and_xeff_fname):
    df = probability_cacher.load_probabilities(p_a1_and_xeff_fname)
    data = {"p_xeff": [], "xeff": []}
    for xeff in tqdm(df.xeff.unique(), desc="calculating p(xeff)"):
        d = df[df['xeff'] == xeff]
        data['xeff'].append(xeff)
        data['p_xeff'].append(p_xeff(xeff, a1s=d.a1, p_a1_and_xeff=d.p_a1_and_xeff))
    df = pd.DataFrame(data)
    fname = os.path.join(OUTDIR, f"p_xeff.h5")
    probability_cacher.store_probabilities(df, fname)
    plot_p_xeff(df.xeff, df.p_xeff, fname.replace('.h5', '.png'))
    print(f"Saved {fname}")


def plot_p_xeff(xeff, p_xeff, fname):
    plt.close('all')
    plt.plot(xeff, p_xeff, c='k')
    plt.ylabel('p(xeff)')
    plt.xlabel('xeff')
    plt.tight_layout()
    plt.savefig(fname)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    compute_and_store_p_xeff("p_param_and_xeff/p_a1_and_xeff.h5")


if __name__ == '__main__':
    main()
