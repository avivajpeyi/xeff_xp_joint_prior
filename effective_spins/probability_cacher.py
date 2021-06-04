import matplotlib.pyplot as plt
import pandas as pd

plt.style.use(
    'https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle')

DATA_KEY = "probabilities"


def store_probabilities(df: pd.DataFrame, fname: str):
    assert ".h5" in fname, f"{fname} is invalid"
    store = pd.HDFStore(fname)
    store.append(key=DATA_KEY, value=df, format="t", data_columns=True)
    store.close()


def load_probabilities(fname) -> pd.DataFrame:
    df = pd.read_hdf(fname, key=DATA_KEY)
    return df


def plot_probs(x, y, p, xlabel, ylabel, plabel, fname):
    plt.tricontour(x, y, p, 15, linewidths=0.5, colors='k')
    cmap = plt.tricontourf(
        x, y, p, 15,
        norm=plt.Normalize(vmax=abs(p).max(), vmin=-abs(p).max()),
        cmap='inferno'
    )
    plt.colorbar(cmap, label=plabel)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(fname)
