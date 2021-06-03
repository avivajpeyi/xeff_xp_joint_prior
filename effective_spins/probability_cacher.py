import pandas as pd

DATA_KEY = "probabilities"


def store_probabilities(df: pd.DataFrame, fname: str):
    assert ".h5" in fname, f"{fname} is invalid"
    store = pd.HDFStore(fname)
    store.append(key=DATA_KEY, value=df, format="t", data_columns=True)
    store.close()


def load_probabilities(fname) -> pd.DataFrame:
    df = pd.read_hdf(fname, key=DATA_KEY)
    return df
