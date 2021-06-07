import argparse
import os

import numpy as np
import pandas as pd
from agn_utils.batch_processing import create_python_script_jobs
from bilby.core.prior import PriorDict, Uniform
from tqdm.auto import tqdm

from effective_spins import priors_conditional_on_xeff
from effective_spins.priors_conditional_on_xeff import p_param_and_xeff
from effective_spins.probability_cacher import plot_probs, store_probabilities

N = 1000

OUTDIR = "p_param_and_xeff"
VALID_PARAMS = ['a1', 'a2', 'q', 'cos2']


def get_prior():
    priors = PriorDict()
    priors["q"] = Uniform(minimum=0, maximum=1)
    priors["a1"] = Uniform(minimum=0, maximum=1)
    priors["a2"] = Uniform(minimum=0, maximum=1)
    priors["cos2"] = Uniform(minimum=-1, maximum=1)
    priors["cos1"] = Uniform(minimum=-1, maximum=1)
    return priors


def generate_dataset(param_key):
    prob_key = f"p_{param_key}_and_xeff"
    pri = get_prior()
    xeffs = np.linspace(-1, 1, N)
    params = np.linspace(pri[param_key].minimum, pri[param_key].maximum, N)

    data = {param_key: [], "xeff": [], prob_key: []}
    for xeff in tqdm(xeffs, desc=f"p({param_key} and xeff)"):
        for param in params:
            _p_param_and_xeff = p_param_and_xeff(param, xeff, pri, param_key)
            data[param_key].append(param)
            data["xeff"].append(xeff)
            data[prob_key].append(_p_param_and_xeff)
    df = pd.DataFrame(data)
    fname = os.path.join(OUTDIR, f"{prob_key}.h5")
    store_probabilities(df, fname)
    plot_probs(
        x=df[param_key], y=df["xeff"], p=df[prob_key],
        xlabel=param_key, ylabel="xeff", plabel=prob_key.replace("_", " "),
        fname=fname.replace('.h5', '.png')
    )
    print(f"Saved {fname}")


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-dag", help="Make dag", action="store_true")
    parser.add_argument("--testing", help="test data gen", action="store_true")
    parser.add_argument("--param_key", help=f"One of {VALID_PARAMS}", type=str)
    args = parser.parse_args()
    return args


def make_dag():
    create_python_script_jobs(
        main_job_name="p_param_and_xeff_calculator",
        python_script=os.path.abspath(__file__),
        job_args_list=[{"param_key": n, } for n in VALID_PARAMS],
        job_names_list=[f"p_{param}_and_xeff" for param in VALID_PARAMS],
    )


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    args = create_parser_and_read_args()
    if args.make_dag:
        make_dag()
    else:
        if args.testing:
            print("TESTING: reducing num samples")
            global N
            N = 20
            priors_conditional_on_xeff.MCMC_SAMPLES = 100

        generate_dataset(param_key=args.param)


if __name__ == '__main__':
    main()
