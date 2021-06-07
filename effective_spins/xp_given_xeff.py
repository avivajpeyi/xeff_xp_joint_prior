from scipy.interpolate import interp1d

from .cupy_utils import xp
from .distribution_rules import prod_dist_interp, inv_dist_interp
from .probability_cacher import load_probabilities

N = 1000

CACHED_DATA_FOLDER = "p_param_given_xeff"
PARAMS = ['a1', 'a2', 'q', 'cos2']
CACHED_DATA = {}


def get_p_param_given_xeff(xeff=0):
    if len(CACHED_DATA) == 0:
        for k in PARAMS:
            param_df = load_probabilities(fname=f"{CACHED_DATA_FOLDER}/p_{k}_given_xeff.h5")
            CACHED_DATA.update({k: param_df})
            print(f"Loading cached probs for {k} ({len(param_df)} datapoints)")

    p_funcs = dict()
    for k, df in CACHED_DATA.items():
        df = df['xeff'] == xeff  # filter data
        p_funcs.update({
            f"p_{k}": interp1d(x=df['k'], y=df[f'p_{k}_given_xeff'])
        })
    return p_funcs['p_a1'], p_funcs['p_a2'], p_funcs['p_q'], p_funcs['p_cos2']


def get_param_grid():
    a1 = xp.linspace(0, 1, N)
    a2 = xp.linspace(0, 1, N)
    q = xp.linspace(0, 1, N)
    cos2 = xp.linspace(-1, 1, N)
    return a1, a2, q, cos2


def p_xp_given_xeff(xp, xeff):
    """
    xp = a1 sqrt(1 - (a+b)^2)

    where
     a = xeff(q+1)/a1
     b = a2qcos2/a1
    """
    a1, a2, q, cos2 = get_param_grid()
    p_a1, p_a2, p_q, p_cos2 = get_p_param_given_xeff(xeff)


    a2qc2_a1, p_a2qc2_a1 = get_p_a2qc2_a1_and_vals(a2, p_a1, p_a2, p_q, p_cos2)

    return 1

def get_p_qplus1_a1_and_vals():
    qplus1_a1 = xp.linspace(-10, 10, N)


def get_p_a2qc2_a1_and_vals(a2, p_a1, p_a2, p_q, p_cos2):
    """p of a2qcos2/a1"""
    a2q = xp.linspace(0, 1, N)
    p_a2q = prod_dist_interp(z_vals=a2q, a_vals=a2, pdf_a=p_a2, pdf_b=p_q)

    a2qc2 = xp.linspace(-1, 1, N)
    p_a2qc2 = prod_dist_interp(z_vals=a2qc2, a_vals=a2q, pdf_a=p_a2q, pdf_b=p_cos2)

    inv_a1 = xp.linspace(-10, 10, N)
    p_inv_a1 = inv_dist_interp(z_vals=inv_a1,pdf_a=p_a1)

    a2qc2_a1 = xp.linspace(-10, 10, N)
    p_a2qc2_a1 = prod_dist_interp(z_vals=a2qc2_a1, a_vals=a2qc2, pdf_a=p_a2qc2, pdf_b=p_inv_a1)
    return a2qc2_a1, p_a2qc2_a1


