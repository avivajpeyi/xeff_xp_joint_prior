from scipy.interpolate import interp1d

from .cupy_utils import xp
from .distribution_rules import prod_dist_interp, inv_dist_interp, translate_dist_interp, sum_dist_interp
from .probability_cacher import load_probabilities

N = 1000

CACHED_DATA_FOLDER = "studies/data/p_param_given_xeff"
PARAMS = ['a1', 'a2', 'q', 'cos2']
CACHED_DATA = {}


def get_p_param_given_xeff(xeff=0):
    if len(CACHED_DATA) == 0:
        for k in PARAMS:
            param_df = load_probabilities(fname=f"{CACHED_DATA_FOLDER}/p_{k}_given_xeff.h5")
            CACHED_DATA.update({k: param_df})
            print(f"Loading cached probs for {k} ({len(param_df)} datapoints)")

    p_funcs = dict()
    for k in CACHED_DATA.keys():
        df = CACHED_DATA[k].copy()
        df = df[df['xeff'] == xeff]  # filter data
        p_funcs.update({
            f"p_{k}": interp1d(x=df[k], y=df[f'p_{k}_given_xeff'], bounds_error=False)
        })
    return p_funcs['p_a1'], p_funcs['p_a2'], p_funcs['p_q'], p_funcs['p_cos2']


def get_param_grid():
    a1 = xp.linspace(0, 1, N)
    a2 = xp.linspace(0, 1, N)
    q = xp.linspace(0, 1, N)
    cos2 = xp.linspace(-1, 1, N)
    return a1, a2, q, cos2


def get_p_xp_given_xeff_and_vals(xeff):
    """
    xp = a1 sqrt(1 - (a+b)^2)
       = a1 * d

    where
     a = xeff(q+1)/a1
     b = a2qcos2/a1
     c = a + b
     d = sqrt(1-c^2)
    """
    a1, a2, q, cos2 = get_param_grid()
    p_a1, p_a2, p_q, p_cos2 = get_p_param_given_xeff(xeff)
    d, p_d = get_p_d_and_vals(a2, p_a1, p_a2, p_q, p_cos2)

    p_xp = prod_dist_interp(z_vals=d, a_vals=a1, pdf_a=p_a1, pdf_b=p_d)

    return p_xp


def get_p_sqrt_x2_plus_1_dist(z_vals, pdf_a):
    """
    let g(a) = sqt(1-a^2)
    and h(z) = inv(g(z)) = sqt(1-z^2)
    note dh(z)/dz = z/sqt(1-z^2)

    F_{Z}(z) = F_{A}(h(z)) * |dh(z)/dz|
    """
    h = xp.sqrt(1 - z_vals ** 2)
    dh_dz = z_vals / h
    _pdf_z = pdf_a(h) * xp.abs(dh_dz)
    return interp1d(x=z_vals, y=_pdf_z, bounds_error=False)


def get_p_qplus1_a1_and_vals(p_q, p_a1):
    qplus1 = xp.linspace(1, 2, N)
    p_qplus1 = translate_dist_interp(z_vals=qplus1, pdf_a=p_q, translate=1)

    p_inv_a1 = get_p_inv_a1(p_a1)

    qplus1_a1 = xp.linspace(0, 20, N)
    p_qplus1_a1 = prod_dist_interp(z_vals=qplus1_a1, a_vals=qplus1, pdf_a=p_qplus1, pdf_b=p_inv_a1)
    return qplus1_a1, p_qplus1_a1


def get_p_a2qc2_a1_and_vals(a2, p_a1, p_a2, p_q, p_cos2):
    """p of a2qcos2/a1"""
    a2q = xp.linspace(0, 1, N)
    p_a2q = prod_dist_interp(z_vals=a2q, a_vals=a2, pdf_a=p_a2, pdf_b=p_q)

    a2qc2 = xp.linspace(-1, 1, N)
    p_a2qc2 = prod_dist_interp(z_vals=a2qc2, a_vals=a2q, pdf_a=p_a2q, pdf_b=p_cos2)

    p_inv_a1 = get_p_inv_a1(p_a1)

    a2qc2_a1 = xp.linspace(-10, 10, N)
    p_a2qc2_a1 = prod_dist_interp(z_vals=a2qc2_a1, a_vals=a2qc2, pdf_a=p_a2qc2, pdf_b=p_inv_a1)
    return a2qc2_a1, p_a2qc2_a1


def get_p_c_and_vals(a2, p_a1, p_a2, p_q, p_cos2):
    """p of xeff(q+1)/a1 + a2qcos2/a1"""
    a, p_a = get_p_qplus1_a1_and_vals(p_q, p_a1)
    b, p_b = get_p_a2qc2_a1_and_vals(a2, p_a1, p_a2, p_q, p_cos2)

    c = xp.linspace(0, 1, N)
    p_c = sum_dist_interp(z_vals=c, a_vals=a, pdf_a=p_a, pdf_b=p_b)
    return c, p_c


def get_p_d_and_vals(a2, p_a1, p_a2, p_q, p_cos2):
    """p of sqrt(1-c**2)"""
    c, p_c = get_p_c_and_vals(a2, p_a1, p_a2, p_q, p_cos2)
    d = xp.linspace(0, 1, N)
    p_d = get_p_sqrt_x2_plus_1_dist(z_vals=d, pdf_a=p_c)
    return d, p_d


def get_p_inv_a1(p_a1):
    inv_a1 = xp.linspace(-10, 10, N)
    return inv_dist_interp(z_vals=inv_a1, pdf_a=p_a1)
