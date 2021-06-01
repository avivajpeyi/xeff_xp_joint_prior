"""a1, a2, q, theta1, theta2 conditional on xeff"""
import numpy as np
import pandas as pd
from scipy import stats

from .xeff_prior import get_marginalised_chi_eff as p_xeff

MCMC_SAMPLES = 10000


def _xeff_min_max(a1, a2, q, c2):
    xeff_lim = a1 / (1 + q) + (a2 * q * c2) / (1 + q)
    return -xeff_lim, xeff_lim


def _p_xeff_given_a1a2qc2(xeff, a1, a2, q, c2):
    xeff_min, xeff_max = _xeff_min_max(a1, a2, q, c2)
    p_xeff_given_a1a2qc2 = stats.uniform.pdf(
        xeff, loc=xeff_min, scale=xeff_max - xeff_min
    )
    return p_xeff_given_a1a2qc2


@np.vectorize
def _p_param_given_xeff(param, xeff, init_a1a2qcos2_prior, param_key):
    s = pd.DataFrame(init_a1a2qcos2_prior.sample(MCMC_SAMPLES))
    s[param_key] = param
    p_xeff_given_a1a2qc2 = _p_xeff_given_a1a2qc2(xeff, s.a1, s.a2, s.q, s.cos2)
    p_param = init_a1a2qcos2_prior[param_key].prob(param)
    return 1.0 / p_xeff(xeff) * np.sum(p_param * p_xeff_given_a1a2qc2) / MCMC_SAMPLES


def a1_prior_given_xeff(a1, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(a1, xeff, init_a1a2qcos2_prior, "a1")


def a2_prior_given_xeff(a2, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(a2, xeff, init_a1a2qcos2_prior, "a2")


def q_prior_given_xeff(q, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(q, xeff, init_a1a2qcos2_prior, "q")


def cos2_prior_given_xeff(cos2, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(cos2, xeff, init_a1a2qcos2_prior, "cos2")
