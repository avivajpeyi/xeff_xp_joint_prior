"""a1, a2, q, theta1, theta2 conditional on xeff"""
import numpy as np
import pandas as pd
from scipy import stats

MCMC_SAMPLES = 10000


def _xeff_min_max(a1, a2, q, c2):
    xeff_lim = a1 / (1 + q) + (a2 * q * c2) / (1 + q)
    return -xeff_lim, xeff_lim


def _p_xeff_given_a1a2qc2(xeff, a1, a2, q, c2):
    """p(xeff|a1,a2,q,c2)"""
    xeff_min, xeff_max = _xeff_min_max(a1, a2, q, c2)
    p_xeff_given_a1a2qc2 = stats.uniform.pdf(
        xeff, loc=xeff_min, scale=xeff_max - xeff_min
    )
    return p_xeff_given_a1a2qc2


def _p_param_and_xeff(param, xeff, init_a1a2qcos2_prior, param_key):
    """p(param and xeff)"""
    s = pd.DataFrame(init_a1a2qcos2_prior.sample(MCMC_SAMPLES))
    s[param_key] = param
    p_xeff_given_other = _p_xeff_given_a1a2qc2(xeff, s.a1, s.a2, s.q, s.cos2)
    p_param = init_a1a2qcos2_prior[param_key].prob(param)
    p_xeff_given_other = np.nan_to_num(p_xeff_given_other)
    return (1.0 / MCMC_SAMPLES) * np.sum(p_xeff_given_other * p_param)


def _p_param_given_xeff(param, xeff, init_a1a2qcos2_prior, param_key):
    """p(param|xeff)"""
    return (1.0 / (p_xeff(xeff)) * _p_param_and_xeff(param, xeff, init_a1a2qcos2_prior, param_key))


def a1_prior_given_xeff(a1, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(a1, xeff, init_a1a2qcos2_prior, "a1")

def a1_and_xeff_prior(a1, xeff, init_a1a2qcos2_prior):
    return _p_param_and_xeff(a1, xeff, init_a1a2qcos2_prior, "a1")


def a2_prior_given_xeff(a2, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(a2, xeff, init_a1a2qcos2_prior, "a2")


def q_prior_given_xeff(q, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(q, xeff, init_a1a2qcos2_prior, "q")


def cos2_prior_given_xeff(cos2, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(cos2, xeff, init_a1a2qcos2_prior, "cos2")

def p_xeff(xeff, init_a1a2qcos2_prior):
    a1_vals = np.linspace(0,1, MCMC_SAMPLES)
    p_a1_and_xeff = a1_and_xeff_prior(a1_vals, xeff, init_a1a2qcos2_prior)
    return np.sum(p_a1_and_xeff) / MCMC_SAMPLES