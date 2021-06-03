"""a1, a2, q, theta1, theta2 conditional on xeff"""
import numpy as np
import pandas as pd
from bilby.core.prior import PriorDict
from scipy import stats

MCMC_SAMPLES = 1000
INTEGRATION_POINTS = 100

def xeff_lim(a1: np.ndarray, a2: np.ndarray, q: np.ndarray, cos2: np.ndarray):
    """Max lim xeff given a1, a2, q, c2 (and assuming c1 in [-1,1])."""
    return a1 / (1 + q) + (a2 * q * cos2) / (1 + q)


def p_xeff_given_a1a2qc2(
        param: float, xeff: float,
        init_a1a2qcos2_prior: PriorDict, param_key: str
) -> float:
    """p(xeff|a1,a2,q,c2), O(n)"""
    s = pd.DataFrame(init_a1a2qcos2_prior.sample(MCMC_SAMPLES))
    s[param_key] = param
    _xeff_lim = xeff_lim(s.a1, s.a2, s.q, s.cos2)
    return stats.uniform.pdf(xeff, loc=-_xeff_lim, scale=2 * _xeff_lim)


@np.vectorize
def p_param_and_xeff(
        param: float, xeff: float,
        init_a1a2qcos2_prior: PriorDict, param_key: str
) -> float:
    """p(param and xeff), O(n^2)"""
    p_xeff_given_other = p_xeff_given_a1a2qc2(
        param, xeff, init_a1a2qcos2_prior, param_key
    )
    p_param = init_a1a2qcos2_prior[param_key].prob(param)
    p_xeff_given_other = np.nan_to_num(p_xeff_given_other)
    # dont need p_other, only p_param as MCMC
    return (1.0 / MCMC_SAMPLES) * np.sum(p_xeff_given_other * p_param)


def p_param_given_xeff(
        param: float, xeff: float,
        init_a1a2qcos2_prior: PriorDict, param_key: str
) -> float:
    """p(param|xeff), O(n^3)"""
    _p_param_and_xeff = p_param_and_xeff(param, xeff, init_a1a2qcos2_prior, param_key)
    return _p_param_and_xeff / p_xeff(xeff, init_a1a2qcos2_prior)


def p_xeff(xeff, init_a1a2qcos2_prior):
    """
    p(xeff) = int_{ai \in a} p(a and xeff) da, O(n^3)
    """
    a1_vals = np.linspace(0, 1, INTEGRATION_POINTS)
    a1_key = 'a1'
    da = a1_vals[1] - a1_vals[0]
    p_a1_and_xeff = p_param_and_xeff(a1_vals, xeff, init_a1a2qcos2_prior, a1_key)
    return np.sum(p_a1_and_xeff  * da)
