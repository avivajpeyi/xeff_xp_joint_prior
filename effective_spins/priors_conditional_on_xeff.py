"""a1, a2, q, theta1, theta2 conditional on xeff"""
from typing import List, Optional

import pandas as pd
from bilby.core.prior import PriorDict

from .cupy_utils import uniform, xp

MC_SAMPLES = 100000
INTEGRATION_POINTS = 10000


def xeff_lim(a1: xp.ndarray, a2: xp.ndarray, q: xp.ndarray, cos2: xp.ndarray):
    """Min and max xeff lim given a1, a2, q, c2 (and assuming c1 in [-1,1])."""
    return (-a1 + (a2 * q * cos2)) / (1 + q), (a1 + (a2 * q * cos2)) / (1 + q)


def p_xeff_given_a1a2qc2(
        param: float, xeff: float,
        init_a1a2qcos2_prior: PriorDict, param_key: str
) -> float:
    """p(xeff|a1,a2,q,c2), O(n)"""
    s = pd.DataFrame(init_a1a2qcos2_prior.sample(MC_SAMPLES))
    s[param_key] = param
    xeff_min, xeff_max = xeff_lim(s.a1, s.a2, s.q, s.cos2)
    return uniform.pdf(xeff, loc=xeff_min, scale=xeff_max - xeff_min)


def p_param_and_xeff(
        param: float, xeff: float,
        init_a1a2qcos2_prior: PriorDict, param_key: str
) -> float:
    """p(param_key and xeff), O(n^2)"""
    p_xeff_given_other = xp.asanyarray(
        p_xeff_given_a1a2qc2(
            param, xeff, init_a1a2qcos2_prior, param_key
        ))
    p_param = init_a1a2qcos2_prior[param_key].prob(param)
    p_xeff_given_other = xp.nan_to_num(p_xeff_given_other)
    # dont need p_other, only p_param as MC
    return (1.0 / MC_SAMPLES) * xp.sum(p_xeff_given_other * p_param)


def p_param_given_xeff(
        param: Optional[float] = 0, xeff: Optional[float] = 0,
        init_a1a2qcos2_prior: Optional[PriorDict] = None,
        param_key: Optional[str] = "",
        _p_param_and_xeff: Optional[List] = [],
        _p_xeff: Optional[List] = []
) -> float:
    """p(param_key|xeff), O(n^3)"""
    if len(_p_xeff) == 0:
        _p_param_and_xeff = p_param_and_xeff(
            param, xeff, init_a1a2qcos2_prior, param_key)
        _p_xeff = p_xeff(xeff, init_a1a2qcos2_prior)
    return _p_param_and_xeff / _p_xeff


def p_xeff(xeff: float, init_a1a2qcos2_prior: Optional[PriorDict] = None,
           a1s=[], p_a1_and_xeff=[]) -> float:
    """
    p(xeff) = int_{ai \in a} p(a and xeff) da, O(n^3)
    """
    if len(p_a1_and_xeff) == 0 and len(a1s) == 0:
        a1s = xp.linspace(0, 1, INTEGRATION_POINTS)
        p_a1_and_xeff = [p_param_and_xeff(a1, xeff, init_a1a2qcos2_prior, 'a1') for a1
                         in a1s]
    return xp.trapz(y=p_a1_and_xeff, x=a1s, axis=0)
