import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from effective_spins import priors_conditional_on_xeff
from . import utils


class TestPriorConditionalOnXeff(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/conditional_testoutdir"
        os.makedirs(self.outdir, exist_ok=True)
        self.p = utils.get_traditional_prior()
        self.xeffs = [-0.75,  -0.25,  0.1,  0.5]
        self.N = 50

    # def tearDown(self):
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def plot_conditional_prior(self, param, xeffs, param_key):
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        for xeff in tqdm(xeffs, desc=f"p({param_key} and xeff) "):
            priors_conditional_on_xeff.MCMC_SAMPLES = 1000
            p_param = [priors_conditional_on_xeff.p_param_and_xeff(
                param=p,
                xeff=xeff,
                init_a1a2qcos2_prior=self.p,
                param_key=param_key,
            ) for p in param]
            axs[0].plot(param, p_param, "-", label=f"xeff={xeff}")
        for xeff in tqdm(xeffs, desc=f"p({param_key}|xeff) "):
            priors_conditional_on_xeff.MCMC_SAMPLES = 100
            p_param = [priors_conditional_on_xeff.p_param_given_xeff(
                param=p,
                xeff=xeff,
                init_a1a2qcos2_prior=self.p,
                param_key=param_key,
            ) for p in param]
            axs[1].plot(param, p_param, "-")
        axs[0].set_ylabel(f"p({param_key} and xeff)")
        axs[1].set_ylabel(f"p({param_key}|xeff)")
        axs[0].set_xlabel(param_key)
        axs[0].set_xlim(min(param), max(param))
        axs[0].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(f"{self.outdir}/p_{param_key}_and_xeff.png")

    def test_a1_conditional_priors(self):
        a1 = np.linspace(0, 1, self.N)
        self.plot_conditional_prior(param=a1, xeffs=self.xeffs, param_key="a1")

    def test_q_conditional_priors(self):
        q = np.linspace(0, 1, self.N)
        self.plot_conditional_prior(param=q, xeffs=self.xeffs, param_key="q")

    def test_a2_conditional_priors(self):
        a2 = np.linspace(0, 1, self.N)
        self.plot_conditional_prior(param=a2, xeffs=self.xeffs, param_key="a2")

    def test_cos2_conditional_priors(self):
        cos2 = np.linspace(-1, 1, self.N)
        self.plot_conditional_prior(param=cos2, xeffs=self.xeffs, param_key="cos2")

    def test_simple_val(self):
        a1 = 0.5
        xeff = 0.1
        p_a1 = priors_conditional_on_xeff.p_param_given_xeff(param=a1, xeff=xeff, init_a1a2qcos2_prior=self.p, param_key="a1"
        )
        self.assertNotEqual(p_a1, 1)


if __name__ == "__main__":
    unittest.main()
