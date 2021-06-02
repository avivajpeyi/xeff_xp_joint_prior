import os
import shutil
import unittest

from effective_spins.priors_conditional_on_xeff import p_xeff
from . import utils

import numpy as np

class TestXeffPrior(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/xeff_testoutdir"
        os.makedirs(self.outdir, exist_ok=True)

    # def tearDown(self):
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_prior_on_xeff(self):
        samples = utils.get_traditional_samples()
        fig = utils.plot_funct_and_samples(
            vectorized_pxeff,
            samples.xeff,
            [-1, 1],
            [r"$\chi_{\rm eff}$", r"$p(\chi_{\rm eff})$"],
        )
        fig.savefig(f"{self.outdir}/p_xeff.png")

    def test_simple_p_xeff_calc(self):
        p_xeff_val = p_xeff(xeff=1, init_a1a2qcos2_prior=utils.get_traditional_prior())
        self.assertEqual(p_xeff_val, 0)
        p_xeff_val = p_xeff(xeff=-1, init_a1a2qcos2_prior=utils.get_traditional_prior())
        self.assertEqual(p_xeff_val, 0)
        p_xeff_val = p_xeff(xeff=0, init_a1a2qcos2_prior=utils.get_traditional_prior())
        self.assertGreater(p_xeff_val, 0)

@np.vectorize
def vectorized_pxeff(xeff):
    init_p = utils.get_traditional_prior()
    return p_xeff(xeff, init_p)

if __name__ == "__main__":
    unittest.main()
