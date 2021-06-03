import os
import unittest

import pytest
from effective_spins.priors_conditional_on_xeff import p_xeff

from . import utils


class TestXeffPrior(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/xeff_testoutdir"
        os.makedirs(self.outdir, exist_ok=True)
        self.p = utils.get_traditional_prior()

    # def tearDown(self):
    #     import shutil
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_simple_p_xeff_calc(self):
        p_xeff_val = p_xeff(xeff=1, init_a1a2qcos2_prior=self.p)
        self.assertEqual(p_xeff_val, 0)
        p_xeff_val = p_xeff(xeff=-1, init_a1a2qcos2_prior=self.p)
        self.assertEqual(p_xeff_val, 0)
        p_xeff_val = p_xeff(xeff=0.0, init_a1a2qcos2_prior=self.p)
        self.assertGreater(p_xeff_val, 1.9)
        self.assertLess(p_xeff_val, 3)

    @pytest.mark.plot
    def test_plot_xeff_samples_and_p_xeff(self):
        samples = utils.get_traditional_samples()
        wrapped_peff = lambda x: p_xeff(x, self.p)
        fig = utils.plot_funct_and_samples(
            wrapped_peff,
            samples.xeff,
            [-1, 1],
            [r"$\chi_{\rm eff}$", r"$p(\chi_{\rm eff})$"]
        )
        fig.savefig(f"{self.outdir}/p_xeff.png")


if __name__ == "__main__":
    unittest.main()
