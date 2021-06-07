import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest

from effective_spins import distribution_rules
from .utils import uniform_distribution


class TestDistributionRules(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/distribution_outdir"
        os.makedirs(self.outdir, exist_ok=True)
        self.N = 10000
        self.a, self.pdf_a, self.dist_a = uniform_distribution(N=self.N)
        self.b, self.pdf_b, self.dist_b = uniform_distribution(0, 1, N=self.N)
        self.a_vals = np.linspace(-10, 10, self.N)
        self.z_vals = np.linspace(-10, 10, self.N)

    # def tearDown(self):
    #     import shutil
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_sum_rule(self):
        pdf_c = distribution_rules.sum_distribution(
            z_vals=self.z_vals, a_vals=self.a_vals, pdf_a=self.dist_a.prob, pdf_b=self.dist_b.prob
        )
        self.assertGreater(np.sum(pdf_c), 1)

        dist_c = distribution_rules.sum_dist_interp(z_vals=self.z_vals, a_vals=self.a_vals, pdf_a=self.dist_a.prob,
                                                    pdf_b=self.dist_b.prob)
        self.assertGreater(np.sum(dist_c(self.z_vals)), 1)

    def test_product_rule(self):
        pdf_c = distribution_rules.product_distribution(
            z_vals=self.z_vals, a_vals=self.a_vals, pdf_a=self.dist_a.prob, pdf_b=self.dist_b.prob
        )
        self.assertGreater(np.sum(pdf_c), 1)

        dist_c = distribution_rules.prod_dist_interp(z_vals=self.z_vals, a_vals=self.a_vals, pdf_a=self.dist_a.prob,
                                                     pdf_b=self.dist_b.prob)
        self.assertGreater(np.sum(dist_c(self.z_vals)), 1)

    def test_translate_rule(self):
        s = 2
        t = 1

        pdf_c = distribution_rules.translate_distribution(
            z_vals=self.z_vals, pdf_a=self.dist_a.prob, scale=s, translate=t
        )
        self.assertGreater(np.sum(pdf_c), 1)

        dist_c = distribution_rules.translate_dist_interp(z_vals=self.z_vals, pdf_a=self.dist_a.prob, scale=s,
                                                          translate=t)
        self.assertGreater(np.sum(dist_c(self.z_vals)), 1)

    @pytest.mark.plot
    def test_trans_dist_plots(self):
        s, t = 2, 1
        plot_distribution(
            (self.a * s) + t,
            distribution_rules.translate_distribution(
                z_vals=self.z_vals, pdf_a=self.dist_a.prob, scale=s, translate=t
            ),
            self.z_vals,
            f"{self.outdir}/translate.png",
            title="translate and scale of unif",
            bins=np.linspace(-10, 10, 100),
        )

    def test_inverse_rule(self):
        a, pdf_a, dist_a = uniform_distribution(N=self.N)
        z_vals = np.linspace(-10, 10, self.N)
        pdf_c = distribution_rules.inverse_distribution(
            z_vals=self.z_vals, pdf_a=self.dist_a.prob,
        )
        self.assertGreater(np.sum(pdf_c), 1)

        dist_c = distribution_rules.inv_dist_interp(z_vals=z_vals, pdf_a=dist_a.prob)
        self.assertGreater(np.sum(dist_c(z_vals)), 1)

    @pytest.mark.plot
    def test_sum_dist_plots(self):
        plot_distribution(
            self.a + self.b,
            distribution_rules.sum_distribution(
                z_vals=self.z_vals, a_vals=self.a_vals, pdf_a=self.dist_a.prob, pdf_b=self.dist_b.prob
            ),
            self.z_vals,
            f"{self.outdir}/sum.png",
            title="Sum of Unif"
        )

    @pytest.mark.plot
    def test_prod_dist_plots(self):
        plot_distribution(
            self.a * self.b,
            distribution_rules.product_distribution(
                z_vals=self.z_vals, a_vals=self.a_vals, pdf_a=self.dist_a.prob, pdf_b=self.dist_b.prob
            ),
            self.z_vals,
            f"{self.outdir}/prod.png",
            title="Prod of Unif"
        )

    @pytest.mark.plot
    def test_inv_dist_plots(self):
        plot_distribution(
            1 / self.a,
            distribution_rules.inverse_distribution(
                z_vals=self.z_vals, pdf_a=self.dist_a.prob
            ),
            self.z_vals,
            f"{self.outdir}/inverse.png",
            title="Inverse of unif",
            bins=np.linspace(-10, 10, 100),
        )


def plot_distribution(samples, pdf, xvals, fname, title="", bins=50):
    plt.hist(samples, density=True, label="Hist(samples) PDF", bins=bins)
    plt.plot(xvals, pdf, label="Numerical PDF")
    plt.xlim(min(xvals), max(xvals))
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.legend()
    plt.suptitle(title)
    plt.savefig(fname)
    plt.close("all")


if __name__ == "__main__":
    unittest.main()
