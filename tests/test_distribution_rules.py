import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest
from .utils import uniform_distribution

from effective_spins import distribution_rules


class TestDistributionRules(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/distribution_outdir"
        os.makedirs(self.outdir, exist_ok=True)
        self.N = 10000



    # def tearDown(self):
    #     import shutil
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_sum_rule(self):
        a, pdf_a, dist_a = uniform_distribution(N=self.N)
        b, pdf_b, dist_b = uniform_distribution(0, 1, N=self.N)
        a_vals = np.linspace(-1, 1, self.N)
        z_vals = np.linspace(-1, 2, self.N)
        pdf_c = distribution_rules.sum_distribution(
            z_vals=z_vals, a_vals=a_vals, pdf_a=dist_a.prob, pdf_b=dist_b.prob
        )
        self.assertGreater(np.sum(pdf_c), 1)

    def test_product_rule(self):
        a, pdf_a, dist_a = self.uniform_distribution(N=self.N)
        b, pdf_b, dist_b = self.uniform_distribution(0, 1,N=self.N)
        a_vals = np.linspace(-1, 1, self.N)
        z_vals = np.linspace(-1, 1, self.N)
        pdf_c = distribution_rules.product_distribution(
            z_vals=z_vals, a_vals=a_vals, pdf_a=dist_a.prob, pdf_b=dist_b.prob
        )
        self.assertGreater(np.sum(pdf_c), 1)

    def test_inverse_rule(self):
        a, pdf_a, dist_a = self.uniform_distribution(N=self.N)
        a_vals = np.linspace(-10, 10, self.N)
        pdf_c = distribution_rules.inverse_distribution(
            a_vals=a_vals, pdf_a=dist_a.prob
        )
        self.assertGreater(np.sum(pdf_c), 1)

    @pytest.mark.plot
    def test_dist_plots(self):
        a, pdf_a, dist_a = self.uniform_distribution(N=self.N)
        b, pdf_b, dist_b = self.uniform_distribution(0, 1,N=self.N)
        a_vals = np.linspace(-10, 10, self.N)
        z_vals = np.linspace(-10, 10, self.N)
        samples_inverse = 1.0 / a
        samples_sum = a + b
        sumples_prod = a * b

        plot_distribution(
            samples_sum,
            distribution_rules.sum_distribution(
                z_vals=z_vals, a_vals=a_vals, pdf_a=dist_a.prob, pdf_b=dist_b.prob
            ),
            z_vals,
            f"{self.outdir}/sum.png",
            title="Sum of Unif"
        )
        plot_distribution(
            sumples_prod,
            distribution_rules.product_distribution(
                z_vals=z_vals, a_vals=a_vals, pdf_a=dist_a.prob, pdf_b=dist_b.prob
            ),
            z_vals,
            f"{self.outdir}/prod.png",
            title="Prod of Unif"
        )

        plot_distribution(
            samples_inverse,
            distribution_rules.inverse_distribution(
                a_vals=a_vals, pdf_a=dist_a.prob
            ),
            a_vals,
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
