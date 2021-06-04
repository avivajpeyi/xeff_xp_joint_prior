import os
import unittest

import numpy as np
import pandas as pd

from effective_spins import distribution_rules
from effective_spins import probability_cacher
from .utils import uniform_distribution


class TestCaching(unittest.TestCase):

    def setUp(self):
        self.outdir = "tests/caching_test_outdir"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_caching(self):
        a, pdf_a, dist_a = uniform_distribution()
        b, pdf_b, dist_b = uniform_distribution(0, 1)
        a_vals = np.linspace(-1, 1, len(a))
        z_vals = np.linspace(-1, 2, len(a))
        pdf_c = distribution_rules.sum_distribution(
            z_vals=z_vals, a_vals=a_vals, pdf_a=dist_a.prob, pdf_b=dist_b.prob
        )
        df = pd.DataFrame(dict(a=a_vals, b=z_vals, p=pdf_c))
        fname = f"{self.outdir}/dat.h5"
        probability_cacher.store_probabilities(df, fname)
        probability_cacher.load_probabilities(fname)

    def test_plotting(self):
        a = np.linspace(0, 1, 100)
        b = np.linspace(0, 1, 100)
        A, B = np.meshgrid(a, b)
        c = A * 2 + 4 * B ** 2
        fname = f"{self.outdir}/dat.png"
        probability_cacher.plot_probs(A.ravel(), B.ravel(), c.ravel(), 'a', 'b', 'c',
                                      fname)


if __name__ == '__main__':
    unittest.main()
