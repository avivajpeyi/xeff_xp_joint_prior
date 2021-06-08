import os
import unittest

import matplotlib.pyplot as plt
import numpy as np


from effective_spins import xp_given_xeff
from . import utils


class TestXpGivenXeffPrior(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/xp_given_xeff_testoutdir"
        os.makedirs(self.outdir, exist_ok=True)

    # def tearDown(self):
    #     import shutil
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    # def test_xp_given_xeff_prior(self):
    #     val = xp_given_xeff.p_xp_given_xeff(xp=0.1, xeff=1)


    def test_plots(self):
        """
        xp = a1 sqrt(1 - (a+b)^2)
        = a1 * d

        where
        a = xeff(q+1)/a1
        b = a2qcos2/a1
        c = a + b
        d = sqrt(1-c^2)

        """
        xeff_range = [0.49, 0.51]
        s = utils.get_traditional_samples(10 ** 7)
        s = s[s['xeff'] > xeff_range[0]]
        s = s[s['xeff'] < xeff_range[1]]
        avg_xeff = 0.5
        print(len(s))
        fig, axs = plt.subplots(5, 1, figsize=(3, 8))
        params = dict(a1=s.a1, a2=s.a2, q=s.q, cos2=s.cos2, xeff=s.xeff)
        large_bin = np.linspace(0, 20, 100)
        med_bin = np.linspace(-1, 1, 100)
        small_bin = np.linspace(0, 1, 100)
        axs[0].hist(a_calc(**params), density=True, bins=large_bin)
        axs[1].hist(b_calc(**params), density=True, bins=med_bin)
        axs[2].hist(c_calc(**params), density=True, bins=small_bin)
        axs[3].hist(d_calc(**params), density=True, bins=small_bin)
        axs[4].hist(xp_calc(**params), density=True, bins=small_bin)
        axs[0].set_xlabel('z=a')
        axs[1].set_xlabel('z=b')
        axs[2].set_xlabel('z=c')
        axs[3].set_xlabel('z=d')
        axs[4].set_xlabel('z=xp')
        for a in axs:
            a.set_ylabel('p(z)')
        axs[0].set_title(f"xeff~{avg_xeff:0.2}")

        a_func, b_func, c_func, d_func, xp_func = get_functions(avg_xeff)
        axs[0].plot(large_bin, a_func(large_bin))
        axs[1].plot(med_bin, b_func(med_bin))
        axs[2].plot(small_bin, c_func(small_bin))
        axs[3].plot(small_bin, d_func(small_bin))
        axs[4].plot(small_bin, xp_func(small_bin))

        plt.tight_layout()
        plt.savefig(f'{self.outdir}/hist.png')
        plt.close()


    def test_p_a2q(self):
        xeff_range = [0.49, 0.51]
        s = utils.get_traditional_samples(10 ** 7)
        s = s[s['xeff'] > xeff_range[0]]
        s = s[s['xeff'] < xeff_range[1]]
        avg_xeff = 0.5

        a1, a2, q, cos2 = xp_given_xeff.get_param_grid()
        p_a1, p_a2, p_q, p_cos2 = xp_given_xeff.get_p_param_given_xeff(xeff=0.5)

        a2q = np.linspace(0, 1, 100)
        p_a2q = xp_given_xeff.prod_dist_interp(z_vals=a2q, a_vals=a2, pdf_a=p_a2, pdf_b=p_q)
        _p_a2q_vals = p_a2q(a2q)
        self.assertGreater(sum(_p_a2q_vals), 0)
        plt.close('all')
        plt.hist(s.a2 * s.q, density=True, bins=50)
        plt.plot(a2q, _p_a2q_vals)
        plt.ylabel(f"P(a2q|xeff={avg_xeff})")
        plt.ylim(0,5)
        plt.xlabel(f"a2q|xeff={avg_xeff}")
        plt.tight_layout()
        plt.savefig(f"{self.outdir}/a2q_test.png")


    def test_plot_p_param_given_xeff(self):
        xeff_range = [0.49, 0.51]
        s = utils.get_traditional_samples(10 ** 7)
        s = s[s['xeff'] > xeff_range[0]]
        s = s[s['xeff'] < xeff_range[1]]
        avg_xeff = 0.5

        a1, a2, q, cos2 = xp_given_xeff.get_param_grid()
        p_a1, p_a2, p_q, p_cos2 = xp_given_xeff.get_p_param_given_xeff(avg_xeff)

        fig, axs = plt.subplots(4, 1, figsize=(3, 7))

        axs[0].hist(s.a1, density=True, bins=50)
        axs[1].hist(s.a2, density=True, bins=50)
        axs[2].hist(s.q, density=True, bins=50)
        axs[3].hist(s.cos2, density=True, bins=50)

        axs[0].set_xlabel('z=a1')
        axs[1].set_xlabel('z=a2')
        axs[2].set_xlabel('z=q')
        axs[3].set_xlabel('z=cos2')

        for a in axs:
            a.set_ylabel('p(z)')
        axs[0].set_title(f"xeff~{avg_xeff:0.2}")

        axs[0].plot(a1, p_a1(a1))
        axs[1].plot(a2, p_a2(a2))
        axs[2].plot(q, p_q(q))
        axs[3].plot(cos2, p_cos2(cos2))

        plt.tight_layout()
        plt.savefig(f'{self.outdir}/original_hist.png')
        plt.close()


def get_functions(xeff):
    a1, a2, q, cos2 = xp_given_xeff.get_param_grid()
    p_a1, p_a2, p_q, p_cos2 = xp_given_xeff.get_p_param_given_xeff(xeff)
    _, a_func = xp_given_xeff.get_p_qplus1_a1_and_vals(p_q, p_a1)
    _, b_func = xp_given_xeff.get_p_a2qc2_a1_and_vals(a2, p_a1, p_a2, p_q, p_cos2)
    _, c_func = xp_given_xeff.get_p_c_and_vals(a2, p_a1, p_a2, p_q, p_cos2)
    _, d_func = xp_given_xeff.get_p_c_and_vals(a2, p_a1, p_a2, p_q, p_cos2)
    xp_func = xp_given_xeff.get_p_xp_given_xeff_and_vals(xeff)
    return a_func, b_func, c_func, d_func, xp_func


def add_b_prediction_to_plot(b_vals, ax, xeff):
    a1, a2, q, cos2 = xp_given_xeff.get_param_grid()
    p_a1, p_a2, p_q, p_cos2 = xp_given_xeff.get_p_param_given_xeff(xeff)
    vals, p_func = xp_given_xeff.get_p_a2qc2_a1_and_vals(a2, p_a1, p_a2, p_q, p_cos2)
    ax.plot(p_func(b_vals), b_vals)


def a_calc(a1, a2, q, cos2, xeff):
    return np.nan_to_num(xeff * (q + 1) / a1)


def b_calc(a1, a2, q, cos2, xeff):
    return np.nan_to_num(a2 * q * cos2 / a1)


def c_calc(a1, a2, q, cos2, xeff):
    return np.nan_to_num(a_calc(a1, a2, q, cos2, xeff) - b_calc(a1, a2, q, cos2, xeff))


def d_calc(a1, a2, q, cos2, xeff):
    return np.nan_to_num(np.sqrt(1 - c_calc(a1, a2, q, cos2, xeff) ** 2))


def xp_calc(a1, a2, q, cos2, xeff):
    return a1 * d_calc(a1, a2, q, cos2, xeff)


if __name__ == "__main__":
    unittest.main()
