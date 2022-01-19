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
        self.setup_data()

    def setup_data(self):
        xeff_range = [0.508, 0.51]
        s = utils.get_traditional_samples(10 ** 7)
        s = s[s['xeff'] > xeff_range[0]]
        self.s = s[s['xeff'] < xeff_range[1]]
        self.avg_xeff = 0.5095095095095095
        a1, a2, q, cos2 = xp_given_xeff.get_param_grid()
        p_a1, p_a2, p_q, p_cos2 = xp_given_xeff.get_p_param_given_xeff(xeff=self.avg_xeff)
        self.param_grid = (a1, a2, q, cos2)
        self.p_params = (p_a1, p_a2, p_q, p_cos2)

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
        s = self.s
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
        axs[0].set_title(f"xeff~{self.avg_xeff:0.2}")

        a_func, b_func, c_func, d_func, xp_func = get_functions(self.avg_xeff)
        axs[0].plot(large_bin, a_func(large_bin), linewidth=3)
        axs[1].plot(med_bin, b_func(med_bin), linewidth=3)
        axs[2].plot(small_bin, c_func(small_bin), linewidth=3)
        axs[3].plot(small_bin, d_func(small_bin), linewidth=3)
        axs[4].plot(small_bin, xp_func(small_bin), linewidth=3)

        plt.tight_layout()
        plt.savefig(f'{self.outdir}/hist.png')
        plt.close()

    def test_p_a2q(self):
        s = self.s
        a1, a2, q, cos2 = self.param_grid
        p_a1, p_a2, p_q, p_cos2 = self.p_params
        a2q = np.linspace(0.01, 1, 100)
        p_a2q = xp_given_xeff.prod_dist_interp(z_vals=a2q, a_vals=a2, pdf_a=p_a2, pdf_b=p_q)
        _p_a2q_vals = p_a2q(a2q)
        self.assertGreater(sum(_p_a2q_vals), 0)
        plt.close('all')
        plt.hist(s.a2 * s.q, density=True, bins=50)
        plt.plot(a2q, _p_a2q_vals, linewidth=3)
        plt.ylabel(f"P(a2q|xeff={self.avg_xeff:0.2f})")
        plt.ylim(0, 5)
        plt.xlabel(f"a2q|xeff={self.avg_xeff:0.2f}")
        plt.tight_layout()
        plt.savefig(f"{self.outdir}/a2q_test.png")

    def test_c(self):
        s = self.s
        a1, a2, q, cos2 = self.param_grid
        p_a1, p_a2, p_q, p_cos2 = self.p_params
        a, a_func = xp_given_xeff.get_p_xeffqplus1_a1_and_vals(p_q, p_a1, self.avg_xeff)
        b, b_func = xp_given_xeff.get_p_a2qc2_a1_and_vals(a2, p_a1, p_a2, p_q, p_cos2)
        c, c_func = xp_given_xeff.get_p_c_and_vals(a2, p_a1, p_a2, p_q, p_cos2, self.avg_xeff)
        p_neg_b = xp_given_xeff.translate_dist_interp(-b, b_func, scale=-1.0)
        plt.close('all')
        fig, axes = plt.subplots(3, 1, figsize=(4, 7))

        axes[0].hist(a_calc(s.a1,s.a2,s.q,s.cos2,s.xeff), density=True, bins=np.linspace(0,2, 100))
        axes[1].hist(-b_calc(s.a1,s.a2,s.q,s.cos2,s.xeff), density=True, bins=np.linspace(-1,1, 100))
        axes[2].hist(c_calc(s.a1,s.a2,s.q,s.cos2,s.xeff), density=True, bins=50)

        axes[0].plot(a, a_func(a), linewidth=3)
        axes[0].set_xlim(0,2)
        axes[1].plot(-b, p_neg_b(-b), linewidth=3)
        axes[1].set_xlim(-1,1)
        axes[2].plot(c, c_func(c), linewidth=3)

        axes[0].set_xlabel("a")
        axes[1].set_xlabel("-b")
        axes[2].set_xlabel("a-b")

        plt.tight_layout()
        plt.savefig(f"{self.outdir}/c_test.png")

    def test_p_xeffqplus1_a1(self):
        s = self.s
        p_a1, p_a2, p_q, p_cos2 = self.p_params

        # p_a
        a, p_a = xp_given_xeff.get_p_xeffqplus1_a1_and_vals(p_q=p_q, p_a1=p_a1, xeff=self.avg_xeff)
        _p_a_vals = p_a(a)
        self.assertGreater(sum(_p_a_vals), 0)

        # p(q+1)
        qplus1 = np.linspace(0.1, 2, 100)
        p_qplus1 = xp_given_xeff.translate_dist_interp(z_vals=qplus1, pdf_a=p_q, translate=1)
        _p_qplus1_vals = p_qplus1(qplus1)

        # p(xeff(q+1))
        xeffqplus1 = np.linspace(0.1, 2, 100)
        p_xeffqplus1 = xp_given_xeff.translate_dist_interp(z_vals=xeffqplus1, pdf_a=p_qplus1, scale=self.avg_xeff)
        _p_xeffqplus1_vals = p_xeffqplus1(xeffqplus1)

        # p_inv_a1
        p_inv_a1 = xp_given_xeff.get_p_inv_a1(p_a1)
        inv_a1 = np.linspace(0, 20)
        _p_inv_a1_vals = p_inv_a1(inv_a1)

        plt.close('all')
        fig, axes = plt.subplots(3, 1, figsize=(4, 7))

        axes[0].hist(1 / s.a1, density=True, bins=50)
        axes[1].hist(s.xeff * (s.q + 1), density=True, bins=50)
        axes[2].hist(s.xeff * (s.q + 1) / s.a1, density=True, bins=50)

        axes[0].plot(inv_a1, _p_inv_a1_vals, linewidth=3)
        axes[1].plot(xeffqplus1, _p_xeffqplus1_vals, linewidth=3)
        axes[2].plot(a, _p_a_vals, linewidth=3)

        axes[0].set_xlabel("1/a1")
        axes[1].set_xlabel("xeff(q+1)")
        axes[2].set_xlabel("xeff(q+1)/a1")

        plt.tight_layout()
        plt.savefig(f"{self.outdir}/a_test.png")

    def test_plot_p_param_given_xeff(self):
        s = self.s
        a1, a2, q, cos2 = self.param_grid
        p_a1, p_a2, p_q, p_cos2 = self.p_params

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
            a.set_ylabel(f'p(z|xeff)')
        axs[0].set_title(f"xeff~{self.avg_xeff:0.2}")

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
    _, a_func = xp_given_xeff.get_p_xeffqplus1_a1_and_vals(p_q, p_a1, xeff)
    _, b_func = xp_given_xeff.get_p_a2qc2_a1_and_vals(a2, p_a1, p_a2, p_q, p_cos2)
    _, c_func = xp_given_xeff.get_p_c_and_vals(a2, p_a1, p_a2, p_q, p_cos2, xeff)
    _, d_func = xp_given_xeff.get_p_d_and_vals(a2, p_a1, p_a2, p_q, p_cos2, xeff)
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
