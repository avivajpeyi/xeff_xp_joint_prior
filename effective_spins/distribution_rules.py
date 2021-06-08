from typing import Callable, Optional

from scipy.interpolate import interp1d

from .cupy_utils import trapz, xp


def product_distribution(
        z_vals: xp.array,
        a_vals: xp.array,
        pdf_a: Callable,
        pdf_b: Callable,
        kwargs_a={},
        kwargs_b={},
):
    """
    f_{Z}(z) = int f_{A}(a) f_{B}(z/a) 1/|a| da
    """
    fz = xp.zeros(len(z_vals))
    for i, z in enumerate(z_vals):
        fa = pdf_a(a_vals, **kwargs_a)
        fb = pdf_b(z / a_vals, **kwargs_b)
        fz[i] = trapz(y=fa * fb * (1 / xp.abs(a_vals)), x=a_vals)
    return fz


def sum_distribution(
        z_vals: xp.array,
        a_vals: xp.array,
        pdf_a: Callable,
        pdf_b: Callable,
        kwargs_a={},
        kwargs_b={},
):
    """
    f_{Z}(z) = sum_{a in A} f_{A}(a) f_{B}(z-a) da
    """
    fz = xp.zeros(len(z_vals))
    for i, z in enumerate(z_vals):
        fa = pdf_a(a_vals, **kwargs_a)
        fb = pdf_b(z - a_vals, **kwargs_b)
        fz[i] = trapz(y=fa * fb, x=a_vals)
    return fz


def inverse_distribution(z_vals, pdf_a, kwargs_a={}):
    """
    z=1/a
    f_{Z}(z) = (1/|z|)^2 * f_{A}(1/z)
    """
    return (1 / xp.abs(z_vals) ** 2) * pdf_a(1.0 / z_vals, **kwargs_a)


def translate_distribution(z_vals, pdf_a, scale=1.0, translate=0.0, kwargs_a={}):
    """
    s = scale constant
    t = translate constant
    z = s * a + t
    f_{Z}(z) = (1/|s|) * f_{A}((z-t)/s)
    """
    return (1 / xp.abs(scale)) * pdf_a(((z_vals - translate) / scale), **kwargs_a)


def prod_dist_interp(z_vals: xp.ndarray, a_vals: xp.ndarray, pdf_a: Callable, pdf_b: Callable) -> Callable:
    pdf_z = product_distribution(z_vals=z_vals, a_vals=a_vals, pdf_a=pdf_a, pdf_b=pdf_b)
    return interp1d(x=z_vals, y=pdf_z, bounds_error=False)


def sum_dist_interp(z_vals: xp.ndarray, a_vals: xp.ndarray, pdf_a: Callable, pdf_b: Callable) -> Callable:
    pdf_z = sum_distribution(z_vals=z_vals, a_vals=a_vals, pdf_a=pdf_a, pdf_b=pdf_b)
    return interp1d(x=z_vals, y=pdf_z, bounds_error=False)


def inv_dist_interp(z_vals: xp.ndarray, pdf_a: Callable) -> Callable:
    pdf_z = inverse_distribution(z_vals=z_vals, pdf_a=pdf_a)
    return interp1d(x=z_vals, y=pdf_z, bounds_error=False)


def translate_dist_interp(z_vals: xp.ndarray, pdf_a: Callable, scale: Optional[float] = 1.0,
                          translate: Optional[float] = 0.0) -> Callable:
    pdf_z = translate_distribution(z_vals=z_vals, pdf_a=pdf_a, scale=scale, translate=translate)
    return interp1d(x=z_vals, y=pdf_z, bounds_error=False)
