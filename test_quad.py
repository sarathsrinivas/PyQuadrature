import torch as tc
from .ball import *
from itertools import product
import pytest as pyt

n = (10, 20, 30, 50)
a = (-5, -1, 0, 0.5)
b = (1, 5)

tparams = list(product(n, a, b))


@pyt.mark.parametrize("n,a,b", tparams)
def test_gauss_legendre(n, a, b, tol=1e-7):

    x, w = gauss_legendre(n, a=a, b=b)

    I1 = x.mul(x).mul_(w).sum(-1).numpy()
    I2 = x.sin().mul_(w).sum(-1).numpy()
    I3 = x.mul(-1.0).exp_().mul_(w).sum(-1).numpy()

    I1_exact = (1 / 3.0) * (b**3 - a**3)
    I2_exact = np.cos(a) - np.cos(b)
    I3_exact = np.exp(-a) - np.exp(-b)

    assert np.allclose(I1, I1_exact, atol=tol)
    assert np.allclose(I2, I2_exact, atol=tol)
    assert np.allclose(I3, I3_exact, atol=tol)


@pyt.mark.parametrize("nleb", (9, 11, 13, 15, 17))
def test_lebedev_quad(nleb, tol=1e-6):

    th, phi, w = lebedev_quad(nleb)

    r = tc.ones_like(th)

    x, y, z = sph_to_cart(r, th, phi)

    x2 = x.square()
    y2 = y.square()
    z2 = z.square()

    I1 = (1 + x + y2 + x2 * y + x2 * x2 + y2 * y2 * y +
          x2 * y2 * z2).mul_(w).sum(-1).numpy()
    I2 = x.mul(y).mul(z).mul_(w).sum(-1).numpy()
    I3 = tc.tanh(z - x - y).add_(1.0).mul_(w).sum(-1).numpy()
    I4 = w.sum(-1).numpy()

    I1_exct = 19.388114662154152
    I2_exct = 0
    I3_exct = 12.566370614359172
    I4_exct = 12.566370614359172

    assert np.allclose(I1, I1_exct, atol=tol)
    assert np.allclose(I2, I2_exct, atol=tol)
    assert np.allclose(I3, I3_exct, atol=tol)
    assert np.allclose(I4, I4_exct, atol=tol)


nr = (10, 20, 50)
nleb = (9, 11, 13, 15, 17)
rmax = (1, 3, 5)

tparams = list(product(nr, nleb, rmax))


@pyt.mark.parametrize("nr,nleb,rmax", tparams)
def test_ball_quad(nr, nleb, rmax, tol=1e-6):

    r, th, phi, w = ball_quad(rmax=rmax, nr=nr, nleb=nleb)

    x, y, z = sph_to_cart(r, th, phi)

    I1 = w.sum(-1).numpy()
    I1_exct = (4 * np.pi / 3.0) * rmax**3

    I2 = x.square().mul(y.square()).mul(z.square()).mul_(w).sum(-1).numpy()
    I2_exct = 4 * np.pi * rmax**9 / 945

    assert np.allclose(I1, I1_exct, atol=tol)
    assert np.allclose(I2, I2_exct, atol=tol)
