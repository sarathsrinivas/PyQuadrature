import torch as tc
import numpy as np
from PyQuadrature import *
from itertools import product
import pytest as pyt

n = (10, 20, 30, 50)
a = (-5, -1, 0, 0.5)
b = (1, 5)

tparams = list(product(n, a, b))


@pyt.mark.parametrize("n,a,b", tparams)
def test_gauss_legendre(n, a, b, tol=1e-7):

    x, w = gauss_legendre(n, a=a, b=b)

    I = np.empty(3)
    I_exct = np.empty_like(I)

    I[0] = x.mul(x).mul_(w).sum(-1).numpy()
    I_exct[0] = (1 / 3.0) * (b**3 - a**3)

    I[1] = x.sin().mul_(w).sum(-1).numpy()
    I_exct[1] = np.cos(a) - np.cos(b)

    I[2] = x.mul(-1.0).exp_().mul_(w).sum(-1).numpy()
    I_exct[2] = np.exp(-a) - np.exp(-b)

    assert np.allclose(I, I_exct, atol=tol)


@pyt.mark.parametrize("nleb", (9, 11, 13, 15, 17))
def test_lebedev_quad(nleb, tol=1e-6):

    th, phi, w = lebedev_quad(nleb)

    r = tc.ones_like(th)

    x, y, z = sph_to_cart(r, th, phi)

    x2 = x.square()
    y2 = y.square()
    z2 = z.square()

    I = np.empty(4)
    I_exct = np.empty_like(I)

    I[0] = (1 + x + y2 + x2 * y + x2 * x2 + y2 * y2 * y +
            x2 * y2 * z2).mul_(w).sum(-1).numpy()
    I[1] = x.mul(y).mul(z).mul_(w).sum(-1).numpy()
    I[2] = tc.tanh(z - x - y).add_(1.0).mul_(w).sum(-1).numpy()
    I[3] = w.sum(-1).numpy()

    I_exct[0] = 19.388114662154152
    I_exct[1] = 0
    I_exct[2] = 12.566370614359172
    I_exct[3] = 12.566370614359172

    assert np.allclose(I, I_exct, atol=tol)


nr = (10, 20, 50)
nleb = (9, 11, 13, 15, 17)
rmax = (1, 3, 5)

tparams = list(product(nr, nleb, rmax))


@pyt.mark.parametrize("nr,nleb,rmax", tparams)
def test_ball_quad(nr, nleb, rmax, tol=1e-6):

    r, th, phi, w = ball_quad(rmax=rmax, nr=nr, nleb=nleb)

    x, y, z = sph_to_cart(r, th, phi)

    I = np.empty(5)
    I_exct = np.empty_like(I)

    I[0] = w.sum(-1).numpy()
    I_exct[0] = (4 * np.pi / 3.0) * rmax**3

    I[1] = x.square().mul(y.square()).mul(z.square()).mul_(w).sum(-1).numpy()
    I_exct[1] = 4 * np.pi * rmax**9 / 945

    I[2] = x.mul(y).mul(z).mul_(w).sum(-1).numpy()
    I_exct[2] = 0

    I[3] = x.mul(z).mul_(w).sum(-1).numpy()
    I_exct[3] = 0

    I[4] = y.mul(z).mul_(w).sum(-1).numpy()
    I_exct[4] = 0

    assert np.allclose(I, I_exct, atol=tol)
