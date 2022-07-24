import torch as tc
from torch import Tensor
from itertools import product
import pytest as pyt
from PyQuadrature import (
    Integrator,
    Gauss_Legendre_quad,
    Ball_lebedev_gauss,
    Ball_lebedev_gauss_cart,
    Tensor_prod_quadrature,
)


def fun_1d(a: Tensor, x: Tensor) -> Tensor:
    f = a[:, None].mul(x[None, :]).sin_()
    return f


def fun_1d_integ(a: Tensor, xmin: float, xmax: float) -> Tensor:
    integ_f = tc.cos(a * xmin).div_(a) - tc.cos(a * xmax).div_(a)
    return integ_f


n = (10, 20, 30)
na = (5, 10)
xmin = (0, 1)
xmax = (1, 2)

tparam = list(product(n, na, xmin, xmax))


@pyt.mark.parametrize("n, na, xmin, xmax", tparam)
def test_gauss_legendre(n: int, na: int, xmin: float, xmax: float) -> None:
    a = tc.rand(na)
    integ_a_comp = fun_1d_integ(a, xmin, xmax)

    gleg = Gauss_Legendre_quad(n)
    limits = {"x": (xmin, xmax)}
    x, wt = gleg.get_quadrature(limits)
    f_ax = fun_1d(a, x)
    integ_a = gleg.integrate(f_ax, wt)

    assert tc.allclose(integ_a, integ_a_comp)


def fun_3d(a: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    t = x * y * z
    f = a[:, None].mul(t[None, :])
    return f


def fun_3d_integ(a: Tensor, xlim: tuple, ylim: tuple, zlim: tuple) -> Tensor:
    xi = 0.5 * (xlim[1] ** 2 - xlim[0] ** 2)
    yi = 0.5 * (ylim[1] ** 2 - ylim[0] ** 2)
    zi = 0.5 * (zlim[1] ** 2 - zlim[0] ** 2)

    integ = a.mul(xi * yi * zi)
    return integ


ns = ((10, 10, 10), (20, 10, 10), (10, 20, 30))
na = (5, 10)
xlim = ((1, 2), (0, 1))
ylim = ((0, 1), (1, 2))
zlim = ((0, 1), (1, 2))

tparam = list(product(n, na, xlim, ylim, zlim))


@pyt.mark.parametrize("ns, na, xlim, ylim, zlim", tparam)
def test_tensor_prod_quadrature(
    ns: tuple, na: int, xlim: tuple, ylim: tuple, zlim: tuple
) -> None:
    a = tc.ones(na)
    integ_a_comp = fun_3d_integ(a, xlim, ylim, zlim)

    nx = n[0]
    ny = n[1]
    nz = n[2]

    quadx = Gauss_Legendre_quad(nx)
    quady = Gauss_Legendre_quad(ny)
    quadz = Gauss_Legendre_quad(nz)

    quad3d = Tensor_prod_quadrature([quadx, quady, quadz])
    limits = {"x": {"x": xlim}, "y": {"x": ylim}, "z": {"x": zlim}}

    x, wt = quad3d.get_quadrature(limits)

    f_axyz = fun_3d(a, x[:, 0], x[:, 1], x[:, 2])

    integ_a = quad3d.integrate(f_axyz, wt)

    assert tc.allclose(integ_a, integ_a_comp)

    return None
