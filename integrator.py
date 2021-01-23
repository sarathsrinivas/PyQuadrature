import torch as tc
from torch import Tensor
from typing import Sequence
from lib_quadrature.ball import ball_quad, sph_to_cart, gauss_legendre


class Integrator:
    """
     Base class for numerical integrations using quadratures.
    """

    def __init__(self) -> None:
        self.limits: dict = NotImplemented
        self.nx: int = NotImplemented
        self.dim: int = NotImplemented
        self.x: Tensor = NotImplemented
        self.wt: Tensor = NotImplemented
        return None

    def get_quadrature(self, limits: dict) -> None:
        raise NotImplementedError

    def integrate(self, f_x: Tensor) -> Tensor:
        integ = f_x.mul(self.wt).sum(-1)
        return integ

    def integrate_error(self, f_x: Tensor, err_f_x: Tensor) -> Tensor:
        raise NotImplementedError


class Gauss_Legendre_quad(Integrator):
    """
     Gauss Legendre quadrature with Lebedev for 1D.
    """

    def __init__(self, n: int = 10) -> None:
        super().__init__()
        self.dim = 1
        self.n = n
        return None

    def get_quadrature(self, limits: dict) -> None:
        a = limits["x"][0]
        b = limits["x"][1]
        self.x, self.wt = gauss_legendre(self.n, a=a, b=b)
        return None


class Ball_lebedev_gauss(Integrator):
    """
     3-ball quadrature with Lebedev for unit sphere and Guass for radial.
    """

    def __init__(self, nleb: int = 17, nr: int = 20) -> None:
        super().__init__()
        self.dim = 3
        self.nr = nr
        self.nleb = nleb
        return None

    def get_quadrature(self, limits: dict) -> None:
        rmax = limits["rmax"]
        r, th, phi, wt = ball_quad(rmax=rmax, nr=self.nr, nleb=self.nleb)
        self.x = tc.cat([r, th, phi])
        self.wt = wt
        return None


class Ball_lebedev_gauss_cart(Ball_lebedev_gauss):
    """
     3-ball quadrature with Lebedev for unit sphere and Guass for radial
     in cartesian form.
    """

    def get_quadrature(self, limits: dict) -> None:
        super().get_quadrature(limits)
        self.x = sph_to_cart(*self.x)
        return None


T_Integ = callable


class Tensor_prod_quadrature(Integrator):
    """
     Compose tensor product of quadratures
    """

    def __init__(self, integs: Sequence[Integrator]) -> None:
        super().__init__()
        self.integs = integs
        self.dim = sum([integ.dim for integ in self.integs])
        return None

    def get_quadrature(self, limits: dict) -> None:
        assert len(limits) == len(self.integs)

        for integ, limit in zip(self.integs, limits.values()):
            integ.get_quadrature(limit)

        xs = [integ.x for integ in self.integs]
        wts = [integ.wt for integ in self.integs]

        self.x = tc.stack([x.flatten() for x in tc.meshgrid(*xs)]).transpose_(
            -1, -2
        )

        self.wt = (
            tc.stack([wt.flatten() for wt in tc.meshgrid(*wts)])
            .transpose_(-1, -2)
            .prod(-1)
        )

        return None
