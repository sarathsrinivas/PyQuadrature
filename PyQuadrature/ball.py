import torch as tc
import numpy as np
import scipy.special as sp
import pathlib

leb_path = pathlib.Path(__file__).parent.absolute()

tc.set_default_tensor_type(tc.DoubleTensor)


def gauss_legendre(n, a=0, b=1):

    x, w = sp.roots_legendre(n)

    x = tc.tensor(x)
    w = tc.tensor(w)

    x.mul_(0.5 * (b - a)).add_(0.5 * (b + a))
    w.mul_(0.5 * (b - a))

    return x, w


def lebedev_quad(nleb):

    leb_file = str(leb_path) + "/leb_data/lebedev_{:03}.txt".format(nleb)

    print(leb_file)

    dat = np.loadtxt(leb_file)

    D2R = 1.74532925199433E-02

    th = tc.tensor(D2R * dat[:, 1])
    phi = tc.tensor(D2R * dat[:, 0])
    w = tc.tensor(4.0 * np.pi * dat[:, 2])

    return th, phi, w


def ball_quad(rmax=1, nr=10, nleb=15):

    r, wr = gauss_legendre(nr, a=0, b=rmax)

    wr.mul_(r.square())

    th, phi, wleb = lebedev_quad(nleb)

    w = wr[:, None].mul(wleb[None, :])

    r, th, phi = tc.broadcast_tensors(r[:, None], th[None, :], phi[None, :])

    r = r.reshape(-1, 1).squeeze()
    th = th.reshape(-1, 1).squeeze()
    phi = phi.reshape(-1, 1).squeeze()
    w = w.reshape(-1, 1).squeeze()

    return r, th, phi, w


def sph_to_cart(r, th, phi):

    x = phi.cos().mul_(th.sin()).mul_(r)
    y = phi.sin().mul_(th.sin()).mul_(r)
    z = th.cos().mul_(r)

    return x, y, z
