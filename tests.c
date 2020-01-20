#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_quadrature.h"

#define PI (3.1415926535897)

double test_gauss_grid_create(unsigned long n, int tfun)
{
	unsigned long i;
	double *x, *w;
	double a = 0, b = 2, In[3], Incomp[3], Indiff[3];

	fprintf(stderr, "test_gauss_grid_create(n = %lu) tfun = %d):\n", n, tfun);

	x = malloc(n * sizeof(double));
	assert(x);
	w = malloc(n * sizeof(double));
	assert(w);

	gauss_grid_create(n, x, w, a, b);

	In[0] = 0;
	In[1] = 0;
	In[2] = 0;
	for (i = 0; i < n; i++) {
		In[0] += x[i] * x[i] * w[i];
		In[1] += sin(x[i]) * w[i];
		In[2] += exp(-x[i]) * w[i];
	}

	Incomp[0] = (b * b * b / 3) - (a * a * a / 3);
	Incomp[1] = cos(a) - cos(b);
	Incomp[2] = exp(-a) - exp(-b);

	Indiff[0] = fabs(Incomp[0] - In[0]);
	Indiff[1] = fabs(Incomp[1] - In[1]);
	Indiff[2] = fabs(Incomp[2] - In[2]);

	free(x);
	free(w);

	return Indiff[tfun];
}

double test_lebedev_grid(int leb, int tfun, char *leb_path)
{

	double *th, *phi, *wleb, x, y, z, x2, y2, z2, II[4], Icomp[4], Idiff[4];
	unsigned long ngrid, l;

	fprintf(stderr, "test_get_lebedev_grid(leb = %d, tfun = %d):\n", leb, tfun);

	Icomp[0] = 19.388114662154152;
	Icomp[1] = 0;
	Icomp[2] = 12.566370614359172;
	Icomp[3] = 12.566370614359172;

	II[0] = 0;
	II[1] = 0;
	II[2] = 0;
	II[3] = 0;

	get_lebedev_grid(&th, &phi, &wleb, leb, &ngrid, leb_path);
	assert(th);
	assert(phi);
	assert(wleb);

	for (l = 0; l < ngrid; l++) {
		x = cos(phi[l]) * sin(th[l]);
		y = sin(phi[l]) * sin(th[l]);
		z = cos(th[l]);

		x2 = x * x;
		y2 = y * y;
		z2 = z * z;

		II[0] += wleb[l] * (1 + x + y2 + x2 * y + x2 * x2 + y2 * y2 * y + x2 * y2 * z2);
		II[1] += wleb[l] * x * y * z;
		II[2] += wleb[l] * (1 + tanh(z - x - y));
		II[3] += wleb[l];
	}

	for (l = 0; l < 4; l++) {
		Idiff[l] = fabs(II[l] - Icomp[l]);
	}

	free(th);
	free(phi);
	free(wleb);

	return Idiff[tfun];
}

double test_ball_quadrature(unsigned long nr, double rmax, int nleb, char *path)
{

	double vol, vol_num, *q, *wq, r;
	unsigned long i, nq;

	fprintf(stderr, "test_ball_quadrature(nr = %lu, rmax = %f, nleb = %d):\n", nr, rmax, nleb);

	get_ball_quadrature(&q, &wq, &nq, nleb, nr, rmax, path);
	assert(q);
	assert(wq);

	vol_num = 0;
	for (i = 0; i < nq; i++) {
		r = q[3 * i + 0];
		vol_num += wq[i] * r * r;
	}

	vol = 4 * PI * rmax * rmax * rmax / 3;

	free(q);
	free(wq);

	return fabs(vol - vol_num);
}

int verify(double terr, double tol)
{
	int ret;
	if (terr > tol) {
		ret = 1;
		fprintf(stderr, "T-ERROR: %+.15E TOL: %+.0E TEST FAILED  ***\n\n", terr, tol);
	} else {
		ret = 0;
		fprintf(stderr, "T-ERROR: %+.15E TOL: %+.0E TEST PASSED\n\n", terr, tol);
	}

	return ret;
}

void test_lib_quadrature(char *path)
{
	unsigned long ng;

	ng = 50;

	verify(test_gauss_grid_create(ng, 0), 1E-10);
	verify(test_gauss_grid_create(ng, 1), 1E-10);
	verify(test_gauss_grid_create(ng, 2), 1E-10);

	verify(test_lebedev_grid(15, 0, path), 1E-10);
	verify(test_lebedev_grid(15, 1, path), 1E-10);
	verify(test_lebedev_grid(15, 2, path), 1E-10);

	verify(test_lebedev_grid(17, 0, path), 1E-10);
	verify(test_lebedev_grid(17, 1, path), 1E-10);
	verify(test_lebedev_grid(17, 2, path), 1E-10);

	verify(test_ball_quadrature(10, 1, 15, path), 1E-7);
	verify(test_ball_quadrature(20, 1, 15, path), 1E-7);
	verify(test_ball_quadrature(10, 1, 17, path), 1E-7);
	verify(test_ball_quadrature(10, 3, 15, path), 1E-7);
}
