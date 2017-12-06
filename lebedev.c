#include "../lib_io/lib_io.h"
#include "lib_quadrature.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define D2R (1.74532925199433E-02)
#define PI (3.1415926535897)

int get_lebedev_grid(double **theta, double **phi, double **wleb, int nleb, unsigned long *ngrid, char *path)
{
	char fname[100];
	unsigned long nr, ncol, i;
	double *mesh;

	sprintf(fname, "%s/lebedev_%03d.txt", path, nleb);
	mesh = read_file(fname, &nr, &ncol);
	assert(mesh);
	*ngrid = nr;
	*theta = malloc(nr * sizeof(double));
	assert(*theta);
	*phi = malloc(nr * sizeof(double));
	assert(*phi);
	*wleb = malloc(nr * sizeof(double));
	assert(*wleb);

	for (i = 0; i < nr; i++) {
		(*theta)[i] = D2R * mesh[ncol * i + 1];
		(*phi)[i] = D2R * mesh[ncol * i];
		(*wleb)[i] = mesh[ncol * i + 2];
	}

	free(mesh);

	return 0;
}

double test_get_lebedev_grid(int leb, int tfun)
{

	double *th, *phi, *wleb, x, y, z, x2, y2, z2, II[4], Icomp[4], Idiff[4];
	unsigned long ngrid, l;

	fprintf(stderr, "test_get_lebedev_grid() test #%d %s:%d\n", tfun, __FILE__, __LINE__);

	Icomp[0] = 19.388114662154152;
	Icomp[1] = 0;
	Icomp[2] = 12.566370614359172;
	Icomp[3] = 12.566370614359172;

	II[0] = 0;
	II[1] = 0;
	II[2] = 0;
	II[3] = 0;

	get_lebedev_grid(&th, &phi, &wleb, leb, &ngrid, "lib_quadrature/leb_data");
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

	II[0] *= 4 * PI;
	II[1] *= 4 * PI;
	II[2] *= 4 * PI;
	II[3] *= 4 * PI;

	for (l = 0; l < 4; l++) {
		Idiff[l] = fabs(II[l] - Icomp[l]);
	}

	free(th);
	free(phi);
	free(wleb);

	return Idiff[tfun];
}
