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
	char fname[50];
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
int test_get_lebedev_grid()
{
	double *theta, *phi, *wleb, x, y, z, x2, y2, z2, I[4], Icomp[4];
	unsigned long ngrid, l, i;
	int nleb[6] = {3, 5, 7, 9, 11, 13};

	Icomp[0] = 19.388114662154152;
	Icomp[1] = 0;
	Icomp[2] = 12.566370614359172;
	Icomp[3] = 12.566370614359172;

	printf("PRECISION | GRID-SIZE | COMPUTED | ANALYTICAL | ERROR\n");
	

	for (i = 0; i < 6; i++) {

		I[0] = 0;
		I[1] = 0;
		I[2] = 0;
		I[3] = 0;

		get_lebedev_grid(&theta, &phi, &wleb, nleb[i], &ngrid, "lib_quadrature/leb_data");

		assert(theta);
		assert(phi);
		assert(wleb);
		for (l = 0; l < ngrid; l++) {
			x = cos(phi[l]) * sin(theta[l]);
			y = sin(phi[l]) * sin(theta[l]);
			z = cos(theta[l]);

			x2 = x * x;
			y2 = y * y;
			z2 = z * z;

			I[0] += wleb[l] * (1 + x + y2 + x2 * y + x2 * x2 + y2 * y2 * y + x2 * y2 * z2);
			I[1] += wleb[l] * x * y * z;
			I[2] += wleb[l] * (1 + tanh(z - x - y));
			I[3] += wleb[l];
		}

		I[0] *= 4 * PI;
		I[1] *= 4 * PI;
		I[2] *= 4 * PI;
		I[3] *= 4 * PI;

		for (l = 0; l < 4; l++)
			printf("%03d %03lu %+.15E %+.15E %+.15E\n", nleb[i], ngrid, I[l], Icomp[l],
			       fabs(I[l] - Icomp[l]));

		free(theta);
		free(phi);
		free(wleb);

		printf("--------------------\n");
	}

	return 0;
}
