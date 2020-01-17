#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_io/lib_io.h>
#include "lib_quadrature.h"

#define D2R (1.74532925199433E-02)
#define PI (3.1415926535897)

void get_lebedev_grid(double **theta, double **phi, double **wleb, int nleb, unsigned long *ngrid,
		      char *path)
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
		(*wleb)[i] = 4 * PI * mesh[ncol * i + 2];
	}

	free(mesh);
}
