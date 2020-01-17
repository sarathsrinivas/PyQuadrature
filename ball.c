#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_quadrature.h"

#define PI (3.1415926535897)

void get_ball_quadrature(double **q, double **wq, unsigned long *nq, int nleb, unsigned long nr,
			 double rmax, char *path)
{
	double *r, *omg, *wr, *th, *phi, *womg;
	unsigned long i, j, nomg;

	r = malloc(nr * sizeof(double));
	assert(r);
	wr = malloc(nr * sizeof(double));
	assert(wr);

	gauss_grid_create(nr, r, wr, 0, rmax);

	get_lebedev_grid(&th, &phi, &womg, nleb, &nomg, path);
	assert(th);
	assert(phi);
	assert(womg);

	*nq = nomg * nr;

	*q = malloc(3 * nomg * nr * sizeof(double));
	assert(*q);
	*wq = malloc(nomg * nr * sizeof(double));
	assert(*wq);

	for (i = 0; i < nomg; i++) {
		for (j = 0; j < nr; j++) {

			(*q)[3 * nr * i + 3 * j + 0] = r[j];
			(*q)[3 * nr * i + 3 * j + 1] = th[i];
			(*q)[3 * nr * i + 3 * j + 2] = phi[i];

			(*wq)[nr * i + j] = womg[i] * wr[j];
		}
	}

	free(womg);
	free(phi);
	free(th);
	free(wr);
	free(r);
}
