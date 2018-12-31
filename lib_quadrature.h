#include <stddef.h>
void gauss_grid_create(size_t size, double* x, double* w, double xmin, double xmax);
void gauss_grid_rescale(const double *x1, const double *w1, unsigned long size, double *x, double *w, double xmin, double xmax);
double test_gauss_grid_create(unsigned long n, int tfun);
