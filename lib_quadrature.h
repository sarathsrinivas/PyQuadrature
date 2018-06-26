#include <stddef.h>
int gauss_grid_create(size_t size, double* x, double* w, double xmin, double xmax);
int gauss_grid_rescale(unsigned long size, double *x, double xmin, double xmax);
double test_gauss_grid_create(unsigned long n, int tfun);
