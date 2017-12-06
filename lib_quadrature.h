#include <stddef.h>
int gauss_grid_create(size_t size, double* x, double* w, double xmin, double xmax);
double test_gauss_grid_create(unsigned long n, int tfun);
int get_lebedev_grid(double **theta, double **phi, double **wleb, int nleb, unsigned long *ngrid, char* path);	
double test_get_lebedev_grid(int leb, int tfun);
