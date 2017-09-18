#include <stddef.h>
int gauss_grid_create(size_t size, double* x, double* w, double xmin, double xmax);
int test_gauss_grid_create(void);
int get_lebedev_grid(double **theta, double **phi, double **wleb, int nleb, unsigned long *ngrid, char* path);	
int test_get_lebedev_grid(void);
