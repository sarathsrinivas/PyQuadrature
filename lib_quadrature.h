
/* GAUSS-LEGENDRE QUADRATURE */

void gauss_grid_create(unsigned long size, double *x, double *w, double xmin, double xmax);
void gauss_grid_rescale(const double *x1, const double *w1, unsigned long size, double *x,
			double *w, double xmin, double xmax);

/* LEBEDEV SPHERICAL QUADRATURE */

void get_lebedev_grid(double **theta, double **phi, double **wleb, int nleb, unsigned long *ngrid,
		      char *path);

/* QUADRATURE IN A 3-BALL */

void get_ball_quadrature(double **q, double **wq, unsigned long *nq, int nleb, unsigned long nr,
			 double rmax, char *path);

/* TESTS */

double test_gauss_grid_create(unsigned long n, int tfun);
double test_lebedev_grid(int leb, int tfun, char *leb_path);
double test_ball_quadrature(unsigned long nr, double rmax, int nleb, char *path);
void test_lib_quadrature(char *path);
