// fms_trnlsp.t.cpp - Test nonlinear solver
#include "fms_trnlsp.h"

using namespace fms;

int test_fms_trnlsp()
{
	int ret = 0;
	{
		double x = 1;
		trnslp p(1, 1, &x);
		p.f = [](int, int, const double* x, double* fx) {
			*fx = (*x) * (*x) - 2;
		};
		p.df = [](int, int, const double* x, double* dfx) {
			*dfx = 2 * *x;
		};
		ret = p.init();

		double x_[1] = { 0 }, df_[1] = { 0 };
		ret = p.check(x_, df_);
		
		int rci = 0;
		ret = p.solver(x_, df_, rci);
		
		int iter = 0, st_cr = 0;
		double r1 = 0, r2 = 0;
		ret = p.get(iter, st_cr, r1, r2);
	}

	return ret;
}
int test_fms_trnlsp_ = test_fms_trnlsp();
