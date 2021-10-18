// fms_trnlsp.t.cpp - Test nonlinear solver
#include "fms_trnlsp.h"

using namespace fms;

int test_fms_trnlsp()
{
	{
		// f(x) = x^2 - 2
		auto f = [](int,int,const double* x, double* fx) {
			*fx = (*x) * (*x) - 2; 
		};
		const auto df = [](int, int, const double* x, double* dfx) {
			*dfx = 2 * *x; 
		};
		double x = 1;
		trnslp p(1, 1, &x);
		p.f = f;
		p.df = df;
		p.init();
		int ret = 0, iter = 0, st_cr = 0;
		double r1 = 0, r2 = 0;

		int rci = 0;
		double x_[1] = { 0 }, df_[1] = { 0 };
		ret = p.check(x_, df_);
		ret = p.solver(x_, df_, &rci);
		ret = p.get(iter, st_cr, r1, r2);

	}

	return 0;
}
int test_fms_trnlsp_ = test_fms_trnlsp();
