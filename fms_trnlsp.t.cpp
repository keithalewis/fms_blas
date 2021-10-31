// fms_trnlsp.t.cpp - Test nonlinear solver
#include <cassert>
#include "fms_trnlsp.h"

using namespace fms;

int test_fms_trnlsp()
{
	int ret = 0;
	{
		double x = 1;
		trnslp p(1, 1, &x);
		p.f = [](int, int, const double* x, double* fx, void*) {
			*fx = (*x) * (*x) - 2;
		};
		p.df = [](int, int, const double* x, double* dfx, void*) {
			*dfx = 2 * *x;
		};
		ret = p.init();
		assert(TR_SUCCESS == ret);

		double x_[1] = { 0 }, df_[1] = { 0 };
		ret = p.check(x_, df_);
		assert(TR_SUCCESS == ret);

		int rci = 0;
		ret = p.solver(x_, df_, rci);
		assert(TR_SUCCESS == ret);

		int iter = 0, st_cr = 0;
		double r1 = 0, r2 = 0;
		ret = p.get(iter, st_cr, r1, r2);
		assert(TR_SUCCESS == ret);
	}

	return ret;
}
int test_fms_trnlsp_ = test_fms_trnlsp();

int test_fms_trnlspbc()
{
	int ret = 0;
	{
		double x[2] = { -1, 1 };
		double l[2] = { 0, 0 };
		double u[2] = { 10, 10 };
		trnslpbc p(2, 2, x, l, u);
		p.f = [](int, int, const double* x, double* fx, void*) {
			fx[0] = x[0] + 2 * x[1];
			fx[1] = 3 * x[0] + 4 * x[1];
		};
		p.df = [](int, int, const double*, double* dfx, void*) {
			dfx[0] = 1; // df[0]/dx[0]
			dfx[1] = 3; // df[1]/dx[0]
			dfx[2] = 2; // df[0]/dx[1]
			dfx[3] = 4; // df[1]/dx[1]
		};
		ret = p.init();
		assert(TR_SUCCESS == ret);

		double x_[2] = { 0, 0 };
		double df_[4] = { 0, 0, 0, 0 };
		int two = 2;
		double dx = .0001;
		djacobi((USRFCND)p.f, &two, &two, df_, x_, &dx);
		ret = p.check(x_, df_);
		assert(TR_SUCCESS == ret);

		int rci = 0;
		ret = p.solver(x_, df_, rci);
		assert(TR_SUCCESS == ret);

		int iter = 0, st_cr = 0;
		double r1 = 0, r2 = 0;
		ret = p.get(iter, st_cr, r1, r2);
		assert(TR_SUCCESS == ret);
	}

	return ret;
}
int test_fms_trnlspbc_ = test_fms_trnlspbc();
