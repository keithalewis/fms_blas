// fms_trnlsp.h - Nonlinear least squares solvers
#pragma once
#include <functional>
#include <mkl.h>

namespace fms {

	// min_{x\in R^n} ||y - f(x)||, f:R^n -> R^m
	class trnslp {
		_TRNSP_HANDLE_t handle;
	public:
		int n;
		int m; 
		double* x;
		double eps[6]; // precisions for stop-criteria
		int iter1; // maximum number of iterations
		int iter2; // maximum number of iterations of calculation of trial-step
		double rs; // initial step bound
		int info[6];
		void (*f)(int n, int m, const double* x, double* fx); // function
		void (*df)(int n, int m, const double* x, double* dfx); // Jacobian

		trnslp(int n, int m, double* x, int iter1, int iter2, double rs)
			: handle(nullptr), n(n), m(m), x(x), iter1(iter1), iter2(iter2), rs(rs)
		{
			int ret = dtrnlsp_init(&handle, &n, &m, x, eps, &iter1, &iter2, &rs);
			if (TR_SUCCESS != ret) {
				throw ret;
			}
		}
		trnslp(const trnslp&) = delete;
		trnslp& operator=(const trnslp&) = delete;
		~trnslp()
		{
			if (handle) {
				dtrnlsp_delete(&handle);
				MKL_Free_Buffers();
			}
		}

		int check(const double* fjac, const double* fvec)
		{
			return dtrnlsp_check(&handle, &n, &m, fjac, fvec, eps, info);
		}

		int solve(double* fvec, double* fjac, int* rci)
		{
			return dtrnlsp_solve(&handle, fvec, fjac, rci);
		}

		void set_f(void (*_f)(int n, int m, const double* x, double* fx))
		{
			f = _f;
		}
		void set_df(void (*_df)(int n, int m, const double* x, double* fx))
		{
			df = _df;
		}

		double* solver(double* fvec, double* fjac, int* rci)
		{
			while (TR_SUCCESS == solve(fvec, fjac, rci)) {
				if (*rci < 0) {
					break;
				}
				if (*rci == 1) {
					f(n, m, x, fvec);
				}
				else if (*rci == 2) {
					// return Jacobian
					df(n, m, x, fjac);
				}
			}

			return fvec;
		}
#ifdef _DEBUG
		static int test()
		{
			return 0;
		}
#endif // _DEBUG
	};

} // namespace fms