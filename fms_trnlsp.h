// fms_trnlsp.h - Nonlinear least squares solvers
#pragma once
#include <functional>
#include <mkl.h>

namespace fms {

	// min_{x\in R^n} ||y - f(x)||_2, f:R^n -> R^m
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
		void (*f)(int n, int m, const double* x, double* fx) = 0; // function
		void (*df)(int n, int m, const double* x, double* dfx) = 0; // Jacobian

		trnslp(int n, int m, double* x, int iter1 = 1000, int iter2 = 100, double rs = 0)
			: handle(nullptr), n(n), m(m), x(x), iter1(iter1), iter2(iter2), rs(rs)		{
			for (int i = 0; i < 6; i++)
			{
				eps[i] = 0.00001;
				info[i] = 0;
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

		// eps stopping criteria
		enum {
			EPS_TRUST_REGION_AREA,
			EPS_NORM_FX,
			EPS_JACOBIAN,
			EPS_TRIAL_STEP,
			EPS_NORM_FX_DIFF,
			EPS_TRIAL_STEP_PRECISION,
		};

		int init()
		{
			return dtrnlsp_init(&handle, &n, &m, x, eps, &iter1, &iter2, &rs);
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

		int solver(double* fvec, double* fjac, int* rci)
		{
			int ret;

			while (TR_SUCCESS == (ret = solve(fvec, fjac, rci))) {
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

			return ret;
		}

		int get(int& iter, int& st_cr, double& r1, double& r2)
		{
			return dtrnlsp_get(&handle, &iter, &st_cr, &r1, &r2);
		}
#ifdef _DEBUG
		static int test()
		{
			return 0;
		}
#endif // _DEBUG
	};

} // namespace fms