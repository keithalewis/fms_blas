// fms_trnlsp.h - Nonlinear least squares solvers
#pragma once
#include <stdexcept>
#include <functional>
#include <mkl.h>

namespace fms {

	// Common nonlinear least squares data
	struct trnslp_base {
		// eps stopping criteria
		enum {
			EPS_TRUST_REGION_AREA,
			EPS_NORM_FX,
			EPS_JACOBIAN,
			EPS_TRIAL_STEP,
			EPS_NORM_FX_DIFF,
			EPS_TRIAL_STEP_PRECISION,
		};

		int n;
		int m;
		double* x;
		double eps[6]; // precisions for stop-criteria
		int iter1; // maximum number of iterations
		int iter2; // maximum number of iterations of calculation of trial-step
		double rs; // initial step bound
		int info[6];
		void (*f)(int n, int m, double* x, double* fx, void* data) = 0; // function
		void (*df)(int n, int m, double* x, double* dfx, void* data) = 0; // Jacobian

		trnslp_base(int n, int m, double* x, int iter1 = 1000, int iter2 = 100, double rs = 0.1)
			: n(n), m(m), x(x), iter1(iter1), iter2(iter2), rs(rs) 
		{
			for (int i = 0; i < 6; i++)
			{
				eps[i] = 0.00001;
				info[i] = 0;
			}
		}
		trnslp_base(const trnslp_base&) = delete;
		trnslp_base& operator=(const trnslp_base&) = delete;
		virtual ~trnslp_base()
		{
			mkl_free_buffers();
		}
	};

	// min_{x in R^n} ||f(x)||_2, f:R^n -> R^m
	class trnslp : public trnslp_base {
		_TRNSP_HANDLE_t handle;
	public:
		// eps stopping criteria
		enum {
			EPS_TRUST_REGION_AREA,
			EPS_NORM_FX,
			EPS_JACOBIAN,
			EPS_TRIAL_STEP,
			EPS_NORM_FX_DIFF,
			EPS_TRIAL_STEP_PRECISION,
		};

		trnslp(int n, int m, double* x, int iter1 = 1000, int iter2 = 100, double rs = 0.1)
			: trnslp_base(n, m, x, iter1, iter2, rs), handle(nullptr)
		{
		}
		trnslp(const trnslp&) = delete;
		trnslp& operator=(const trnslp&) = delete;
		~trnslp()
		{
			if (handle) {
				dtrnlsp_delete(&handle);
			}
		}

		int init()
		{
			return dtrnlsp_init(&handle, &n, &m, x, eps, &iter1, &iter2, &rs);
		}

		int check(const double* fjac, const double* fvec)
		{
			int ret = dtrnlsp_check(&handle, &n, &m, fjac, fvec, eps, info);
			
			if (TR_SUCCESS != ret) {
				if (info[0]) {
					switch (info[0]) {
					case 1:
						throw std::runtime_error("handle: not allocated");
					default:
						throw std::runtime_error("handle: unknown exception");
					}
				}
				else if (info[1]) {
					switch (info[1]) {
					case 1:
						throw std::runtime_error("fjac: not allocated");
					case 2:
						throw std::runtime_error("fjac: contains NaN");
					case 3:
						throw std::runtime_error("fjac: contains Inf");
					default:
						throw std::runtime_error("fjac: unknown exception");
					}
				}
				else if (info[2]) {
					switch (info[2]) {
					case 1:
						throw std::runtime_error("fvec: not allocated");
					case 2:
						throw std::runtime_error("fvec: contains NaN");
					case 3:
						throw std::runtime_error("fvec: contains Inf");
					default:
						throw std::runtime_error("fvec: unknown exception");
					}
				}
				else if (info[3]) {
					switch (info[3]) {
					case 1:
						throw std::runtime_error("eps: not allocated");
					case 2:
						throw std::runtime_error("eps: contains NaN");
					case 3:
						throw std::runtime_error("eps: contains Inf");
					case 4:
						throw std::runtime_error("eps: contains value <= 0");
					default:
						throw std::runtime_error("eps: unknown exception");
					}
				}
			}

			return ret;
		}

		int solve(double* fvec, double* fjac, int* rci)
		{
			return dtrnlsp_solve(&handle, fvec, fjac, rci);
		}

		int solver(double* fvec, double* fjac, int& rci, void* data = nullptr)
		{
			int ret;

			while (TR_SUCCESS == (ret = solve(fvec, fjac, &rci))) {
				if (rci < 0) {
					break;
				}
				if (rci == 1) {
					f(n, m, x, fvec, data);
				}
				else if (rci == 2) {
					// return Jacobian
					df(n, m, x, fjac, data);
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

	// min_{x in R^n} ||f(x)||_2, f:R^n -> R^m, l <= x <= u
	class trnslpbc : public trnslp_base {
		_TRNSPBC_HANDLE_t handle;
		const double* l;
		const double* u;
	public:
		trnslpbc(int n, int m, double* x, const double* l, const double* u, int iter1 = 1000, int iter2 = 100, double rs = 0.1)
			: trnslp_base(n, m, x, iter1, iter2, rs), handle(nullptr), l(l), u(u)
		{
		}
		trnslpbc(const trnslpbc&) = delete;
		trnslpbc& operator=(const trnslpbc&) = delete;
		~trnslpbc()
		{
			if (handle) {
				dtrnlspbc_delete(&handle);
			}
		}

		int init()
		{
			return dtrnlspbc_init(&handle, &n, &m, x, l, u, eps, &iter1, &iter2, &rs);
		}

		int check(const double* fjac, const double* fvec)
		{
			int ret = dtrnlspbc_check(&handle, &n, &m, fjac, fvec, l, u, eps, info);

			if (TR_SUCCESS != ret) {
				if (info[0]) {
					switch (info[0]) {
					case 1:
						throw std::runtime_error("handle: not allocated");
					default:
						throw std::runtime_error("handle: unknown exception");
					}
				}
				else if (info[1]) {
					switch (info[1]) {
					case 1:
						throw std::runtime_error("fjac: not allocated");
					case 2:
						throw std::runtime_error("fjac: contains NaN");
					case 3:
						throw std::runtime_error("fjac: contains Inf");
					default:
						throw std::runtime_error("fjac: unknown exception");
					}
				}
				else if (info[2]) {
					switch (info[2]) {
					case 1:
						throw std::runtime_error("fvec: not allocated");
					case 2:
						throw std::runtime_error("fvec: contains NaN");
					case 3:
						throw std::runtime_error("fvec: contains Inf");
					default:
						throw std::runtime_error("fvec: unknown exception");
					}
				}
				else if (info[3]) {
					switch (info[3]) {
					case 1:
						throw std::runtime_error("lower: not allocated");
					case 2:
						throw std::runtime_error("lower: contains NaN");
					case 3:
						throw std::runtime_error("lower: contains Inf");
					case 4:
						throw std::runtime_error("lower: greater than upper");
					default:
						throw std::runtime_error("lower: unknown exception");
					}
				}
				else if (info[4]) {
					switch (info[4]) {
					case 1:
						throw std::runtime_error("upper: not allocated");
					case 2:
						throw std::runtime_error("upper: contains NaN");
					case 3:
						throw std::runtime_error("upper: contains Inf");
					case 4:
						throw std::runtime_error("upper: less than lower");
					default:
						throw std::runtime_error("upper: unknown exception");
					}
				}
				else if (info[5]) {
					switch (info[5]) {
					case 1:
						throw std::runtime_error("eps: not allocated");
					case 2:
						throw std::runtime_error("eps: contains NaN");
					case 3:
						throw std::runtime_error("eps: contains Inf");
					case 4:
						throw std::runtime_error("eps: contains value <= 0");
					default:
						throw std::runtime_error("eps: unknown exception");
					}
				}
			}

			return ret;
		}

		int solve(double* fvec, double* fjac, int* rci)
		{
			return dtrnlspbc_solve(&handle, fvec, fjac, rci);
		}

		int solver(double* fvec, double* fjac, int& rci, void* data = nullptr)
		{
			int ret;

			while (TR_SUCCESS == (ret = solve(fvec, fjac, &rci))) {
				if (rci < 0) {
					break;
				}
				if (rci == 1) {
					f(n, m, x, fvec, data);
				}
				else if (rci == 2) {
					// return Jacobian
					df(n, m, x, fjac, data);
				}
			}

			return ret;
		}

		int get(int& iter, int& st_cr, double& r1, double& r2)
		{
			return dtrnlspbc_get(&handle, &iter, &st_cr, &r1, &r2);
		}
#ifdef _DEBUG
		static int test()
		{
			return 0;
		}
#endif // _DEBUG
	};

} // namespace fms