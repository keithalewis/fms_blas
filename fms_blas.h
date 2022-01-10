// fms_blas.h - BLAS wrappers
#pragma once
#include <algorithm>
#include <type_traits>
#pragma warning(disable: 26812)
#include "fms_blas_pack.h"
#include "fms_blas_vector_alloc.h"
#include "fms_blas_matrix_alloc.h"
#include "fms_blas3.h"

#define INTEL_ONEMKL "https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/"
#define ONEMKL_CBLAS "top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/"

//                    https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/
//                    top/blas-and-sparse-blas-routines/blas-routines.html
// https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-rot.html

#define INTEL_CBLAS(x) INTEL_ONEMKL ONEMKL_CBLAS "cblas-" x ".html"

namespace blas {

	template<class T>
	constexpr bool is_float = std::is_same_v<float, std::remove_cv_t<T>>;
	template<class T>
	constexpr bool is_double = std::is_same_v<double, std::remove_cv_t<T>>;

	template<class X, X c>
	class constant : public blas::vector<X> {
		X c_;
	public:
		constant(int n)
			: blas::vector<X>(n, nullptr, 0), c_(c)
		{
			blas::vector<X>::v = &c_;
		}
	};

	// n 0's
	template<class X>
	inline constexpr blas::vector<X> zero(int n)
	{
		return constant<X, 0>(n);
	}

	// n 1's
	template<class X>
	inline constexpr blas::vector<X> one(int n)
	{
		return constant<X, 1>(n);
	}

#ifdef _DEBUG
	template<class X>
	inline int constant_test()
	{
		{
			constant<X, 2> two(3);
			assert(3 == two.size());
			assert(2 == two[0]);
			assert(2 == two[1]);
			assert(2 == two[2]);
		}

		return 0;
	}
#endif // _DEBUG

	template<class F, class X>
	inline void apply(const F& f, const vector<X>& v, vector<X>& w)
	{
		std::transform(v.begin(), v.end(), w.begin(), f);
	}
	template<class F, class X>
	inline void apply(const F& f, vector<X>& v)
	{
		apply(f, v, v);
	}

	// e_j = (0,...,1_j,...0)
	template<class X>
	inline blas::vector_alloc<double> e(int j, int n)
	{
		blas::vector_alloc<double> e_j(n);

		e_j.fill(0);
		e_j[j] = 1;

		return e_j;
	}

	// x . (1, 1, ...)
	template<class X>
	X sum(const blas::vector<X>& x)
	{
		return blas::dot(x, one<X>(x.size()));
	}

	// x' A x
	template<class X, class Y>
	X quad(CBLAS_UPLO uplo, const blas::matrix<X>& A, const blas::vector<Y>& x)
	{
		int n = x.size();
		if (n != A.rows() || n != A.columns()) {
			return std::numeric_limits<X>::quiet_NaN();
		}

		std::remove_const_t<X> s = 0;

		for (int i = 0; i < n; ++i) {
			s += A(i, i) * x[i] * x[i];
			if (uplo == CblasUpper) {
				for (int j = i; j < n; ++j) {
					s += 2 * A(i, j) * x[i] * x[j];
				}
			}
			else if (uplo == CblasLower) {
				for (int j = 0; j <= i; ++j) {
					s += 2 * A(i, j) * x[i] * x[j];
				}
			}
			else {
				return std::numeric_limits<X>::quiet_NaN();
			}
		}

		return s;
	}

}