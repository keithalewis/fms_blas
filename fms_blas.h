// fms_blas.h - BLAS wrappers
#pragma once
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

	// n 1's
	template<class X>
	inline constexpr blas::vector<X> one(int n)
	{
		static X one = 1;

		return blas::vector<X>(n, &one, 0);
	}

	// x . (1, 1, ...)
	template<class X>
	X sum(const blas::vector<X>& x)
	{
		return blas::dot(x, one<X>(x.size()));
	}


	// x . y
	inline double dot(size_t n, const double* x, const double* y, size_t stride = 1)
	{
		double s = 0;

		for (size_t i = 0; i < n; ++i) {
			s += x[i] * y[i * stride];
		}

		return s;
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