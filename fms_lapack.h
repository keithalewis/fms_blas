// fms_lapack.h - LAPACK wrappers
#pragma once
#include <mkl_lapacke.h>
#include "fms_blas.h"

namespace lapack {

	// Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix.
	// A = U'U or LL'
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/matrix-factorization-lapack-computational-routines/potrf.html
	template<class X>
	inline int potrf(CBLAS_UPLO uplo, blas::matrix<X>& a)
	{
		char ul = uplo == CblasUpper ? 'U' : 'L';

		if constexpr (blas::is_float<X>) {
			return LAPACKE_spotrf(LAPACK_ROW_MAJOR, ul, a.rows(), a.data(), a.ld());
		}
		if constexpr (blas::is_double<X>) {
			return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, ul, a.rows(), a.data(), a.ld());
		}
	}

	// Computes the inverse of a symmetric (Hermitian) positive-definite matrix using the Cholesky factorization.
	// Before calling this routine, call ?potrf to factorize A.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/matrix-inversion-lapack-computational-routines/potri.html
	template<class X>
	inline int potri(CBLAS_UPLO uplo, blas::matrix<X>& a)
	{
		char ul = uplo == CblasUpper ? 'U' : 'L';

		if constexpr (blas::is_float<X>) {
			return LAPACKE_spotri(LAPACK_ROW_MAJOR, ul, a.rows(), a.data(), a.ld());
		}
		if constexpr (blas::is_double<X>) {
			return LAPACKE_dpotri(LAPACK_ROW_MAJOR, ul, a.rows(), a.data(), a.ld());
		}
	}

	// Solves a system of linear equations with a Cholesky-factored symmetric (Hermitian) positive-definite coefficient matrix.
	// Solve A * X = B for X where A is positive definite.
	// The columns of B are the solutions on exit.
	// Before calling this routine, you must call ?potrf to compute the Cholesky factorization of A.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/solve-sys-of-linear-equations-lapack-computation/potrs.html
	template<class T, class U>
	inline int potrs(CBLAS_UPLO uplo, const blas::matrix<T>& a, blas::matrix<U>& b)
	{
		char ul = uplo == CblasUpper ? 'U' : 'L';

		if constexpr (blas::is_float<T>) {
			return LAPACKE_spotrs(LAPACK_ROW_MAJOR, ul, a.rows(), b.ld(),
				a.data(), a.ld(), b.data(), b.ld());
		}
		if constexpr (blas::is_double<T>) {
			return LAPACKE_dpotrs(LAPACK_ROW_MAJOR, ul, a.rows(), b.ld(),
				a.data(), a.ld(), b.data(), b.ld());
		}
	}

#ifdef _DEBUG
	template<class T>
	inline int potr_test()
	{
		constexpr T eps = std::numeric_limits<T>::epsilon();
		const auto eq = [](T a, T b, T tol) { return fabs(a - b) <= tol;  };

		{
			T a[] = { 1,2,
				      2,13 };
			blas::matrix<T> a_(2, 2, a);
			potrf(CblasLower, a_);
			assert(1 == a[0]);
			assert(a[1] == a[1]);
			assert(2 == a[2]);
			assert(3 == a[3]);

			//a[1] = 0;
			potri(CblasLower, a_); // a^-1;
			assert(a[1] == a[1]); // upper corner untouched
			a[1] = a[2]; // actual inverse

			T a1[] = { 1, 2, 2, 13 };
			T c[4];
			blas::matrix<T> c_ = blas::gemm(a_, blas::matrix(2, 2, a1), blas::matrix(2, 2, c));
			assert(eq(1, c[0], 3*eps));
			assert(eq(0, c[1], 3*eps));
			assert(eq(0, c[2], eps));
			assert(eq(1, c[3], eps));

			a_.copy(4, a1);
			potrf(CblasLower, a_);
			c_.copy(4, a1);
			assert(0 == potrs(CblasLower, a_, c_));
			assert(eq(1, c[0], 3 * eps));
			assert(eq(0, c[1], 3 * eps));
			assert(eq(0, c[2], eps));
			assert(eq(1, c[3], eps));
		}
		{
			T a[] = { 1,2,
					  2,13 };
			blas::matrix<T> a_(2, 2, a);
			potrf(CblasUpper, a_);
			assert(1 == a[0]);
			assert(2 == a[1]);
			assert(a[2] == a[2]);
			assert(3 == a[3]);

			potri(CblasUpper, a_); // a^-1;
			assert(a[2] == a[2]); // upper corner untouched
			a[2] = a[1]; // actual inverse

			T a1[] = { 1, 2, 2, 13 };
			T c[4];
			blas::matrix<T> c_ = blas::gemm(a_, blas::matrix(2, 2, a1), blas::matrix(2, 2, c));
			assert(eq(1, c[0], 3 * eps));
			assert(eq(0, c[1], 3 * eps));
			assert(eq(0, c[2], eps));
			assert(eq(1, c[3], eps));

			a_.copy(4, a1);
			potrf(CblasUpper, a_);
			c_.copy(4, a1);
			assert(0 == potrs(CblasUpper, a_, c_));
			assert(eq(1, c[0], 3 * eps));
			assert(eq(0, c[1], 3 * eps));
			assert(eq(0, c[2], eps));
			assert(eq(1, c[3], eps));
		}

		return 0;
	}
#endif // _DEBUG

	// Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix using packed storage.
	// A = U' U for real data, if uplo='U'
	// A = L L' for real data, if uplo='L'
	// where L is a lower triangular matrix and U is upper triangular.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/matrix-factorization-lapack-computational-routines/pptrf.html
	template<class X>
	inline int pptrf(CBLAS_UPLO uplo, blas::matrix<X>& a)
	{
		char ul = uplo == CblasUpper ? 'U' : 'L';

		if constexpr (blas::is_float<X>) {
			return LAPACKE_spptrf(LAPACK_ROW_MAJOR, ul, a.rows(), a.data());
		}
		if constexpr (blas::is_double<X>) {
			return LAPACKE_dpptrf(LAPACK_ROW_MAJOR, ul, a.rows(), a.data());
		}
	}

	// Solves a system of linear equations with a packed Cholesky-factored symmetric (Hermitian) positive-definite coefficient matrix.
	// A X = B
	// The columns of B are the solutions on exit.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/solve-sys-of-linear-equations-lapack-computation/pptrs.html
	template<class T, class U>
	inline int pptrs(CBLAS_UPLO uplo, const blas::matrix<T>& a, blas::matrix<U>& b)
	{
		char ul = uplo == CblasUpper ? 'U' : 'L';

		if constexpr (blas::is_float<T>) {
			return LAPACKE_spptrs(LAPACK_ROW_MAJOR, ul, a.rows(), b.ld(),
				a.data(), b.data(), b.ld());
		}
		if constexpr (blas::is_double<T>) {
			return LAPACKE_dpptrs(LAPACK_ROW_MAJOR, ul, a.rows(), b.ld(),
				a.data(), b.data(), b.ld());
		}
	}
#ifdef _DEBUG
	template<class T>
	inline int pptr_test()
	{
		constexpr T eps = std::numeric_limits<T>::epsilon();
		const auto eq = [](T a, T b, T tol) { return fabs(a - b) <= tol;  };

		{
			T a[] = { 1,0,2,13 };
			T ap[3];
			blas::packl(2, a, ap);
			blas::matrix<T> a_(2, 2, ap);
			assert(0 == pptrf(CblasLower, a_));
			assert(1 == ap[0]);
			assert(2 == ap[1]);
			assert(3 == ap[2]);

			a[1] = a[2];
			blas::matrix<T> b(2, 2, a);
			assert(0 == pptrs(CblasLower, a_, b));
			assert(1 == a[0]);
			assert(0 == a[1]);
			assert(0 == a[2]);
			assert(1 == a[3]);
		}
		{
			T a[] = { 1,2,0,13 };
			T ap[3];
			blas::packu(2, a, ap);
			blas::matrix<T> a_(2, 2, ap);
			assert(0 == pptrf(CblasUpper, a_));
			assert(1 == ap[0]);
			assert(2 == ap[1]);
			assert(3 == ap[2]);

			a[2] = a[1]; 
			blas::matrix<T> b(2, 2, a);
			assert(0 == pptrs(CblasUpper, a_, b));
			assert(1 == a[0]);
			assert(0 == a[1]);
			assert(0 == a[2]);
			assert(1 == a[3]);
		}


		return 0;
	}
#endif // _DEBUG


} // namespace lapack
