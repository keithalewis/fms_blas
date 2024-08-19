// fms_lapack.h - LAPACK wrappers
#pragma once
#include <stdexcept>
#include <mkl_lapacke.h>
#include "fms_blas.h"

namespace lapack {

#define LAPACK(X) \
	X(pftrf) \
	X(pftrs) \
	X(potrf) \
	X(potri) \
	X(potrs) \
	X(pptrf) \
	X(pptrs) \
	X(gesv)  \
	X(gglse) \
	X(ggglm) \

#define LAPACK_(T, F) static constexpr decltype(LAPACKE_##T##F)* F = LAPACKE_##T##F;

	template<class T>
	struct lapack {};

	template<>
	struct lapack<float> {
#define LAPACK_S(F) LAPACK_(s, F)
		LAPACK(LAPACK_S)
#undef LAPACK_S
	};

	template<>
	struct lapack<double> {
#define LAPACK_D(F) LAPACK_(d, F)
		LAPACK(LAPACK_D)
#undef LAPACK_D
	};

#undef LAPACK_
#undef LAPACK

	// BLAS to LAPACK conversion
	inline char UpLo(CBLAS_UPLO ul)
	{
		switch (ul) {
		case CblasUpper:
			return 'U';
		case CblasLower:
			return 'L';
		}

		return 0;
	}
	inline char Trans(CBLAS_TRANSPOSE trans)
	{
		switch (trans) {
		case CblasTrans:
			return 'T';
		case CblasNoTrans:
			return 'N';
		case CblasConjTrans:
			return 'C';
		}

		return 0;
	}

	// Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix using the Rectangular Full Packed (RFP) format.
	// A = U'U or LL'
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/matrix-factorization-lapack-computational-routines/pftrf.html#pftrf 
	template<class X>
	inline int pftrf(CBLAS_UPLO uplo, blas::matrix<X>& a)
	{
		return lapack<X>::pftrf(LAPACK_ROW_MAJOR, Trans(a.trans()), UpLo(uplo), a.rows(), a.data());
	}

	// Solves a system of linear equations with a Cholesky-factored symmetric (Hermitian) positive-definite coefficient matrix using the Rectangular Full Packed (RFP) format.
	// Solve A * X = B for X where A is positive definite.
	// The columns of B are the solutions on exit.
	// Before calling this routine, you must call ?pftrf to compute the Cholesky factorization of A.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/solve-sys-of-linear-equations-lapack-computation/pftrs.html
	template<class X>
	inline int pftrs(CBLAS_UPLO uplo, const blas::matrix<X>& a, blas::matrix<X>& b)
	{
		return lapack<X>::pftrs(LAPACK_ROW_MAJOR, Trans(a.trans()), UpLo(uplo), a.rows(), b.ld(), a.data(), a.ld(), b.data(), b.ld());
	}


	// Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix.
	// A = U'U or LL'
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/matrix-factorization-lapack-computational-routines/potrf.html
	template<class X>
	inline int potrf(CBLAS_UPLO uplo, blas::matrix<X>& a)
	{
		return lapack<X>::potrf(LAPACK_ROW_MAJOR, UpLo(uplo), a.rows(), a.data(), a.ld());
	}

	// Computes the inverse of a symmetric (Hermitian) positive-definite matrix using the Cholesky factorization.
	// Before calling this routine, call ?potrf to factorize A.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/matrix-inversion-lapack-computational-routines/potri.html
	template<class X>
	inline int potri(CBLAS_UPLO uplo, blas::matrix<X>& a)
	{
		return lapack<X>::potri(LAPACK_ROW_MAJOR, UpLo(uplo), a.rows(), a.data(), a.ld());
	}

	// Solves a system of linear equations with a Cholesky-factored symmetric (Hermitian) positive-definite coefficient matrix.
	// Solve A * X = B for X where A is positive definite.
	// The columns of B are the solutions on exit.
	// Before calling this routine, you must call ?potrf to compute the Cholesky factorization of A.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/solve-sys-of-linear-equations-lapack-computation/potrs.html
	template<class X, class Y>
	inline int potrs(CBLAS_UPLO uplo, const blas::matrix<X>& a, blas::matrix<Y>& b)
	{
		return lapack<X>::potrs(LAPACK_ROW_MAJOR, UpLo(uplo), a.rows(), b.ld(), a.data(), a.ld(), b.data(), b.ld());
	}
	template<class X, class Y>
	inline int potrs(CBLAS_UPLO uplo, const blas::matrix<X>& a, blas::vector<Y>& b)
	{
		blas::matrix<Y> b_(b.size(), 1, b.data());

		return potrs(uplo, a, b_);
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
			assert(2 == a[1]);
			assert(eq(2, a[2], eps));
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
			assert(eq(1, a[0], eps));
			assert(eq(2, a[1], eps));
			assert(eq(2, a[2], eps));
			assert(eq(3, a[3], eps));

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
	inline int pptrf(blas::tp<X>& a)
	{
		return lapack<X>::pptrf(LAPACK_ROW_MAJOR, UpLo(a.uplo()), a.rows(), a.data());
	}

	// Solves a system of linear equations with a packed Cholesky-factored symmetric (Hermitian) positive-definite coefficient matrix.
	// A X = B
	// The columns of B are the solutions on exit.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-computational-routines/solve-sys-of-linear-equations-lapack-computation/pptrs.html
	template<class T, class U>
	inline int pptrs(const blas::tp<T>& a, blas::matrix<U>& b)
	{
		return lapack<T>::pptrs(LAPACK_ROW_MAJOR, UpLo(a.uplo()), a.rows(), b.ld(), a.data(), b.data(), b.ld());
	}
	template<class T, class U>
	inline int pptrs(const blas::tp<T>& a, blas::vector<U>& b)
	{
		blas::matrix<double> b_(b.size(), 1, b.data());

		return pptrs(a, b_);
	}
#ifdef _DEBUG
	template<class T>
	inline int pptr_test()
	{
		//constexpr T eps = std::numeric_limits<T>::epsilon();
		const auto eq = [](T a, T b, T tol) { return fabs(a - b) <= tol;  };

		{
			T a[] = { 1,0,2,13 };
			T ap[3];
			blas::packl(2, a, ap);
			blas::tp<T> a_(blas::matrix<T>(2, ap), CblasLower, CblasNonUnit);
			assert(0 == pptrf(a_));
			assert(1 == ap[0]);
			assert(2 == ap[1]);
			assert(3 == ap[2]);

			a[1] = a[2];
			blas::tp<T> b(blas::matrix<T>(2, a), CblasLower, CblasNonUnit);
			assert(0 == pptrs(a_, b));
			assert(1 == a[0]);
			assert(0 == a[1]);
			assert(0 == a[2]);
			assert(1 == a[3]);
		}
		{
			T a[] = { 1,2,0,13 };
			T ap[3];
			blas::packu(2, a, ap);
			blas::tp<T> a_(blas::matrix(2, ap), CblasUpper, CblasNonUnit);
			assert(0 == pptrf(a_));
			assert(1 == ap[0]);
			assert(2 == ap[1]);
			assert(3 == ap[2]);

			a[2] = a[1]; 
			blas::tp<T> b(blas::matrix<T>(2, a), CblasUpper, CblasNonUnit);
			assert(0 == pptrs(a_, b));
			assert(1 == a[0]);
			assert(0 == a[1]);
			assert(0 == a[2]);
			assert(1 == a[3]);
		}


		return 0;
	}
#endif // _DEBUG

	// Computes the solution to the system of linear equations with a square coefficient matrix Aand multiple right - hand sides.
	// The routine solves for X the system of linear equations A*X = B, 
	// where A is an n-by-n matrix, the columns of matrix B are individual right-hand sides,
	// and the columns of X are the corresponding solutions.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-driver-routines/gesv.html
	template<class T>
	int gesv(blas::ge<T>& a, blas::ge<T>& b, int* ipiv = nullptr)
	{
		if (!ipiv) {
			ipiv = _alloca(a.rows());
		}
		return lapack<T>::gesv(LAPACK_ROW_MAJOR, a.rows(), b.columns(), a.data(), a.ld(), ipiv, b.data(), b.ld());
	}


	// The routine solves the linear equality-constrained least squares (LSE) problem:
	// minimize || A * x - c ||_2 subject to B * x = d
	// where A is an m x n matrix, B is a p x n matrix, c is a given m vector, and d is a given p vector.
	// It is assumed that p ≤ n ≤ m+p, and rank(B) = p, rank([A;B]) = n.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem/lapack-least-squares-eigenvalue-problem-driver/gen-linear-least-squares-lls-problem-lapack-driver/gglse.html#gglse
	template<class T>
	inline int gglse(blas::ge<T>& a, blas::ge<T>& b, blas::vector<T>& c, blas::vector<T>& d, blas::vector<T>& x)
	{
		return lapack<T>::gglse(LAPACK_ROW_MAJOR, a.rows(), a.columns(), b.columns(), a.data(), a.ld(), b.data(), b.ld(), c.data(), d.data(), x.data());
	}

	// The routine solves a general Gauss-Markov linear model (GLM) problem:
	// minimizex ||y||_2 subject to d = A x + B y
	// where A is an n x m matrix, B is an n x p matrix, and d is a given n vector.
	// It is assumed that m ≤ n ≤ m + p, and rank(A) = m and rank([A;B]) = n.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem/lapack-least-squares-eigenvalue-problem-driver/gen-linear-least-squares-lls-problem-lapack-driver/ggglm.html#ggglm
	template<class T>
	inline int ggglm(blas::ge<T>& a, blas::ge<T>& b, blas::ge<T>& d, blas::vector<T>& x, blas::vector<T>& y)
	{
		return lapack<T>::ggglm(LAPACK_ROW_MAJOR, a.rows(), a.columns(), b.columns(), a.data(), a.ld(), b.data(), b.ld(), d.data(), x.data(), y.data());
	}

} // namespace lapack
