// fms_lapack.h - LAPACK wrappers
#pragma once
#include <mkl_lapacke.h>
#include "fms_blas.h"

namespace lapack {

	// CBLAS to LAPACK conversion
	template<CBLAS_TRANSPOSE T>
	struct cblas_trans { static char const trans;  };
	template<>
	struct cblas_trans<CblasTrans> { static const char trans = 'T'; };
	template<>
	struct cblas_trans<CblasNoTrans> { static const char trans = 'N'; };

	template<CBLAS_UPLO UL>
	struct cblas_uplo { static const char uplo; };
	template<>
	struct cblas_uplo<CblasUpper> { static const char uplo = 'U'; };
	template<>
	struct cblas_uplo<CblasLower> { static const char uplo = 'L'; };

	// a = u'u if upper, a = ll' if lower
	template<class X>
	inline int potrf(CBLAS_UPLO uplo, blas::matrix<X>& a)
	{
		int ret = INT_MAX;

		char ul = uplo == CblasUpper ? 'U' : 'L';

		if constexpr (std::is_same_v<X, float>) {
			ret = LAPACKE_spotrf(LAPACK_ROW_MAJOR, ul, a.rows(), a.data(), a.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			ret = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, ul, a.rows(), a.data(), a.ld());
		}

		return ret;
	}

#ifdef _DEBUG

	template<class X>
	inline int potrf_test()
	{
		{
			X _u[4];
			auto u = blas::matrix<X>(2, 2, _u).copy({ 1, 2, 0, 1 });

			X _a[4];
			auto a = blas::gemm(u.transpose(), u, _a);
			ensure(a.rows() == 2);
			ensure(a.columns() == 2);
			ensure(a.trans() == CblasNoTrans);
			ensure(a.as_vector().equal({ 1, 2, 2, 5 }));

			potrf(CblasUpper, a);
			ensure(a.equal(u, CblasUpper));
		}
		{
			X _l[4];
			auto l = blas::matrix<X>(2, 2, _l).copy({ 1, 0, 2, 1 });

			X _a[4];
			auto a = blas::gemm(l, l.transpose(), _a);
			ensure(a.as_vector().equal({ 1, 2, 2, 5 }));

			potrf(CblasLower, a);
			ensure(a.equal(l, CblasLower));
			ensure(!a.equal(l)); // potrf messes up lower part
		}
		
		return 0;
	}
#endif // _DEBUG


	template<class X>
	inline int potri(CBLAS_UPLO uplo, blas::matrix<X>& a)
	{
		int ret = INT_MAX;

		char ul = uplo == CblasUpper ? 'U' : 'L';

		if constexpr (std::is_same_v<X, float>) {
			ret = LAPACKE_spotri(LAPACK_ROW_MAJOR, ul, a.rows(), a.data(), a.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			ret = LAPACKE_dpotri(LAPACK_ROW_MAJOR, ul, a.rows(), a.data(), a.ld());
		}

		return ret;
	}

#ifdef _DEBUG

	template<class X>
	inline int potri_test()
	{
		{
			// upper
			X _u[4];
			auto u = blas::matrix<X>(2, 2, _u).copy({ 1, 2, 0, 1 });
			
			X _a[4];
			auto a = blas::gemm<X>(u.transpose(), u, _a);
			ensure(a.as_vector().equal({ 1, 2, 2, 5 }));;

			X _b[4];
			auto b = blas::matrix<X>(2, 2, _b).copy(a);
			ensure(b.equal(a));

			/*
			potrf<X>(CblasUpper, a); // prepare
			a(1, 0) = X(0);
			potri<X>(CblasUpper, a); // inverse of cholesky
			blas::trmm(CblasLower, a.transpose(), a);

			X _id[4];
			auto id = blas::gemm(a, b, _id);
			blas::identity_matrix<2, X> id2;
			ensure(id.equal(id2));
			*/
		}

		return 0;
	}
#endif // _DEBUG

	// Solve A * X = B for X where A is positive definite.
	// The columns of B are the solutions on exit.
	// Must have 0 in lower or upper region?
	template<class T>
	inline int potrs(CBLAS_UPLO uplo, const blas::matrix<T>& a, blas::matrix<T>& b)
	{
		char ul = uplo == CblasUpper ? 'U' : 'L';

		if constexpr (std::is_same_v<T, float>) {
			return LAPACKE_spotrs(LAPACK_ROW_MAJOR, ul, a.rows(), b.ld(), 
				a.data(), a.ld(), b.data(), b.ld());
		}
		if constexpr (std::is_same_v<T, double>) {
			return LAPACKE_dpotrs(LAPACK_ROW_MAJOR, ul, a.rows(), b.ld(), 
				a.data(), a.ld(), b.data(), b.ld());
		}
	}

#ifdef _DEBUG

	template<class T>
	inline int potrs_test()
	{
		/*
		{
			T _a[4], _b[4], _x[4], _y[4];
			auto a = blas::matrix<T>(2, 2, _a).copy({ 1, 2, 0, 5 });
			auto b = blas::matrix<T>(2, 2, _b).copy({ 1, 2, 3, 4 });
			auto x = blas::matrix<T>(2, 2, _x).copy(b);
			ensure(x.equal(b));

			potrf<T>(CblasUpper, a);
			potrs<T>(CblasUpper, a, x);
			auto b0 = blas::gemv<T>(a, x.row(0), _y);
			ensure(b0.equal(b.row(0)));
		}
		*/

		return 0;
	}

#endif // _DEBUG

} // namespace lapack
