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
	inline int potrf(blas::matrix<X>& a, CBLAS_UPLO uplo = CblasLower)
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
			X _m[4];
			X _a[4];
			auto m = blas::matrix<X>(2, 2, _m);
			auto a = blas::matrix<X>(2, 2, _a);

			m.copy(std::initializer_list<X>({ 1, 2, 0, 1 }));
			a = blas::gemm(m.transpose(), m, a.data()); 
			ensure(a.equal(std::initializer_list<X>({ 1, 2, 2, 5 })));
			auto au = a.uplo<CblasUpper>();
			potrf(au);
			ensure(au.equal(m.uplo<CblasUpper>()));
		}
		{
			X _m[4];
			X _a[4];
			auto m = blas::matrix<X>(2, 2, _m);
			auto a = blas::matrix<X>(2, 2, _a);

			m.copy(std::initializer_list<X>({ 1, 0, 2, 1 }));
			a = blas::gemm(m, m.transpose(), a.data());
			ensure(a.equal(std::initializer_list<X>({ 1, 2, 2, 5 })));
			auto au = a.uplo<CblasLower>();
			potrf(au);
			ensure(au.equal(m.uplo<CblasLower>()));
		}

		return 0;
	}
#endif // _DEBUG


	template<class X>
	inline int potri(blas::matrix<X>& a, CBLAS_UPLO uplo = CblasLower)
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
			X _a[4];
			X _b[4];
			blas::matrix<X> a(2, 2, _a);
			blas::matrix<X> b(2, 2, _b);
			
			a.copy(std::initializer_list<X>({ X(1), X(2), X(2), X(5) }));
			b.copy(a);

			potrf<X>(b, CblasUpper);
			potri<X>(b, CblasUpper);

			/*
			X _c[4];
			blas::matrix<X> c(2, 2, _c);
			c = blas::gemm(a, b, c.data());
			blas::identity_matrix<2, X> id2;
			ensure(c.equal(id2));
			*/
		}

		return 0;
	}
#endif // _DEBUG

} // namespace lapack
