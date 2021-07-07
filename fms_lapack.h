// fms_lapack.h - LAPACK wrappers
#pragma once
#include <mkl_lapacke.h>
#include "fms_blas.h"

namespace lapack {

	// BLAS to LAPACK conversion
	template<CBLAS_TRANSPOSE T>
	struct blas_trans { char trans;  };
	template<>
	struct blas_trans<CblasTrans> { char trans = 'T'; };
	template<>
	struct blas_trans<CblasNoTrans> { char trans = 'N'; };

	template<CBLAS_UPLO UL>
	struct blas_uplo { char uplo; };
	template<>
	struct blas_uplo<CblasUpper> { char uplo = 'U'; };
	template<>
	struct blas_uplo<CblasLower> { char uplo = 'L'; };

	// a = u'u if upper, a = ll' if lower
	template<class X, CBLAS_TRANSPOSE TRANS, CBLAS_UPLO UPLO>
	inline int potrf(blas::matrix<X, TRANS, UPLO>& a)
	{
		ensure(a.rows() == a.columns());
		ensure(UPLO != cblas::CblasNoUplo);

		int ret = INT_MAX;

		if constexpr (std::is_same_v<X, float>) {
			ret = LAPACKE_spotrf(LAPACK_ROW_MAJOR, cblas_traits<UPLO>::uplo, a.rows(), a.data(), a.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			ret = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, cblas_traits<UPLO>::uplo, a.rows(), a.data(), a.ld());
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
			auto m = blas::matrix<X>(2, 2, _m).upper();
			auto a = blas::matrix<X>(2, 2, _a).upper();

			m.copy(std::initializer_list<X>({ 1, 2, 0, 1 }));
			a = blas::gemm(blas::matrix<X>::transpose(m), m, a.data()); // [1 2; 2 5]
			potrf<X>(a.upper());
			ensure(a.equal(m));
			/*
			m.assign({ 1, 0, 3, 1 });
			a = blas::gemm(m, blas::matrix<X>::transpose(m), a.data()); // [1 3; 3 10]
			potrf<X>(a.lower());
			ensure(a.equal(m.lower()));
			*/
		}

		return 0;
	}
#endif // _DEBUG


	template<class X, CBLAS_TRANSPOSE TRANS, CBLAS_UPLO UPLO>
	inline int potri(blas::matrix<X,TRANS,UPLO>& a)
	{
		ensure(a.rows() == a.columns());
		ensure(UPLO != cblas::CblasNoUplo);

		int ret = INT_MAX;

		if constexpr (std::is_same_v<X, float>) {
			ret = LAPACKE_spotri(LAPACK_ROW_MAJOR, cblas_traits<UPLO>::uplo, a.rows(), a.data(), a.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			ret = LAPACKE_dpotri(LAPACK_ROW_MAJOR, cblas_traits<UPLO>::uplo, a.rows(), a.data(), a.ld());
		}

		return ret;
	}

#ifdef _DEBUG

	template<class X>
	inline int potri_test()
	{
		/*
		{
			X _a[4];
			X _b[4];

			blas::matrix<X> a(2, 2, _a);
			blas::matrix<X> b(2, 2, _b);
			a.copy(std::initializer_list<X>({ X(1), X(2), X(2), X(5) }));
			b.copy(a);

			potrf<X>(b.upper());
			potri<X>(b);

			X _c[4];
			blas::matrix<X> c(2, 2, _c);
			c = blas::gemm(a, b, c.data());
			ensure(c(0, 0) == 1);
		}
		*/

		return 0;
	}
#endif // _DEBUG

} // namespace lapack
