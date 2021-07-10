// fms_blas3.h - BLAS level 3
#pragma once
#include "fms_blas2.h"

namespace blas {
	//
	// BLAS level 3
	// 

	// general matrix multiplication using preallocated memory in _c
	// C = alpha op(A) * op(B) + beta C 
	template<class T>
	inline matrix<T> gemm(const matrix<T>& a, const matrix<T>& b, T* _c, T alpha = 1, T beta = 0)
	{
		int m = a.rows();
		int k = a.columns();
		ensure(k == b.rows());
		int n = b.columns();

		matrix<T> c(m, n, _c);

		int lda = a.ld();
		int ldb = b.ld();
		int ldc = c.ld();

		if constexpr (std::is_same_v<T, float>) {
			cblas_sgemm(CblasRowMajor, a.trans(), b.trans(), m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
		}
		if constexpr (std::is_same_v<T, double>) {
			cblas_dgemm(CblasRowMajor, a.trans(), b.trans(), m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
		}

		return c;
	}

#ifdef _DEBUG
	template<class T>
	inline int gemm_test()
	{
		{
			T _a[6];
			matrix<T> a(2, 3, _a); // [1 2 3; 4 5 6]
			std::iota(a.begin(), a.end(), T(1));

			T _c[6];
			matrix<T> c(2, 3, _c);
			identity_matrix<2, T> id2;
			c = gemm<T>(id2, a, c.data());
			ensure(c.equal(a));
			std::fill(c.begin(), c.end(), T(-1));

			identity_matrix<3, T> id3;
			c = gemm<T>(a, id3, c.data());
			ensure(c.equal(a));

			c = gemm<T>(id3, a.transpose(), c.data());
			ensure(c.equal(a.transpose()));

			c = gemm<T>(a.transpose(), id2, c.data());
			ensure(c.equal(a.transpose()));
		}

		return 0;
	}

#endif // _DEBUG

	// b = alpha op(a)*b or b = alpha b*op(a)
	template<class T>
	inline matrix<T>& trmm(CBLAS_SIDE lr, CBLAS_UPLO ul, const matrix<T>& a, matrix<T>& b, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		if constexpr (std::is_same_v<T, float>) {
			cblas_strmm(CblasRowMajor, lr, ul, a.trans(), diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());
		}
		if constexpr (std::is_same_v<T, double>) {
			cblas_dtrmm(CblasRowMajor, lr, ul, a.trans(), diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());
		}

		return b;
	}
	template<class T>
	inline matrix<T>& trmm(CBLAS_UPLO ul, const matrix<T>& a, matrix<T>& b, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		return trmm(CblasLeft, ul, a, b, alpha, diag);
	}
	template<class T>
	inline matrix<T>& trmm(matrix<T>& b, CBLAS_UPLO ul, const matrix<T>& a, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		return trmm(CblasRight, ul, a, b, alpha, diag);
	}

#ifdef _DEBUG

	template<class T>
	inline int trmm_test()
	{
		{
			T _i[] = { 1, 2, 3, 1 };
			const matrix<T> i(2, 2, _i); // [1 2; 3 1]
			T _a[6];
			matrix<T> a(2, 3, _a);
			std::iota(a.begin(), a.end(), T(1));

			// [1 2  * [1 2 3
			//  . 1] *  4 5 6]
			// = [1 + 8, 2 + 10, 3 + 12
			//    4      5       6]
			trmm<T>(CblasLeft, CblasUpper, i, a);
			ensure(a.rows() == 2);
			ensure(a.columns() == 3);
			ensure(a(0, 0) == 9);
			ensure(a(0, 1) == 12);
			ensure(a(0, 2) == 15);
			ensure(a(1, 0) == 4);
			ensure(a(1, 1) == 5);
			ensure(a(1, 2) == 6);
		}
		{
			T _i[] = { 1, 2, 3, 1 };
			const matrix<T> i(2, 2, _i); // [1 2; 3 1]
			T _a[6];
			matrix<T> a(2, 3, _a);
			std::iota(a.begin(), a.end(), T(1));

			// [1 .  * [1 2 3
			//  3 1] *  4 5 6]
			// = [1      2      3
			//    3 + 4, 6 + 5, 9 + 6]
			trmm<T>(CblasLeft, CblasLower, i, a);
			ensure(a.rows() == 2);
			ensure(a.columns() == 3);
			ensure(a(0, 0) == 1); ensure(a(0, 1) == 2); ensure(a(0, 2) == 3);
			ensure(a(1, 0) == 7); ensure(a(1, 1) == 11); ensure(a(1, 2) == 15);
		}
		{
			T _i[] = { 1, 2, 3, 1 };
			const matrix<T> i(2, 2, _i); // [1 2; 3 1]
			T _a[6];
			matrix<T> a(3, 2, _a);
			std::iota(a.begin(), a.end(), T(1));

			// [1 2    [1 2
			//  3 4  *  . 1]
			//  5 6]
			// = [1, 2 + 2
			//    3, 6 + 4
			//    5, 10 + 6]
			trmm<T>(CblasRight, CblasUpper, i, a);
			ensure(a.rows() == 3);
			ensure(a.columns() == 2);
			ensure(a(0, 0) == 1);
			ensure(a(0, 1) == 4);
			ensure(a(1, 0) == 3);
			ensure(a(1, 1) == 10);
			ensure(a(2, 0) == 5);
			ensure(a(2, 1) == 16);
		}
		{
			T _i[] = { 1, 2, 3, 1 };
			const matrix<T> i(2, 2, _i); // [1 2; 3 1]
			T _a[6];
			matrix<T> a(3, 2, _a);
			std::iota(a.begin(), a.end(), T(1));

			// [1 2    [1 .
			//  3 4  *  3 1]
			//  5 6]
			// = [1 + 6,  2
			//    3 + 12, 4
			//    5 + 18, 6]
			trmm<T>(CblasRight, CblasLower, i, a);
			ensure(a.rows() == 3);
			ensure(a.columns() == 2);
			ensure(a(0, 0) == 7);
			ensure(a(0, 1) == 2);
			ensure(a(1, 0) == 15);
			ensure(a(1, 1) == 4);
			ensure(a(2, 0) == 23);
			ensure(a(2, 1) == 6);
		}

		return 0;
	}

#endif // _DEBUG


} // namespace blas
