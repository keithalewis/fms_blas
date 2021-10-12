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

		if constexpr (is_float<T>) {
			cblas_sgemm(CblasRowMajor, a.trans(), b.trans(), m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
		}
		if constexpr (is_double<T>) {
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

	// b = alpha op(a)*b or b = alpha b*op(a) using b in place
	template<class T>
	inline matrix<T>& trmm(CBLAS_SIDE lr, CBLAS_UPLO uplo, const matrix<T>& a, matrix<T>& b, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		if constexpr (is_float<T>) {
			cblas_strmm(CblasRowMajor, lr, uplo, a.trans(), diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());
		}
		if constexpr (is_double<T>) {
			cblas_dtrmm(CblasRowMajor, lr, uplo, a.trans(), diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());
		}

		return b;
	}
	template<class T>
	inline matrix<T>& trmm(CBLAS_UPLO uplo, const matrix<T>& a, matrix<T>& b, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		return trmm(CblasLeft, uplo, a, b, alpha, diag);
	}
	template<class T>
	inline matrix<T>& trmm(matrix<T>& b, CBLAS_UPLO uplo, const matrix<T>& a, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		return trmm(CblasRight, uplo, a, b, alpha, diag);
	}

	template<class T>
	inline matrix<T>& trmm(const triangular_matrix<T>& a, matrix<T>& b, T alpha = 1)
	{
		return trmm(CblasLeft, a.uplo, a, b, alpha, a.diag);
	}
	template<class T>
	inline matrix<T>& trmm(matrix<T>& b, const triangular_matrix<T>& a, T alpha = 1)
	{
		return trmm(CblasRight, a.uplo, a, b, alpha, a.diag);
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

	// Solve op(A)*X = alpha B or X*op(A) = alpha B for X where A is triangular
	template<class T>
	inline matrix<T>& trsm(CBLAS_SIDE lr, CBLAS_UPLO uplo, const matrix<T>& a, matrix<T>& x, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		if constexpr (is_float<T>) {
			cblas_strsm(CblasRowMajor, lr, uplo, a.trans(), diag,
				x.rows(), x.columns(), alpha, a.data(), a.ld(), x.data(), x.ld());
		}
		if constexpr (is_double<T>) {
			cblas_dtrsm(CblasRowMajor, lr, uplo, a.trans(), diag,
				x.rows(), x.columns(), alpha, a.data(), a.ld(), x.data(), x.ld());
		}

		return x;
	}
	// solve op(A)*X = B
	template<class T>
	inline matrix<T>& trsm(CBLAS_UPLO uplo, const matrix<T>& a, matrix<T>& x, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		return trsm(CblasLeft, uplo, a, x, alpha, diag);
	}
	// solve X*op(A) = B
	template<class T>
	inline matrix<T>& trsm(matrix<T>& x, CBLAS_UPLO uplo, const matrix<T>& a, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		return trsm(CblasRight, uplo, a, x, alpha, diag);
	}
	template<class T>
	inline matrix<T>& trsm(const triangular_matrix<T>& a, matrix<T>& x, T alpha = 1)
	{
		return trsm(CblasLeft, a.uplo, a, x, alpha, a.diag);
	}
	// solve X*op(A) = B
	template<class T>
	inline matrix<T>& trsm(matrix<T>& x, const triangular_matrix<T>& a, T alpha = 1)
	{
		return trsm(CblasRight, a.uplo, a, x, alpha, a.diag);
	}

#ifdef _DEBUG
	template<class T>
	inline int trsm_test()
	{
		{
			T _a[6], _b[6], _x[6];
			auto a = matrix<T>(2, 3, _a);

			a.copy({ 1,2,3,4,5,6 });
			auto b = matrix<T>(2, 2, _b).copy({ 7,8,9,10 });
			auto x = matrix<T>(2, 2, _x).copy(b);

			trmm(CblasUpper, a, x);
			trsm(CblasUpper, a, x);
			ensure(x.equal(b));

			trmm(CblasLower, a, x);
			trsm(CblasLower, a, x);
			ensure(x.equal(b));
		}

		return 0;
	}
#endif // _DEBUG

	// Performs a Hermitian rank-k update.
	// C = alpha A * op(A) + beta C 
	template<class T>
	inline matrix<T> syrk(CBLAS_UPLO uplo, const matrix<T>& a, T* _c, T alpha = 1, T beta = 0)
	{
		int n = a.rows();
		int k = a.columns();
	
		matrix<T> c(n, n, _c);

		int lda = a.ld();
		int ldc = c.ld();

		if constexpr (std::is_same_v<T, float>) {
			cblas_ssyrk(CblasRowMajor, uplo, a.trans(), n, k, alpha, a.data(), lda, beta, c.data(), ldc);
		}
		if constexpr (std::is_same_v<T, double>) {
			cblas_dsyrk(CblasRowMajor, uplo, a.trans(), n, k, alpha, a.data(), lda, beta, c.data(), ldc);
		}

		return c;
	}

#ifdef _DEBUG
	template<class T>
	inline int blas3_test()
	{
		gemm_test<T>();
		trmm_test<T>();
		trsm_test<T>();

		return 0;
	}
#endif // _DEBUG


} // namespace blas
