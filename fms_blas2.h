// fms_blas2.h - BLAS level 2
#pragma once
#include "fms_blas_matrix.h"
#include "fms_blas1.h"

namespace blas {

	// y = alpha op(A)*x + beta y
	template<class T, class U, class V>
	inline vector<V> gemv(const matrix<T>& a, const vector<U>& x, vector<V>& y, V alpha = V(1), V beta = V(0))
	{
		if constexpr (is_float<T>) {
			cblas_sgemv(CblasRowMajor, a.trans(), a.rows(), a.columns(), alpha, a.data(), a.ld(),
				x.data(), x.incr(), beta, y.data(), y.incr());
		}
		if constexpr (is_double<T>) {
			cblas_dgemv(CblasRowMajor, a.trans(), a.rows(), a.columns(), alpha, a.data(), a.ld(),
				x.data(), x.incr(), beta, y.data(), y.incr());
		}

		return y;
	}

	// scale rows of m by v
	template<class T>
	inline matrix<T> scal(const vector<T>& v, matrix<T> m)
	{
		for (int i = 0; i < m.rows(); ++i) {
			scal<T>(v[i], m.row(i));
		}

		return m;
	}

#ifdef _DEBUG

	template<class T>
	inline int scal_test()
	{
		T _v[] = { 1,2,3 };
		{
			T _a[6];
			matrix<T> a(2, 3, _a);
			std::iota(a.begin(), a.end(), T(1));

			scal<T>(vector<T>(2, _v), a); // rows
			ensure(a(0, 0) == 1);   ensure(a(0, 1) == 2);   ensure(a(0, 2) == 3);
			ensure(a(1, 0) == 2 * 4); ensure(a(1, 1) == 2 * 5); ensure(a(1, 2) == 2 * 6);
		}
		{
			T _a[6];
			matrix<T> a(2, 3, _a, CblasTrans);
			std::iota(a.begin(), a.end(), T(1));
			// {1 4; 2 5; 3 6}

			scal<T>(vector<T>(3, _v), a); // columns
			ensure(a(0, 0) == 1);   ensure(a(0, 1) == 4);
			ensure(a(1, 0) == 2 * 2); ensure(a(1, 1) == 2 * 5);
			ensure(a(2, 0) == 3 * 3); ensure(a(2, 1) == 3 * 6);
		}

		return 0;
	}

#endif // _DEBUG

	// Computes a matrix-vector product using a triangular packed matrix.
	// x = op(A)*x where A is triangular
	template<class T, class U>
	inline vector<U> trmv(CBLAS_UPLO uplo, const matrix<T>& a, vector<U>& x, CBLAS_DIAG diag = CblasNonUnit)
	{
		ensure(a.rows() == a.columns());
		ensure(a.rows() == x.size());

		if constexpr (is_float<T>) {
			cblas_strmv(CblasRowMajor, uplo, a.trans(), diag, a.rows(), a.data(), a.ld(), x.data(), x.incr());
		}
		if constexpr (is_double<T>) {
			cblas_dtrmv(CblasRowMajor, uplo, a.trans(), diag, a.rows(), a.data(), a.ld(), x.data(), x.incr());
		}

		return x;
	}

	// Solve op(A) x = b for x where A is triangular and x = b on entry.
	template<class T, class U>
	inline vector<U> trsv(CBLAS_UPLO uplo, const matrix<T>& a, vector<U>& x, CBLAS_DIAG diag = CblasNonUnit)
	{
		if constexpr (is_float<T>) {
			cblas_strsv(CblasRowMajor, uplo, a.trans(), diag, a.rows(), a.data(), a.ld(), x.data(), x.incr());
		}
		if constexpr (is_double<T>) {
			cblas_dtrsv(CblasRowMajor, uplo, a.trans(), diag, a.rows(), a.data(), a.ld(), x.data(), x.incr());
		}

		return x;
	}

#ifdef _DEBUG

	template<class T>
	inline int trsv_test()
	{
		{
			T _a[4], _b[2], _x[2];
			auto a = matrix<T>(2, 2, _a);
			auto b = vector<T>(2, _b).copy({ 4,5 });
			auto x = vector(2, _x).copy(b);

			T _y1[2], _y2[2];
			auto y1 = vector<T>(_y1);
			auto y2 = vector<T>(_y2);

			// y = A b
			a.copy({ 1, 2, 0, 3 });
			y1 = gemv<T>(a, b, y1);
			y2.copy(b);
			y2 = trmv<T>(CblasUpper, a, y2);
			y2 = trsv<T>(CblasUpper, a, y2);
			ensure(y2.equal(b));

			a.copy({ 1, 0, 2, 3 });
			y1 = gemv<T>(a, b, y1);
			y2.copy(b);
			y2 = trmv<T>(CblasLower, a, y2);
			ensure(y2.equal(y1));
			y2 = trsv<T>(CblasLower, a, y2);
			ensure(y2.equal(b));
		}

		return 0;
	}

#endif // _DEBUG

	// Computes a matrix-vector product using a triangular packed matrix.
	// x = op(A)*x where A is triangular
	template<class T, class U>
	inline vector<U> tpmv(CBLAS_UPLO uplo, const matrix<T>& a, vector<U>& x, CBLAS_DIAG diag = CblasNonUnit)
	{
		ensure(a.rows() == a.columns());
		ensure(a.rows() == x.size());

		if constexpr (is_float<T>) {
			cblas_stpmv(CblasRowMajor, uplo, a.trans(), diag, a.rows(), a.data(), x.data(), x.incr());
		}
		if constexpr (is_double<T>) {
			cblas_dtpmv(CblasRowMajor, uplo, a.trans(), diag, a.rows(), a.data(), x.data(), x.incr());
		}

		return x;
	}

	// Performs a rank - 1 update of a symmetric matrix.
	// A = alpha x x' + A
	template<class T>
	inline matrix<T> syr(CBLAS_UPLO uplo, T alpha, const vector<T>& x, matrix<T>& a)
	{
		if constexpr (is_float<T>) {
			cblas_ssyr(CblasRowMajor, uplo, x.size(), alpha, x.data(), x.incr(), a.data(), a.ld());
		}
		if constexpr (is_double<T>) {
			cblas_dsyr(CblasRowMajor, uplo, x.size(), alpha, x.data(), x.incr(), a.data(), a.ld());
		}

		return a;
	}	

	// performs the matrix-vector operation
	// y = alpha A x + beta y
	// where A is an n by n symmetric matrix	
	template<class T>
	inline vector<T> symv(CBLAS_UPLO uplo, const matrix<T>& a, const vector<T>& x, blas::vector<T>& y, T alpha = 1, T beta = 0)
	{
		if constexpr (is_float<T>) {
			cblas_ssymv(CblasRowMajor, uplo, x.size(), alpha, a.data(), a.ld(), x.data(), x.incr(), beta, x.data(), x.incr());
		}
		if constexpr (is_double<T>) {
			cblas_dsymv(CblasRowMajor, uplo, x.size(), alpha, a.data(), a.ld(), x.data(), x.incr(), beta, y.data(), y.incr());
		}

		return y;
	}

#ifdef _DEBUG

	template<class T>
	inline int blas2_test()
	{
		scal_test<T>();
		trsv_test<T>();

		return 0;
	}

#endif // _DEBUG


} // namespace blas