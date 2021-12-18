// fms_blas2.h - BLAS level 2
#pragma once
#include "fms_blas_matrix.h"
#include "fms_blas1.h"

namespace blas {

#define BLAS_MV(X) \
	X(ge) \
	X(tb) \
	X(tp) \
	X(tr) \
	X(sp) \
	X(sy) \

#define BLAS_MV_(T, F) static constexpr decltype(cblas_##T##F##mv)* F = cblas_##T##F##mv;

	template<class T> struct mv { };

	template<>
	struct mv<float> {
#define BLAS_MVS(F) BLAS_MV_(s, F)
		BLAS_MV(BLAS_MVS)
#undef BLAS_MVS
	};

	template<>
	struct mv<double> {
#define BLAS_MVD(F) BLAS_MV_(d, F)
		BLAS_MV(BLAS_MVD)
#undef BLAS_MVD
	};

#undef BLAS_MV_
#undef BLAS_MV

	// Computes a matrix-vector product using a general matrix.
	// y = alpha op(A)*x + beta y
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-gemv.html
	template<class T>
	inline vector<T> gemv(const matrix<T>& a, const vector<T>& x, vector<T> y, T alpha = T(1), T beta = T(0))
	{
		ensure(a.columns() == x.size());
		ensure(a.rows() == y.size());

		mv<T>::ge(CblasRowMajor, a.trans(), a.r, a.c, alpha, a.data(), a.c, x.data(), x.incr(), beta, y.data(), y.incr());

		return y;
	}
#ifdef _DEBUG
	template<class T>
	inline int gemv_test()
	{
		{
			T a[] = { 1,2,3,
					  4,5,6 };
			T x[] = { 1, 2, 3 };
			T y[2];

			auto y_ = gemv(matrix(2, 3, a), vector(x), vector(y));
			assert(1 * 1 + 2 * 2 + 3 * 3 == y[0]);
			assert(4 * 1 + 5 * 2 + 6 * 3 == y[1]);
		}
		{
			T a[] = { 1,2,3,
					  4,5,6 };
			T x[] = { 1, 2 };
			T y[3];

			auto y_ = gemv(transpose(matrix(2, 3, a)), vector(x), vector(y));
			assert(1 * 1 + 4 * 2 == y[0]);
			assert(2 * 1 + 5 * 2 == y[1]);
			assert(3 * 1 + 6 * 2 == y[2]);
		}
		{
			T a[] = { 1, 2, 3, 4, 5, 6 };
			T x[] = { 1, 2, 3 };

			matrix<T> A(2, 3, a);

			auto y = A * vector(x);
			assert(1 * 1 + 2 * 2 + 3 * 3 == y[0]);
			assert(4 * 1 + 5 * 2 + 6 * 3 == y[1]);

			y = transpose(A) * vector(2, x);
			assert(1 * 1 + 4 * 2 == y[0]);
			assert(2 * 1 + 5 * 2 == y[1]);
			assert(3 * 1 + 6 * 2 == y[2]);
		}

		return 0;
	}
#endif // _DEBUG

	// tbmv!!!

	// Computes a matrix-vector product using a triangular packed matrix.
	// x = op(A) x
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-tpmv.html
	template<class T, class U>
	inline vector<U> tpmv(const tp<T>& a, vector<U> x, CBLAS_DIAG diag = CblasNonUnit)
	{
		ensure(a.rows() == a.columns());
		ensure(a.rows() == x.size());

		mv<T>::tp(CblasRowMajor, a.uplo(), a.trans(), diag, a.rows(), a.data(), x.data(), x.incr());

		return x;
	}
#ifdef _DEBUG
	template<class T>
	inline int tpmv_test()
	{
		{
			T a[] = { 1,
					  2, 3,
					  4, 5, 6 };
			tp<T> A(matrix(3, a), CblasLower, CblasNonUnit);
			T x[] = { 1, 2, 3 };
			tpmv(A, vector(x));
			ensure(1                     == x[0]);
			ensure(2 * 1 + 3 * 2         == x[1]);
			ensure(4 * 1 + 5 * 2 + 6 * 3 == x[2]);
		}
		{
			T a[] = { 1,
					  2, 3,
					  4, 5, 6 };
			tp<T> A(matrix(3, a), CblasLower, CblasNonUnit);
			A = transpose(A);
			T x[] = { 1, 2, 3 };
			tpmv(A, vector(x));
			ensure(1 * 1 + 2 * 2 + 4 * 3 == x[0]);
			ensure(        3 * 2 + 5 * 3 == x[1]);
			ensure(                6 * 3 == x[2]);
		}
		{
			T a[] = { 1, 2, 4,
						 3, 5,
							6 };
			tp<T> A(matrix(3, a), CblasUpper, CblasNonUnit);
			T x[] = { 1, 2, 3 };
			tpmv(A, vector(x));
			ensure(1 * 1 + 2 * 2 + 4 * 3 == x[0]);
			ensure(        3 * 2 + 5 * 3 == x[1]);
			ensure(                6 * 3 == x[2]);
		}
		{
			T a[] = { 1, 2, 4,
						 3, 5,
							6 };
			tp<T> A(matrix(3, a), CblasUpper, CblasNonUnit);
			A = transpose(A);
			T x[] = { 1, 2, 3 };
			tpmv(A, vector(x));
			ensure(1 * 1                 == x[0]);
			ensure(2 * 1 + 3 * 2         == x[1]);
			ensure(4 * 1 + 5 * 2 + 6 * 3 == x[2]);
		}

		return 0;
	}
#endif // _DEBUG


	// Computes a matrix-vector product using a triangular matrix.
	// x = op(A)*x 
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-trmv.html
	template<class T>
	inline vector<T> trmv(const tr<T>& a, vector<T> x)
	{
		ensure(a.rows() == a.columns());
		ensure(a.rows() == x.size());

		mv<T>::tr(CblasRowMajor, a.uplo(), a.trans(), a.diag(), a.rows(), a.data(), a.ld(), x.data(), x.incr());

		return x;
	}

	// Computes a matrix-vector product for a packed symmetric matrix.
	// x = op(A) x
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-spmv.html
	template<class T>
	inline vector<T> spmv(CBLAS_UPLO uplo, const matrix<T>& a, const vector<T>& x, vector<T> y, T alpha = 1, T beta = 0)
	{
		ensure(a.rows() == a.columns());
		ensure(a.rows() == x.size());
		ensure(x.size() == y.size());

		mv<T>::sp(CblasRowMajor, uplo, a.rows(), alpha, a.data(), a.ld(), x.data(), x.incr(), beta, y.data().y.incr());

		return x;
	}

	// Computes a matrix-vector product for a symmetric matrix.
	// y = alpha A x + beta y
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-symv.html
	template<class T>
	inline vector<T> symv(CBLAS_UPLO uplo, const matrix<T>& a, const vector<T>& x, vector<T> y, T alpha = 1, T beta = 0)
	{
		ensure(a.rows() == a.columns());
		ensure(a.rows() == x.size());
		ensure(x.size() == y.size());

		mv<T>::sy(CblasRowMajor, uplo, a.rows(), alpha, a.data(), a.ld(), x.data(), x.incr(), beta, y.data().y.incr());

		return x;
	}

#define BLAS_SV(X) \
	X(tr) \

#define BLAS_SV_(T, F) static constexpr decltype(cblas_##T##F##sv)* F = cblas_##T##F##sv;

	template<class T> struct sv { };

	template<>
	struct sv<float> {
#define BLAS_SVS(F) BLAS_SV_(s, F)
		BLAS_SV(BLAS_SVS)
#undef BLAS_SVS
	};

	template<>
	struct sv<double> {
#define BLAS_SVD(F) BLAS_SV_(d, F)
		BLAS_SV(BLAS_SVD)
#undef BLAS_SVD
	};

#undef BLAS_SV_
#undef BLAS_SV

	// Solves a system of linear equations whose coefficients are in a triangular matrix.
	// Solve op(A) x = b for x where A is triangular and x = b on entry.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-trsv.html
	template<class T>
	inline vector<T> trsv(const tr<T>& a, vector<T> x)
	{
		sv<T>::tr(CblasRowMajor, a.uplo(), a.trans(), a.diag(), a.rows(), a.data(), a.ld(), x.data(), x.incr());

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
			y2 = trmv<T>(tr(a, CblasUpper), y2);
			y2 = trsv<T>(tr(a, CblasUpper), y2);
			ensure(y2.equal(b));

			a.copy({ 1, 0, 2, 3 });
			y1 = gemv<T>(a, b, y1);
			y2.copy(b);
			y2 = trmv<T>(tr(a, CblasLower), y2);
			ensure(y2.equal(y1));
			y2 = trsv<T>(tr(a, CblasLower), y2);
			ensure(y2.equal(b));
		}

		return 0;
	}

#endif // _DEBUG

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


#ifdef _DEBUG

	template<class T>
	inline int blas2_test()
	{
		scal_test<T>();
		gemv_test<T>();
		trsv_test<T>();
		tpmv_test<T>();

		return 0;
	}

#endif // _DEBUG


} // namespace blas

template<class T>
inline blas::vector_alloc<T> operator*(const blas::ge<T>& a, const blas::vector<T>& x)
{
	blas::vector_alloc<T> y(a.rows());

	return blas::gemv(a, x, y);
}