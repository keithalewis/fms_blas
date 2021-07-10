// fms_blas2.h - BLAS level 2
#pragma once
#include "fms_blas_matrix.h"
#include "fms_blas1.h"

namespace blas {

	// y = alpha op(A)*x + beta y
	template<class T>
	inline vector<T> gemv(const matrix<T>& a, const vector<T>& x, T* _y, T alpha = T(1), T beta = T(0))
	{
		vector<T> y(a.rows(), _y);

		if constexpr (std::is_same_v<T, float>) {
			cblas_sgemv(CblasRowMajor, a.trans(), a.rows(), a.columns(), alpha, a.data(), a.ld(),
				x.data(), x.incr(), beta, y.data(), y.incr());
		}
		if constexpr (std::is_same_v<T, double>) {
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

} // namespace blas