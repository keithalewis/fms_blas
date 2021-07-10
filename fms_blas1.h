// fms_blas1.h - BLAS level 1
#pragma once
#include "fms_blas_vector.h"

namespace blas {

	//
// BLAS level 1
//

// arg max |x_i|
	template<class T>
	inline auto iamax(const vector<T>& x)
	{
		if constexpr (std::is_same_v <T, float>) {
			return cblas_isamax(x.size(), x.data(), x.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			return cblas_idamax(x.size(), x.data(), x.incr());
		}
	}

	// arg min |x_i|
	template<class T>
	inline auto iamin(const vector<T>& x)
	{
		if constexpr (std::is_same_v <T, float>) {
			return cblas_isamin(x.size(), x.data(), x.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			return cblas_idamin(x.size(), x.data(), x.incr());
		}
	}

	// sum_i |x_i|
	template<class T>
	inline T asum(const vector<T>& x)
	{
		T s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (std::is_same_v <T, float>) {
			s = cblas_sasum(x.size(), x.data(), x.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			s = cblas_dasum(x.size(), x.data(), x.incr());
		}

		return s;
	}

	// y = a x + y
	template<class T>
	inline vector<T> axpy(T a, const vector<T>& x, vector<T> y)
	{
		if constexpr (std::is_same_v <T, float>) {
			cblas_saxpy(x.size(), a, x.data(), x.incr(), y.data(), y.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			cblas_daxpy(x.size(), a, x.data(), x.incr(), y.data(), y.incr());
		}

		return y;
	}

	// x . y
	template<class T>
	inline T dot(const vector<T>& x, const vector<T>& y)
	{
		T s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (std::is_same_v <T, float>) {
			s = cblas_sdot(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			s = cblas_ddot(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}

		return s;
	}

	// sqrt (sum_i v_i^2)
	template<class T>
	inline T nrm2(const vector<T>& x)
	{
		T s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (std::is_same_v <T, float>) {
			s = cblas_snrm2(x.size(), x.data(), x.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			s = cblas_dnrm2(x.size(), x.data(), x.incr());
		}

		return s;
	}

	// x' = c x + y, y' = c y - s x
	template<class T>
	inline void rot(vector<T> x, vector<T> y, T c, T s)
	{
		if constexpr (std::is_same_v <T, float>) {
			cblas_srot(x.size(), x.data(), x.incr(), y.data(), y.incr(), c, s);
		}
		if constexpr (std::is_same_v <T, double>) {
			cblas_drot(x.size(), x.data(), x.incr(), y.data(), y.incr(), c, s);
		}
	}

	template<class T>
	inline void swap(vector<T> x, vector<T> y)
	{
		if constexpr (std::is_same_v <T, float>) {
			cblas_sswap(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			cblas_dswap(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
	}

	// x = a x
	template<class T>
	inline vector<T> scal(T a, vector<T> x)
	{
		if (a != T(1) and x.size() > 0) {
			if constexpr (std::is_same_v <T, float>) {
				cblas_sscal(x.size(), a, x.data(), x.incr());
			}
			if constexpr (std::is_same_v <T, double>) {
				cblas_dscal(x.size(), a, x.data(), x.incr());
			}
		}

		return x;
	}

#ifdef _DEBUG

	template<class T>
	inline int blas1_test()
	{
		{
			T _v[3] = { 1, -2, 3 };
			auto v = vector<T>(_v);

			ensure(2 == iamax<T>(v));
			ensure(0 == iamin<T>(v));
			ensure(6 == asum<T>(v));
			ensure(sqrt(T(1 + 4 + 9)) == nrm2(v));
		}

		return 0;
	}

#endif // _DEBUG


} // namespace blas
