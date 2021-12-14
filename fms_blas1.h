// fms_blas1.h - BLAS level 1
#pragma once
#include <cmath>
#include "fms_blas_vector.h"

namespace blas {

	//
	// BLAS level 1
	//

	template<class T>
	struct cblas {
		static size_t (*iamax)(int, const T*, int);
		static void (*axpy)(int, T, const T*, int, T*, int);
	};
	template<>
	struct cblas<float> {
		static constexpr size_t (*iamax)(int, const float*, int) = cblas_isamax;
		static constexpr void (*axpy)(int, float, const float*, int, float*, int) = cblas_saxpy;
	};
	template<>
	struct cblas<double> {
		static constexpr size_t (*iamax)(int, const double*, int) = cblas_idamax;
		static constexpr void (*axpy)(int, double, const double*, int, double*, int) = cblas_daxpy;
	};

	// arg max |x_i|
	template<class T>
	inline auto iamax(const vector<T>& x)
	{
		return cblas<std::remove_cv_t<T>>::iamax(x.size(), x.data(), x.incr());
	}
#ifdef _DEBUG
	template<class T>
	inline int iamax_test()
	{
		{
			T x[] = { 1,3,2 };
			auto i = iamax(blas::vector(x));
			assert(1 == i);
			assert(3 == x[i]);
		}
		{
			const T x[] = { 1,-3,2 };
			auto i = iamax(blas::vector(x));
			assert(1 == i);
			assert(-3 == x[i]);
		}

		return 0;
	}
#endif // _DEBUG

	// arg min |x_i|
	template<class T>
	inline auto iamin(const vector<T>& x)
	{
		if constexpr (is_float<T>) {
			return cblas_isamin(x.size(), x.data(), x.incr());
		}
		if constexpr (is_double<T>) {
			return cblas_idamin(x.size(), x.data(), x.incr());
		}
	}

	// sum_i |x_i|
	template<class T>
	inline T asum(const vector<T>& x)
	{
		std::remove_const_t<T> s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (is_float<T>) {
			s = cblas_sasum(x.size(), x.data(), x.incr());
		}
		if constexpr (is_double<T>) {
			s = cblas_dasum(x.size(), x.data(), x.incr());
		}

		return s;
	}

	// y = a x + y
	template<class T, class U>
	inline vector<T> axpy(T a, const vector<U>& x, vector<T> y)
	{
		cblas<T>::axpy(x.size(), a, x.data(), x.incr(), y.data(), y.incr());

		return y;
	}
#ifdef _DEBUG
	template<class T>
	inline int axpy_test()
	{
		{
			const T a = 2;
			const T x[] = { 3,4 };
			T y[] = { 5,6 };
			auto z = axpy(a, blas::vector(x), blas::vector(y));
			assert(2 * 3 + 5 == z[0]);
			assert(2 * 4 + 6 == z[1]);
		}

		return 0;
	}
#endif // _DEBUG

	// x . y
	template<class T, class U>
	inline T dot(const vector<T>& x, const vector<U>& y)
	{
		std::remove_const_t<T> s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (is_float<T>) {
			s = cblas_sdot(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
		if constexpr (is_double<T>) {
			s = cblas_ddot(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}

		return s;
	}

	// sqrt (sum_i v_i^2)
	template<class T>
	inline T nrm2(const vector<T>& x)
	{
		std::remove_const_t<T> s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (is_float<T>) {
			s = cblas_snrm2(x.size(), x.data(), x.incr());
		}
		if constexpr (is_double<T>) {
			s = cblas_dnrm2(x.size(), x.data(), x.incr());
		}

		return s;
	}

	// x' = c x + y, y' = c y - s x
	template<class T>
	inline void rot(vector<T> x, vector<T> y, T c, T s)
	{
		if constexpr (is_float<T>) {
			cblas_srot(x.size(), x.data(), x.incr(), y.data(), y.incr(), c, s);
		}
		if constexpr (is_double<T>) {
			cblas_drot(x.size(), x.data(), x.incr(), y.data(), y.incr(), c, s);
		}
	}
	template<class T>
	inline void rot(vector<T> x, vector<T> y, T theta)
	{
		rot(x, y, cos(theta), sin(theta));
	}


	template<class T>
	inline void swap(vector<T> x, vector<T> y)
	{
		if constexpr (is_float<T>) {
			cblas_sswap(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
		if constexpr (is_double<T>) {
			cblas_dswap(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
	}

	// x = a x
	template<class T>
	inline vector<T> scal(T a, vector<T> x)
	{
		if (a != T(1) and x.size() > 0) {
			if constexpr (is_float<T>) {
				cblas_sscal(x.size(), a, x.data(), x.incr());
			}
			if constexpr (is_double<T>) {
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
