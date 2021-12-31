// fms_blas1.h - BLAS level 1
#pragma once
#include <cmath>
#include "fms_blas_vector.h"

namespace blas {

	//
	// BLAS level 1
	//

#define CBLAS1(X) \
	X(asum) \
	X(axpy) \
	X(copy) \
	X(dot) \
	X(nrm2) \
	X(rot) \

#define CBLAS1_I(X) \
	X(amax) \
	X(amin) \

#define CBLAS_(T, F) static constexpr decltype(cblas_##T##F)* F = cblas_##T##F;
#define CBLAS_I(T, F) static constexpr decltype(cblas_i##T##F)* i##F = cblas_i##T##F;

	template<class T> struct cblas { };

	template<> struct cblas<float> {
#define CBLAS_S(F) CBLAS_(s, F)
		CBLAS1(CBLAS_S)
#undef CBLAS_S
#define CBLAS_IS(F) CBLAS_I(s, F)
			CBLAS1_I(CBLAS_IS)
#undef CBLAS_IS
	};

	template<> struct cblas<double> {
#define CBLAS_D(F) CBLAS_(d, F)
		CBLAS1(CBLAS_D)
#undef CBLAS_D
#define CBLAS_ID(F) CBLAS_I(d, F)
			CBLAS1_I(CBLAS_ID)
#undef CBLAS_IS
	};

	// arg max |x_i|
	template<class T>
	inline auto iamax(const vector<T>& x)
	{
		return cblas<std::remove_cv_t<T>>::iamax(x.size(), x.data(), x.incr());
	}
	template<class T>
	inline T max(const vector<T>& x)
	{
		T m = -std::numeric_limits<T>::max();

		for (int i = 0; i < x.size(); ++i) {
			if (x[i] > m) {
				m = x[i];
			}
		}

		return m;
	}
	// y = max(x, y)
	template<class T>
	inline int max(const vector<T>& x, vector<T>& y)
	{
		int n = 0;

		for (int i = 0; i < x.size() && i < y.size(); ++i) {
			if (x[i] > y[i]) {
				++n;
				y[i] = x[i];
			}
		}

		return n;
	}


	// arg min |x_i|
	template<class T>
	inline auto iamin(const vector<T>& x)
	{
		return cblas<std::remove_cv_t<T>>::iamin(x.size(), x.data(), x.incr());
	}
#ifdef _DEBUG
	template<class T>
	inline int maxmin_test()
	{
		{
			T x[] = { 1,3,2 };
			auto i = iamax(vector(x));
			assert(1 == i);
			assert(3 == x[i]);
			auto j = iamin(vector(x));
			assert(0 == j);
			assert(1 == x[j]);
		}
		{
			const T x[] = { -1,-3,2 };
			auto i = iamax(vector(x));
			assert(1 == i);
			assert(-3 == x[i]);
			auto j = iamin(vector(x));
			assert(0 == j);
			assert(-1 == x[j]);
		}

		return 0;
	}
#endif // _DEBUG
	template<class T>
	inline T min(const vector<T>& x)
	{
		T m = std::numeric_limits<T>::max();

		for (int i = 0; i < x.size(); ++i) {
			if (x[i] < m) {
				m = x[i];
			}
		}

		return m;
	}
	// y = min(x, y)
	template<class T>
	inline int min(const vector<T>& x, vector<T>& y)
	{
		int n = 0;

		for (int i = 0; i < x.size() && i < y.size(); ++i) {
			if (x[i] < y[i]) {
				y[i] = x[i];
			}
		}

		return n;
	}

	// sum_i |x_i|
	template<class T>
	inline T asum(const vector<T>& x)
	{
		return cblas<std::remove_cv_t<T>>::asum(x.size(), x.data(), x.incr());
	}

	// y = a x + y
	template<class T, class U, class V>
	inline auto axpy(T a, const vector<U>& x, vector<V> y)
	{
		cblas<V>::axpy(x.size(), a, x.data(), x.incr(), y.data(), y.incr());

		return y;
	}
#ifdef _DEBUG
	template<class T>
	inline int axpy_test()
	{
		{
			const T a = 2;
			T x[] = { 3,4 };
			T y[] = { 5,6 };
			auto z = axpy(a, vector(x), vector(y));
			assert(2 * 3 + 5 == z[0]);
			assert(2 * 4 + 6 == z[1]);
		}
		{
			const T a = 2;
			T y[] = { 5,6 };
			auto z = axpy(a, vector<const T>({ 3, 4 }), vector(y));
			assert(2 * 3 + 5 == z[0]);
			assert(2 * 4 + 6 == z[1]);
		}

		return 0;
	}
#endif // _DEBUG

	// Copies a x to y.
	// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-copy.html
	template<class T, class U>
	inline void copy(const vector<T>& x, vector<U>& y)
	{
		cblas<std::remove_cv_t<T>>::copy(x.size(), x.data(), x.incr(), y.data(), y.incr());
	}

	// x' y
	template<class T, class U>
	inline auto dot(const vector<T>& x, const vector<U>& y)
	{
		return cblas<std::remove_cv_t<T>>::dot(x.size(), x.data(), x.incr(), y.data(), y.incr());
	}
#ifdef _DEBUG
	template<class T>
	inline int dot_test()
	{
		{
			T x[] = { 1, 2 };
			T y[] = { 3, 4 };
			T xy = dot(vector(x), vector(y));
			assert(1 * 3 + 2 * 4 == xy);
		}
		{
			T xy = dot(vector<const T>({ 1,2 }), vector<const T>({ 3,4 }));
			assert(1 * 3 + 2 * 4 == xy);
		}
		{
			T x[] = { 1, 2 };
			T one = 1;
			T xy = dot(vector(x), vector(2, &one, 0));
			assert(3 == xy);
		}

		return 0;
	}
#endif // _DEBUG

	// sqrt (sum_i v_i^2)
	template<class T>
	inline T nrm2(const vector<T>& x)
	{
		return cblas<T>::nrm2(x.size(), x.data(), x.incr());
	}

	// x' = c x + s y, y' = c y - s x
	template<class T>
	inline void rot(vector<T> x, vector<T> y, T c, T s)
	{
		blas<T>::rot(x.size(), x.data(), x.incr(), y.data(), y.incr(), c, s);
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
		maxmin_test<T>();
		axpy_test<T>();
		dot_test<T>();

		return 0;
	}

#endif // _DEBUG


} // namespace blas
