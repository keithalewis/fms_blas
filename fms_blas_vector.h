// fms_blas_vector.h
#pragma once
//#pragma warning(push)
#pragma warning(disable: 4820)
#pragma warning(disable: 4365)
#include <mkl_cblas.h>
//#pragma warning(pop)
#include <algorithm>
#include <cmath>
#include <compare>
#include <stdexcept>
#include <iterator>
#include <numeric>
#include <type_traits>
#include "ensure.h"

namespace blas {

#pragma warning(push)
#pragma warning(disable: 4724)
	// mod returning in range [0, y) 
	template<typename T>
	//	requires std::is_integral_v<T>
	inline T xmod(T x, T y)
	{
		T z = x % y;

		return z >= 0 ? z : z + y;
	}
#pragma warning(pop)

	static inline const char vector_doc[] = R"xyzyx(
	Non-owning strided view of array of T tailored to CBLAS.
)xyzyx";
	template<typename T>
	class vector {
	protected:
		int n, dn;
		T* v;
	public:
		using iterator_category = std::bidirectional_iterator_tag;
		using value_type = T;
		using reference = T&;
		using pointer = T*;
		using difference_type = ptrdiff_t;

		vector()
			: n(0), v(nullptr), dn(1)
		{ }

		// Allocation and lifetime of T* managed externally.
		vector(int n, T* v, int dn = 1)
			: n(n), v(v), dn(dn)
		{ }

		// T _v[] = {1, ...}; vector<T> v(_v);
		template<size_t N>
		vector(T(&v)[N])
			: n(static_cast<int>(N)), v(v), dn(1)
		{ }

		vector(const vector&) = default;
		vector& operator=(const vector&) = default;
		virtual ~vector()
		{ }

		explicit operator bool() const
		{
			return n != 0;
		}
		// size, pointer, and increment equality
		auto operator<=>(const vector&) const = default;

		int size() const
		{
			return n;
		}
		int incr() const
		{
			return dn;
		}
		pointer data()
		{
			return v;
		}
		const pointer data() const
		{
			return v;
		}

		// cyclic index
		T operator[](int i) const
		{
			return v[xmod(i * dn, n * abs(dn))];
		}
		T& operator[](int i)
		{
			return v[xmod(i * dn, n * abs(dn))];
		}

		// usable in range for
		auto begin()
		{
			return *this;
		}
		const auto begin() const
		{
			return *this;;
		}
		auto end()
		{
			return vector(0, v + n * abs(dn), dn);
		}
		const auto end() const
		{
			return vector(0, v + n * abs(dn), dn);
		}
		value_type operator*() const
		{
			return *v;
		}
		reference operator*()
		{
			return *v;
		}
		vector& operator++()
		{
			ensure(dn > 0);

			if (n) {
				--n;
				v += dn;
			}

			return *this;
		}
		vector& operator++(int)
		{
			auto tmp{ *this };

			operator++();

			return *this;
		}

		// equal size and contents, any incr
		bool equal(int _n, const T* _v, int _dn = 1) const
		{
			if (n != _n)
				return false;

			for (int i = 0; i < n; ++i)
				if (operator[](i) != _v[i * _dn])
					return false;

			return true;
		}
		// v.equal({w0, w1, ...})
		bool equal(const std::initializer_list<T>& w)
		{
			return equal((int)w.size(), w.begin(), 1);
		}
		bool equal(const vector& w) const
		{
			return equal(w.size(), w.data(), w.incr());
		}

		// assign values to data
		vector& copy(int n_, const T* v_, int dn_ = 1)
		{
			ensure(n == n_);

			if constexpr (std::is_same_v<T, float>)
				cblas_scopy(n_, v_, dn_, v, dn);
			if constexpr (std::is_same_v<T, double>)
				cblas_dcopy(n_, v_, dn_, v, dn);

			return *this;
		}
		// auto v = vector<T>(n, _v, dn).copy({w0, ...});
		vector& copy(const std::initializer_list<const T>& w)
		{
			return copy((int)w.size(), w.begin(), 1);
		}
		// auto v = vector<T>(n, _v, dn).copy(w);
		vector& copy(const vector& w)
		{
			return copy(w.size(), w.data(), w.incr());
		}

		// set all values to x
		vector& fill(T x)
		{
			for (int i = 0; i < n; ++i) {
				operator[](i) = x;
			}

			return *this;
		}

		// take from front (i > 0) or back (i < 0)
		vector& take(int i)
		{
			i = std::clamp(i, -n, n);

			if (i >= 0) {
				n = i;
			}
			else {
				n = -i;
				v += n + i * abs(dn);
			}

			return *this;
		}

		// drop from front (i > 0) or back (i < 0)
		vector& drop(int i)
		{
			i = std::clamp(i, -n, n);

			if (i > 0) {
				n -= i;
				v += i * abs(dn);
			}
			else if (i < 0) {
				n += i;
			}

			return *this;
		}

#ifdef _DEBUG
		static int test()
		{
			{
				blas::vector<T> v;

				ensure(!v);
				ensure(v.size() == 0);
				ensure(v.data() == nullptr);

				blas::vector<T> v2{ v };
				ensure(!v2);
				ensure(v == v2);
				ensure(!(v2 != v));

				v = v2;
				ensure(!v);
				ensure(v.equal(v2));
			}
			{
				T _v[3];
				auto v = vector<T>(_v).copy({ 1, 2, 3 });

				T _v2[3];
				vector<T> v2(_v2);
				std::iota(v2.begin(), v2.end(), T(1));
				ensure(v.equal(v2));

				v2.copy(v);
				ensure(v2.equal(v));

				ensure(v);
				ensure(v.size() == 3);
				ensure(v[0] == 1);
				ensure(v[1] == 2);
				ensure(v[2] == 3);
				ensure(v[4] == 2);
				ensure(_v[1] == 2);

				v[1] = T(4);
				ensure(v[1] == T(4));
				_v[1] = T(5);
				ensure(v[1] == T(5));
				ensure(v[v.size() + 1] == v[1]);

				vector<T> w(v);
				ensure(v == w);
			}
			{
				T _v[3];
				auto v = vector<T>(3, _v).copy({ 1,2,3 });
				ensure(v);
				ensure(v.size() == 3);
				ensure(v[0] == 1);
				ensure(v[1] == 2);
				ensure(v[2] == 3);

				ensure(v.equal(3, _v));
				ensure(v.equal({ 1, 2, 3 }));
				ensure(v.equal(v));
			}
			{
				T _v[] = { 1,2,3 };
				vector<T> v(_v);
				v.take(2); // {1, 2}
				ensure(v.size() == 2);
				ensure(v[0] == 1);
				v.drop(-1); // {1}
				ensure(v.size() == 1);
				ensure(v[0] = 1);
				v.drop(10);
				ensure(v.size() == 0);
			}
			{
				T _v[6];
				auto v = vector<T>(3, _v, 2).copy({ 1,2,3 });
				auto vi = v.begin();
				ensure(*vi == T(1));
				++vi;
				ensure(*vi == T(2));
				vi++;
				ensure(*vi == T(3));
				++vi;
				ensure(vi == v.end());
			}
			{
				T _v[6];
				auto v = vector<T>(3, _v, 2).copy({ 1,2,3 });

				T i = 1;
				for (auto vi : v) {
					ensure(vi == i);
					i += 1;
				}
			}

			return 0;
		}
#endif // _DEBUG
	};

	template<class T>
	inline vector<T> take(int i, vector<T> v)
	{
		return v.take(i);
	}
	template<class T>
	inline vector<T> drop(int i, vector<T> v)
	{
		return v.drop(i);
	}

} // namespace blas