// fms_blas_vector.h - strided non-owning BLAS vector
#pragma once
//#pragma warning(push)
#pragma warning(disable: 4820)
#pragma warning(disable: 4365)
#include <mkl_cblas.h>
//#pragma warning(pop)
#include <algorithm>
#include <compare>
#include <iterator>
#include <numeric>
#include <type_traits>
//#include "assert.h"

// 
void xerbla(const char* srname, const int* info, const int);

namespace blas {

	constexpr decltype(cblas_sdot)* sd = cblas_sdot;


	// mod returning in range [0, y) 
	constexpr int xmod(int x, int y) noexcept
	{
		if (y == 0) {
			return 0;
		}

		int z = x % y;

		return z + (z < 0)*y;
	}

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

static inline const char documentation[] = R"xyzyx(
Non-owning strided view of array of T tailored to CBLAS.
)xyzyx";

		vector()
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
		vector(vector&&) = default;
		vector& operator=(vector&&) = default;
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
		pointer data()
		{
			return v;
		}
		const pointer data() const
		{
			return v;
		}
		int incr() const
		{
			return dn;
		}

		// cyclic index
		T operator[](int i) const
		{
			return v[xmod(i * dn, n * dn)];
		}
		T& operator[](int i)
		{
			return v[xmod(i * dn, n * dn)];
		}

		// usable in range for
		const auto begin() const
		{
			return *this;;
		}
		const auto end() const
		{
			return vector(0, v + n, dn);
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
			n -= dn;
			v += dn;

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
			for (int i_ = 0; i_ < n_; ++i_) {
				operator[](i_) = v_[i_ * dn_];
			}

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
			return this == &w ? *this : copy(w.size(), w.data(), w.incr());
		}

		// set all values to x
		vector& fill(T x)
		{
			for (int i = 0; i < n; ++i) {
				operator[](i) = x;
			}

			return *this;
		}

		T sum(T t0 = 0) const
		{
			return std::accumulate(begin(), end(), t0);
		}

		// take from front (i > 0) or back (i < 0)
		vector take(int i) const
		{
			i = std::clamp(i, -n, n);

			if (i >= 0) {
				return vector(i, v, dn);
			}
			else {
				return vector(-i, v + n + i * abs(dn));
			}
		}

		// drop from front (i > 0) or back (i < 0)
		vector drop(int i) const
		{
			i = std::clamp(i, -n, n);

			if (i > 0) {
				return vector(n - i, v + i * abs(dn), dn);
			}
			else if (i < 0) {
				return vector(n + i, v, dn);
			}

			return *this;
		}

		// select elements using mask
		vector& mask(const vector<T>& m, const vector<T>& w)
		{
			if (m.size() != w.size()) {
				const int info = -1;
				xerbla("vector<T>::mask", &info, 0);
			}

			int i = 0;
			for (int j = 0; j < m.size(); ++j) {
				if (m[j]) {
					operator[](i) = w[j];
					++i;
				}
			}
			n = i;

			return *this;
		}
		vector& mask(const vector<T>& m)
		{
			return mask(m, *this);
		}

		// distribute elements using mask to vector
		vector& spread(const vector<T>& m, const vector<T>& w)
		{
			if (m.sum() != w.size()) {
				const int info = -1;
				xerbla("vector<T>::mask", &info, 0);
			}


			int j = 0;
			for (int i = 0; i < size(); ++i) {
				if (m[i]) {
					operator[](i) = w[j];
					++j;
				}
				else {
					operator[](i) = 0;
				}
			}

			return *this;
		}

#ifdef _DEBUG
//#include <cassert>
		static int test()
		{
			{
				blas::vector<T> v;

				assert(!v);
				assert(v.size() == 0);
				assert(v.data() == nullptr);

				blas::vector<T> v2{ v };
				assert(!v2);
				assert(v == v2);
				assert(!(v2 != v));

				v = v2;
				assert(!v);
				assert(v.equal(v2));
			}
			{
				T _v[3];
				auto v = vector<T>(_v).copy({ 1, 2, 3 });

				T _v2[3];
				vector<T> v2(_v2);
				std::iota(v2.begin(), v2.end(), T(1));
				assert(v.equal(v2));

				v2.copy(v);
				assert(v2.equal(v));

				assert(v);
				assert(v.size() == 3);
				assert(v[0] == 1);
				assert(v[1] == 2);
				assert(v[2] == 3);
				assert(v[4] == 2);
				assert(_v[1] == 2);

				v[1] = T(4);
				assert(v[1] == T(4));
				_v[1] = T(5);
				assert(v[1] == T(5));
				assert(v[v.size() + 1] == v[1]);

				vector<T> w(v);
				assert(v == w);
			}
			{
				T _v[3];
				auto v = vector<T>(3, _v).copy({ 1,2,3 });
				assert(v);
				assert(v.size() == 3);
				assert(v[0] == 1);
				assert(v[1] == 2);
				assert(v[2] == 3);

				assert(v.equal(3, _v));
				assert(v.equal({ 1, 2, 3 }));
				assert(v.equal(v));
			}
			{
				T _v[] = { 1,2,3 };
				vector<T> v(_v);
				auto vt = v.take(2); // {1, 2}
				assert(vt.size() == 2);
				assert(vt[0] == 1);
				auto vd = vt.drop(-1); // {1}
				assert(vd.size() == 1);
				assert(vd[0] = 1);
				auto vdd = vd.drop(10);
				assert(vdd.size() == 0);
			}
			{
				T _v[6];
				auto v = vector<T>(6, _v, 2).copy({ 1,2,3 });
				auto vi = v.begin();
				assert(*vi == T(1));
				++vi;
				assert(*vi == T(2));
				vi++;
				assert(*vi == T(3));
				++vi;
				assert(vi == v.end());
			}
			{
				T _v[6];
				auto v = vector<T>(6, _v, 2).copy({ 1,2,3 });

				T i = 1;
				for (auto vi : v) {
					assert(vi == i);
					i += 1;
				}
			}
			{
				T v[3] = { 1,2,3 };
				auto v_ = vector(v);
				v_[1] = 4;
				assert(4 == v[1]);
			}
			{
				const T v[3] = { 1,2,3 };
				auto v_ = vector<const T>(v);
				//v_[1] = 4; // read only
				assert(2 == v[1]);
			}
			{
				T v[] = { 1, 2 };
				T m[] = { 1, 0, 1 };
				T w[3];

				vector<T>(w).spread(vector<T>(m), vector<T>(v));
				assert(vector<T>(w).equal({ 1, 0, 2 }));
				vector<T> w_(w);
				w_.mask(vector<T>(m));
				assert(w_.equal(vector<T>(v)));
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