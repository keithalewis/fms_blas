// fms_blas_array.h - Backing store for BLAS vectors and matrices
#pragma once
#include "fms_blas.h"

namespace blas {

	// vector backed by array
	template<class T>
	class vector_array : public vector<T> {
		std::valarray<T> _v;
	public:
		using vector<T>::size;
		using vector<T>::incr;
		using vector<T>::copy;

		vector_array(int n, int dn = 1)
			: vector<T>(n, nullptr, dn), _v(n* abs(dn))
		{
			vector<T>::v = &_v[0];
		}
		vector_array(const vector_array& x)
			: vector<T>(x.size(), nullptr, x.incr()), _v(x.size()* abs(x.incr()))
		{
			vector<T>::v = &_v[0];
			copy(x);
		}
		vector_array& operator=(const vector_array& x)
		{
			if (this != &x) {
				_v.resize(x.size() * abs(x.incr()));
				copy(x);
			}

			return *this;
		}
		// vector_array(vector_array &&)
		// vector_array& operator=(vector_array &&)
		~vector_array()
		{ }

#ifdef _DEBUG
		static int test()
		{
			{
				vector_array<T> v(3);
				ensure(v);
				ensure(v.size() == 3);
				ensure(v.incr() == 1);
				ensure(v.data());
				for (auto vi : v) {
					ensure(vi == T(0));
				}

				vector_array<T> v2{ v };
				ensure(v2);
				ensure(v2.size() == 3);
				ensure(v2.incr() == 1);
				ensure(v2.data());
				ensure(v != v2);
				ensure(v.equal(v2));

				v = v2;
				ensure(!(v == v2)); // different pointers
				ensure(v.equal(v2));

				v[1] = T(4);
				ensure(!v.equal(v2));
			}

			return 0;
		}
#endif // _DEBUG
	};


} // namespace blas
