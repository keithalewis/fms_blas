// fms_blas_vector_array.h - Backing store for BLAS vectors
#pragma once
#include <valarray>
#include "fms_blas_vector.h"

namespace blas {

	// vector backed by array
	template<class T>
	class vector_array : public vector<T> {
		std::valarray<T> _v;
	public:
		using vector<T>::copy;

		vector_array(int n, int dn = 1)
			: vector<T>(n, nullptr, dn), _v(n* abs(dn))
		{
			vector<T>::v = &_v[0];
		}
		vector_array(const vector_array& x)
			: vector_array<T>(x.n, x.dn)
		{
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

	// matrix backed by array
	template<class T>
	class matrix_array : public matrix<T> {
		std::valarray<T> _a;
	public:
		using matrix<T>::copy;

		matrix_array(int r, int c, CBLAS_TRANSPOSE trans = CblasNoTrans)
			: matrix<T>(r, c, nullptr, trans), _a(r * c)
		{
			matrix<T>::a = &_a[0];
		}
		matrix_array(const matrix_array& x)
			: matrix_array(x.r, x.c, x.t)
		{
			copy(x);
		}
		matrix_array& operator=(const matrix_array& x)
		{
			if (this != &x) {
				_a.resize(x.size());
				copy(x);
			}

			return *this;
		}
		// matrix_array(matrix_array &&)
		// matrix_array& operator=(matrix_array &&)
		~matrix_array()
		{ }

#ifdef _DEBUG
		static int test()
		{
			{
				matrix_array<T> m(2,3);
				ensure(m);
				ensure(m.rows() == 2);
				ensure(m.columns() == 3);
				ensure(m.size() == 6);
				ensure(m.trans() == CblasNoTrans);
				ensure(m.data());

				matrix_array<T> m2{ m };
				ensure(m2);
				ensure(m2.rows() == 2);
				ensure(m2.columns() == 3);
				ensure(m2.size() == 6);
				ensure(m2.trans() == CblasNoTrans);
				ensure(m2.data());
				ensure(m != m2);
				ensure(m.equal(m2));

				m = m2;
				ensure(!(m == m2)); // different pointers
				ensure(m.equal(m2));

				m(0,1) = T(7);
				ensure(!m.equal(m2));
			}

			return 0;
		}
#endif // _DEBUG
	};


} // namespace blas