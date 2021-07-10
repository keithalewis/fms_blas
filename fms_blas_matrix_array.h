// fms_blas_matrix_array.h - matrix backed by valarray
#pragma once
#include "fms_blas_matrix.h"

namespace blas {

	// matrix backed by array
	template<class T>
	class matrix_array : public matrix<T> {
		std::valarray<T> _a;
	public:
		using matrix<T>::copy;

		matrix_array(int r, int c, CBLAS_TRANSPOSE trans = CblasNoTrans)
			: matrix<T>(r, c, nullptr, trans), _a(r* c)
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
				matrix_array<T> m(2, 3);
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

				m(0, 1) = T(7);
				ensure(!m.equal(m2));
			}

			return 0;
		}
#endif // _DEBUG
	};



} // namespace blas