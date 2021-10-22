// fms_blas_matrix_alloc.h - matrix backed by allocator
#pragma once
#include <memory>
#include "fms_blas_matrix.h"

namespace blas {

	// matrix backed by array
	template<class T>
	struct matrix_alloc : public matrix<T> {
		std::allocator<T> alloc;

		using matrix<T>::copy;
		using matrix<T>::size;
		using matrix<T>::as_vector;

		matrix_alloc(int r, int c, CBLAS_TRANSPOSE trans = CblasNoTrans)
			: matrix<T>(r, c, nullptr, trans)
		{
			matrix<T>::a = alloc.allocate(r*c);
		}
		matrix_alloc(int r, int c, const T* _m, CBLAS_TRANSPOSE trans = CblasNoTrans)
			: matrix<T>(r, c, nullptr, trans)
		{
			matrix<T>::a = alloc.allocate(r * c);
			as_vector().copy(r * c, _m);
		}
		matrix_alloc(const matrix_alloc& x)
			: matrix_alloc(x.r, x.c, x.t)
		{
			copy(x);
		}
		matrix_alloc& operator=(const matrix_alloc& x)
		{
			if (this != &x) {
				alloc.deallocate(matrix<T>::a, size());
				matrix<T>::a = alloc.allocate(x.size());
				copy(x);
			}

			return *this;
		}
		matrix_alloc(matrix_alloc&&) = default;
		matrix_alloc& operator=(matrix_alloc&&) = default;
		~matrix_alloc()
		{ 
			alloc.deallocate(matrix<T>::a, size());
		}

#ifdef _DEBUG
		static int test()
		{
			{
				matrix_alloc<T> m(2, 3);
				ensure(m);
				ensure(m.rows() == 2);
				ensure(m.columns() == 3);
				ensure(m.size() == 6);
				ensure(m.trans() == CblasNoTrans);
				ensure(m.data());
				m.as_vector().fill(T(0));

				matrix_alloc<T> m2{ m };
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