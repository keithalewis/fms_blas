// fms_blas_vector_alloc.h - blas::vector using allocator
#pragma once
#include <memory>
#include "fms_blas_vector.h"

namespace blas {

	// vector backed by array
	template<class T, class A = std::allocator<T>>
	struct vector_alloc : public vector<T> {
		A alloc;
		//using alloc = std::allocator_traits<A>;
		using vector<T>::copy;
		using vector<T>::capacity;

		vector_alloc()
			: vector<T>(0, nullptr, 0)
		{ }
		vector_alloc(int n, int dn = 1)
			: vector<T>(n, nullptr, dn)
		{
			vector<T>::v = alloc.allocate(n * abs(dn));
		}
		vector_alloc(int n, const T* v, int dn)
			: vector_alloc<T,A>(n, dn)
		{
			copy(n, v, dn);
		}
		vector_alloc(const vector_alloc& x)
			: vector_alloc<T,A>(x.n, x.dn)
		{
			copy(x);
		}
		vector_alloc& operator=(const vector_alloc& x)
		{
			if (this != &x) {
				alloc.deallocate(vector<T>::v, capacity());
				vector<T>::v = alloc.allocate(x.capacity());
				copy(x);
			}

			return *this;
		}
		vector_alloc(vector_alloc &&) = default;
		vector_alloc& operator=(vector_alloc &&) = default;
		~vector_alloc()
		{
			alloc.deallocate(vector<T>::v, capacity());
		}

#ifdef _DEBUG
		static int test()
		{
			{
				vector_alloc<T> v(3);
				ensure(v);
				ensure(v.size() == 3);
				ensure(v.incr() == 1);
				ensure(v.data());  
				v.fill(T(0));
				for (auto vi : v) {
					ensure(vi == T(0));
				}

				vector_alloc<T> v2{ v };
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
			{
				vector_alloc<T> v;
				ensure(v.size() == 0);
				ensure(v.data() == nullptr);
				vector_alloc<T> v2{ v };
				ensure(!v2);
				ensure(v2 == v);
				v = v2;
				ensure(!(v != v2));
			}
			{
				vector_alloc<T> v(vector_alloc<T>(3));
				ensure(v.size() == 3);
			}

			return 0;
		}
#endif // _DEBUG
	};

} // namespace blas
