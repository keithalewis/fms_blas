// fms_blas_vector_alloc.h - blas::vector using allocator
#pragma once
#include <memory>
#include "fms_blas_vector.h"

namespace blas {

	// vector backed by array
	template<class T>
	struct vector_alloc : public vector<std::remove_cv_t<T>> {
		std::allocator<T> alloc;
		//using alloc = std::allocator_traits<A>;
		using vector<T>::copy;
		using vector<T>::capacity;

		vector_alloc()
			: vector<T>(0, nullptr, 0)
		{ }
		vector_alloc(int n_, int dn_ = 1)
			: vector<T>(n_, nullptr, dn_)
		{
			vector<T>::v = alloc.allocate(std::max(1, n_ * abs(dn_)));
		}
		vector_alloc(int n_, const T* v_, int dn_)
			: vector_alloc<T>(n_, dn_)
		{
			if (n_ != 0) {
				copy(n_, v_, dn_);
			}
			else {
				vector<T>::v = nullptr;
			}
		}
		vector_alloc(const vector<T>& x)
			: vector_alloc<T>(x.size(), x.data(), x.incr())
		{
		}
		vector_alloc(const vector_alloc<T>& x)
			: vector_alloc<T>(x.n, x.v, x.dn)
		{
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
		vector_alloc(vector_alloc&& x) noexcept
			: vector<T>(x)
		{
			x.operator=(vector<T>{});
		}
		vector_alloc& operator=(vector_alloc&& x)
		{
			if (this != &x) {
				if (capacity())
					alloc.deallocate(vector<T>::v, capacity());
				vector<T>::n = x.n;
				vector<T>::dn = x.dn;
				vector<T>::v = x.v;
				x.n = x.dn = 0;
				x.v = nullptr;
			}

			return *this;
		}
		~vector_alloc()
		{
			if (capacity())
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
