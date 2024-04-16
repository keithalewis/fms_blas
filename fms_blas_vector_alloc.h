// fms_blas_vector_alloc.h - blas::vector using allocator
#pragma once
#include <memory>
#include "fms_blas_vector.h"

namespace blas {

	// vector backed by array
	template<class T>
	struct vector_alloc : public vector<std::remove_cv_t<T>> {
		size_t capacity;
		std::allocator<T> alloc;
		//using alloc = std::allocator_traits<A>;
		using vector<T>::copy;

		vector_alloc()
			: vector<T>(0, nullptr, 0), capacity(0)
		{ }
		vector_alloc(int n_, int dn_ = 1)
			: vector<T>(n_, nullptr, dn_)
		{
			capacity = dn_ == 0 ? 1 : n_ * abs(dn_);
			if (capacity)
				vector<T>::v = alloc.allocate(capacity);
		}
		vector_alloc(int n_, const T* v_, int dn_)
			: vector_alloc<T>(n_, dn_)
		{
			copy(n_, v_, dn_);
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
				if (capacity)
					alloc.deallocate(vector<T>::v, capacity);
				vector<T>::n = x.n;
				vector<T>::dn = x.dn;
				vector<T>::v = alloc.allocate(x.capacity);
				copy(x);
			}

			return *this;
		}
		vector_alloc(vector_alloc&& x) noexcept
			: vector<T>(x)
		{
			vector<T>::n = std::exchange(x.n, 0);
			vector<T>::dn = std::exchange(x.dn, 0);
			vector<T>::v = std::exchange(x.v, nullptr);
		}
		vector_alloc& operator=(vector_alloc&& x)
		{
			if (this != &x) {
				if (capacity)
					alloc.deallocate(vector<T>::v, capacity);
				vector<T>::n = std::exchange(x.n, 0);
				vector<T>::dn = std::exchange(x.dn, 0);
				vector<T>::v = std::exchange(x.v, nullptr);
			}

			return *this;
		}
		~vector_alloc()
		{
			if (capacity)
				alloc.deallocate(vector<T>::v, capacity);
		}

#ifdef _DEBUG
		static int test()
		{
			{
				vector_alloc<T> v(3);
				assert(v);
				assert(v.size() == 3);
				assert(v.incr() == 1);
				assert(v.data());  
				v.fill(T(0));
				for (auto vi : v) {
					assert(vi == T(0));
				}

				vector_alloc<T> v2{ v };
				assert(v2);
				assert(v2.size() == 3);
				assert(v2.incr() == 1);
				assert(v2.data());
				assert(v != v2);
				assert(v.equal(v2));

				v = v2;
				assert(!(v == v2)); // different pointers
				assert(v.equal(v2));

				v[1] = T(4);
				assert(!v.equal(v2));
			}
			{
				vector_alloc<T> v;
				assert(v.size() == 0);
				assert(v.data() == nullptr);
				vector_alloc<T> v2{ v };
				assert(!v2);
				//assert(v2 == v);
				//v = v2;
				//assert(!(v != v2));
			}
			{
				vector_alloc<T> v(vector_alloc<T>(3));
				assert(v.size() == 3);
			}

			return 0;
		}
#endif // _DEBUG
	};

} // namespace blas
