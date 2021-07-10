// fms_blas_vector_iterator.h - BLAS vector iterator
#pragma once
#include "fms_blas_vector.h"

namespace blas {

	template<class T>
	class vector_iterator {
		vector<T> i;
	public:
		using iterator_category = std::bidirectional_iterator_tag;
		using value_type = T;
		using reference = T&;
		using pointer = T*;
		using difference_type = ptrdiff_t;

		auto operator<=>(const vector&) const = default;
		explict operator bool() const
		{
			return i.size() != 0;
		}

		vector_iterator()
		{ }
		vector_iterator(const vector_iterator&) = default;
		vector_iterator& operator=(const vector_iterator&) = default;
		~vector_iterator()
		{ }

		auto begin()
		{
			return *this;
		}
		const auto begin() const
		{
			return *this;
		}
		auto end()
		{
			return vector<T>(0, v + n * abs(dn), dn);
		}
		const auto end() const
		{
			return vector<T>(0, v + n * abs(dn), dn);
		}

		value_type operator*() const
		{
			return *i;
		}
		reference operator*()
		{
			return *i;
		}
		vector_iterator& operator++()
		{
			++i;

			return *this;
		}
		vector_iterator& operator++(int)
		{
			auto tmp{ *this };

			operator++();

			return *this;
		}
		vector_iterator& operator--()
		{
			--i;

			return *this;
		}
		vector_iterator& operator--(int)
		{
			auto tmp{ *this };

			operator--();

			return *this;
		}

	};


#if 0
	template<class X, class I>
	class indirect_vector {
		// operator[](I::type j) { return v[i[j]]; }
	};

	template<class I = int>
	class slice {
		I start, size, stride;
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = I;
		using difference_type = std::ptrdiff_t;
		using pointer = I*;
		using reference = I&;

		slice(I start, I size, I stride = 1)
			: start(start), size(size), stride(stride)
		{ }
		slice(const slice&) = default;
		slice& operator=(const slice&) = default;
		~slice()
		{ }

		explicit operator bool() const
		{
			return size != 0;
		}
		auto operator<=>(const slice&) const = default;

		auto begin() const
		{
			return *this;
		}
		auto end() const
		{
			return slice(start + size * stride, 0, stride);
		}

		value_type operator*() const
		{
			return start;
		}
		slice& operator++()
		{
			if (size) {
				start += stride;
				--size;
			}

			return *this;
		}
		slice operator++(int)
		{
			auto tmp{ *this };

			operator++();

			return tmp;
		}
#ifdef _DEBUG

		template<class X>
		static int test()
		{

			return 0;
		}

#endif // _DEBUG
	};
#endif 0

} // namespace blas
