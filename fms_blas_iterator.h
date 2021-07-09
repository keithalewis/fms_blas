// fms_blas_iterator.h - BLAS vector iterator
#pragma once
#include "fms_blas.h"

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

} // namespace blas