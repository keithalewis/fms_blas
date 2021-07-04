// fms_blas.h - BLAS wrappers
#pragma once
#include <compare>
#include <iterator>

#pragma warning(push)
#pragma warning(disable: 4820)
#include <mkl_cblas.h>
#pragma warning(pop)

namespace blas {

	// row (n > 0) or column (n < 0) vector
	template<typename X>
	class vector {
		int n;
		X* v;
	public:
		using type = typename X;

		class iterator {
			vector<X> v;
			int i; // index
		public:
			using iterator_category = std::iterator_traits<X>::contiguous_iterator_tag;
			using value_type = X;
			using reference = X&;
			using pointer = X*;

			friend class vector<X>;

			iterator(const vector<X>& v, int i = 0)
				: v(v), i(i)
			{ }

			auto begin() const
			{
				return iterator(v, 0);
			}
			auto end() const
			{
				return iterator(v, v.size());
			}

			auto operator<=>(const iterator&) const = default;

			value_type operator*() const
			{
				return v[i];
			}
			reference operator*()
			{
				return v[i];
			}

			iterator& operator++()
			{
				if (i < v.n) {
					++i;
				}

				return *this;
			}
			iterator& operator--()
			{
				if (i > -v.n) {
					--i;
				}

				return *this;
			}

		};

		vector(int n, X* v)
			: n(n), v(v)
		{ }

		int size() const
		{
			return abs(n);
		}
		X* data()
		{
			return v;
		}
		const X* data() const
		{
			return v;
		}

		explicit operator bool() const
		{
			return n != 0;
		}
		auto operator<=>(const vector&) const = default;
		bool equal(const vector& v_) const
		{
			return n == v_.n and std::equal(v, v + v.size(), v_.v);
		}

		// row or column vector
		CBLAS_TRANSPOSE transpose() const
		{
			return n > 0 ? CblasNoTrans : n < 0 ? CblasTrans
				: static_cast<CBLAS_TRANSPOSE>(0);
		}
		vector transpose(const vector& v) const
		{
			return vector(-n, v);
		}


	};

	template<typename X>
	class basic_matrix {
	protected:
		int r, c;
		X* m;
	public:
		using type = X;

		basic_matrix(int r, int c, X* m)
			: r(r), c(c), m(m)
		{ }
		basic_matrix(const basic_matrix&) = default;
		basic_matrix& operator=(const basic_matrix&) = default;
		virtual ~basic_matrix()
		{ }

		auto operator<=>(const basic_matrix&) const = default;

		int rows() const
		{
			return rows_();
		}
		int columns() const
		{
			return columns_();
		}
		int index(int i, int j) const
		{
			return index_(i, j);
		}
		CBLAS_TRANSPOSE transpose() const
		{
			return transpose_();
		}
		basic_matrix& transpose()
		{
			return transpose_(*this);
		}

		int size() const
		{
			return rows() * columns();
		}
		X* data()
		{
			return m;
		}
		const X* data() const
		{
			return m;
		}
		X operator()(int i, int j) const
		{
			return m[index(i, j)];
		}
		X& operator()(int i, int j)
		{
			return m[index(i, j)];
		}
	private:
		int rows_() const = 0;
		int columns_() const = 0;
		int index_(int i, int j) = 0;
		CBLAS_TRANSPOSE transpose_() const = 0;
		basic_matrix& transpose_() = 0;
	};

	// full
	// upper m(i, j) == 0 if i > j, lower m(i, j) == 0 if i > j
	// unit diagonal: m(i,i) == 1
	// packed: triangular, banded, tensor

	template<class X>
	inline bool upper(const basic_matrix<const X>& m, bool strict = false+)
	{
		for (int i = 0; i < m.rows(); ++i) {
			for (int j = 0; j + strict < i; ++j) {
				if (m(i, j) != 0)
					return false;
			}
		}

		return true;
	}
	template<class X>
	inline bool lower(const basic_matrix<const X>& m, bool strict = false + )
	{
		for (int i = 0; i < m.rows(); ++i) {
			for (int j = i + strict; j < m.columns(); ++j) {
				if (m(i, j) != 0)
					return false;
			}
		}

		return true;
	}

	// r, c >= 0 or r,c < 0 for transpose
	template<class X>
	class full_matrix : public basic_matrix<X> {
		using basic_matrix<X>::basic_matrix;
		using basic_matrix<X>::r;
		using basic_matrix<X>::c;		
		using basic_matrix<X>::m;

		full_matrix(int r, int c, X* m)
			: basic_matrix<X>(r, c, m)
		{
			if (r * c < 0) {
				r = 0;
				c = 0;
				m = nullptr;
			}
		}

		int rows_() const override
		{
			return abs(r);
		}
		int columns_() const override
		{
			return abs(c);
		}
		int index_(int i, int j) const override
		{
			if (r < 0) {
				std::swap(i, j);
			}
			// i = xmod(i, rows_())
			// j = xmod(j, columns_();

			return j + i * columns_();
		}
		CBLAS_TRANSPOSE transpose_() const override
		{
			return r > 0 ? CblasNoTrans : r < 0 ? CblasTrans
				: static_cast<CBLAS_TRANSPOSE>(0);
		}
		full_matrix& transpose_() override
		{
			r = -r;
			c = -c;

			return *this;
		}
	};

	//template<class X, class L, class R>
	//inline auto multiply(const L&, const R&);

	template<class X>
	inline auto mm(const full_matrix<X>& a, const full_matrix<X>& b, X* _c)
	{
		// assert (a.columns() == b.rows()
		full_matrix<X> c(a.rows(), b.columns(), _c);
		if constexpr (std::is_same_v<X, double>) {
			cblas_dgemm(CblasRowMajor, a.transpose(), b.transpose(), a.rows(), b.columns().a.columns(), 1, a.data(), a.lda(), b.data(), b.lda(), 0, c.data(), c.lda());
		}

		return c;
	}

}
