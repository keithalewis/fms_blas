// fms_blas.h - BLAS wrappers
#pragma once
//#pragma warning(push)
#pragma warning(disable: 4820)
#pragma warning(disable: 4365)
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
//#pragma warning(pop)
#include <algorithm>
#include <cmath>
#include <compare>
#include <stdexcept>
#include <iterator>
#include <type_traits>
#include "ensure.h"

namespace blas {

	// non-owning row (n > 0) or column (n < 0) vector
	template<typename X>
	class vector {
	protected:
		int n;
		X* v;
	public:
		using type = typename X;

		vector(int n = 0, X* v = nullptr)
			: n(n), v(v)
		{ }
		template<size_t N>
		vector(X (&v)[N])
			: n(static_cast<int>(N)), v(v)
		{ }
		vector(const vector&) = default;
		vector& operator=(const vector&) = default;
		~vector()
		{ }

		X* begin()
		{
			return v;
		}
		const X* begin() const
		{
			return v;
		}
		X* end()
		{
			return v + abs(n);
		}
		const X* end() const
		{
			return v + abs(n);
		}

		vector& assign(int _n, X* _v)
		{
			for (int i; i < _n and i < abs(n); ++i) {
				v[i] = _v[i];
			}

			return *this;
		}
		vector& assign(const std::initializer_list<X>& _v)
		{
			return assign(static_cast<int>(_v.size()), _v.begin());
		}
		vector& assign(const vector& _v)
		{
			return assign(_v.size(), _v.data());
		}
		// assign
		template<class I>
		vector& assign(I i)
		{
			for (int j = 0; j < size(); ++j) {
				v[j] = *i;
				++i;
			}

			return *this;
		}

		explicit operator bool() const
		{
			return n != 0;
		}
		auto operator<=>(const vector&) const = default;
		bool equal(const vector& v_) const
		{
			return n == v_.n and n == 0 or std::equal(v, v + abs(n), v_.v);
		}

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
		X operator[](int i) const
		{
			return v[i];
		}
		X& operator[](int i)
		{
			return v[i];
		}

		struct slice {
			using iterator_category = std::forward_iterator_tag;
			using value_type = int;

			int start, size, stride;
			slice(int start = 0, int size = 0, int stride = 0)
				: start(start), size(size), stride(stride)
			{ }
			
			bool operator==(const slice&) const = default;

			auto begin() const
			{
				return *this;
			}
			auto end() const
			{
				return slice(start, 0, stride);
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
		};

		class iota {
			X x;
		public:
			using iterator_category = std::forward_iterator_tag;
			using value_type = X;

			iota(X x = 0)
				: x(x)
			{ }

			iota begin() const
			{
				return *this;
			}
			iota end() const
			{
				return iota(std::numeric_limits<X>::max());
			}

			auto operator<=>(const iota&) const = default;
			explicit operator bool() const
			{
				return true;
			}
			value_type operator*() const
			{
				return x;
			}
			iota& operator++()
			{
				++x;

				return *this;
			}
			iota operator++(int)
			{
				auto tmp{ x };

				++x;

				return tmp;
			}
		};
		vector& assign(const vector<X>::slice& s, const X* v_)
		{
			for (auto i : s) {
				v[i] = *v_++;
			}

			return *this;
		}

#ifdef _DEBUG
		static int test()
		{
			{
				blas::vector<X> v;

				ensure(!v);
				ensure(v.size() == 0);
				ensure(v.data() == nullptr);

				blas::vector<X> v2{ v };
				ensure(!v2);
				ensure(v == v2);
				ensure(!(v2 != v));

				v = v2;
				ensure(!v);
				ensure(v.equal(v2));
			}
			{
				X _v[] = { 1,2,3 };
				blas::vector<X> v(_v);
				ensure(v);
				ensure(v.size() == 3);
				ensure(v[0] == 1);


				// blas::vector<X> v2(&_v[0]);
				blas::vector<X> v2(3, &_v[0]);
				ensure(v2.equal(v));

				v[1] = X(4);
				ensure(v[1] == X(4));
				_v[1] = X(5);
				ensure(v[1] == X(5));
			}

			return 0;
		}
#endif // _DEBUG
	};

	

	inline int index(int i, int j, int ld)
	{
		return i * ld + j;
	}
	inline bool samesign(int i, int j)
	{
		return (i >= 0) ^ (j < 0);
	}

	// non owning matrix
	template<typename X>
	class matrix {
	protected:
		int r, c;
		X* a;
		CBLAS_UPLO ul = static_cast<CBLAS_UPLO>(0);
	public:
		using type = X;

		matrix(int r = 0, int c = 0, X* a = nullptr)
			: r(r), c(c), a(a)
		{ }
		matrix(const matrix&) = default;
		matrix& operator=(const matrix&) = default;
		virtual ~matrix()
		{ }

		explicit operator bool() const
		{
			return r != 0 and c != 0;
		}
		auto operator<=>(const matrix&) const = default;
		bool equal(const matrix& a_) const
		{
			if (rows() != a_.rows() or columns() != a_.columns() or ul != a_.ul)
				return false;

			for (int i = 0; i < rows(); ++i) {
				int jlo = (ul == CblasUpper) ? i : 0;
				int jhi = (ul == CblasLower) ? i : columns();
				for (int j = jlo; j < jhi; ++j) {
					if (operator()(i, j) != a_(i, j))
						return false;
				}
			}
			
			return true;
		}

		matrix& assign(int n, const X* pa)
		{
			for (int i = 0; i < size() and i < n; ++i)
				a[i] = pa[i];

			return *this;
		}
		matrix& assign(const std::initializer_list<X>& x)
		{
			return assign(static_cast<int>(x.size()), x.begin());
		}
		matrix& assign(const matrix& m)
		{
			ensure(size() >= m.size());

			r = m.r;
			c = m.c;

			return assign(m.size(), m.data());
		}

		CBLAS_TRANSPOSE trans() const
		{
			return r > 0 ? CblasNoTrans : r < 0 ? CblasTrans : static_cast<CBLAS_TRANSPOSE>(0);
		}
		CBLAS_UPLO uplo() const
		{
			return ul;
		}

		int index(int i, int j) const
		{
			return trans() == CblasNoTrans ? blas::index(i, j, columns()) : blas::index(j, i, rows());
		}

		X operator()(int i, int j) const
		{
			return a[index(i, j)];
		}
		X& operator()(int i, int j)
		{
			return a[index(i, j)];
		}

		int rows() const
		{
			return abs(r);
		}
		int columns() const
		{
			return abs(c);
		}
		int size() const
		{
			return rows() * columns();
		}
		int ld() const // default leading dimension
		{
			return trans() == CblasNoTrans ? columns() : trans() == CblasTrans ? rows() : 0;
		}
		X* data()
		{
			return a;
		}
		const X* data() const
		{
			return a;
		}
		matrix& transpose()
		{
			r = -r;
			c = -c;
			std::swap(r, c);

			return *this;
		}
		static matrix transpose(matrix m)
		{
			return m.transpose();
		}
		matrix& upper()
		{
			ul = CblasUpper;

			return *this;
		}
		static matrix<X> upper(matrix<X> u)
		{
			return u.upper();
		}
		matrix& lower()
		{
			ul = CblasLower;

			return *this;
		}
		static matrix lower(matrix l)
		{
			return l.lower();
		}
		matrix& full()
		{
			ul = static_cast<CBLAS_UPLO>(0);

			return *this;
		}
		static matrix full(matrix m)
		{
			return m.full();
		}

#ifdef _DEBUG

		static int test()
		{
			{
				matrix<X> a;
				ensure(!a);
				ensure(a.rows() == 0);
				ensure(a.columns() == 0);
				ensure(a.size() == 0);
				ensure(a.data() == nullptr);
				
				auto a2{ a };
				ensure(!a2);
				ensure(a2 == a);
				ensure(a2.rows() == 0);
				ensure(a2.columns() == 0);
				ensure(a2.size() == 0);
				ensure(a2.data() == nullptr);

				a = a2;
				ensure(!a);
				ensure(a.rows() == 0);
				ensure(a.columns() == 0);
				ensure(a.size() == 0);
				ensure(a.data() == nullptr);
				ensure(!(a != a2));
			}
			{
				X _a[] = { 1,2,3,4,5,6 };
				matrix<X> a(2, 3, _a);
				ensure(a.rows() == 2);
				ensure(a.columns() == 3);
				ensure(a.size() == 6);
				ensure(a.trans() == CblasNoTrans);
				ensure(a(0, 0) == 1);
				ensure(a(1, 0) == 4);
				ensure(a(1, 2) == 6);
				a.transpose(); 
				ensure(a.rows() == 3);
				ensure(a.columns() == 2);
				ensure(a(0, 0) == 1);
				ensure(a(0, 1) == 4);
				ensure(a(2, 1) == 6);
				a(0, 1) = 7;
				ensure(a(0, 1) == 7);
				ensure(transpose(a)(1, 0) == 7);
			}

			return 0;
		}

#endif // _DEBUG	
	};


	//
	// BLAS level 1
	//

	//
	// BLAS level 2
	//
	 
	// matrix * vector
	// vector * matrix

	//
	// BLAS level 3
	// 
	// matrix multiplication with preallocated memory in _c
	template<class X>
	inline matrix<X> gemm(const matrix<X>& a, const matrix<X>& b, X* _c, X alpha = 1, X beta = 0)
	{
		matrix<X> c(a.rows(), b.columns(), _c);

		if constexpr (std::is_same_v<X, float>) {
			cblas_sgemm(CblasRowMajor, a.trans(), b.trans(), a.rows(), b.columns(), a.columns(), alpha, a.data(), a.ld(), b.data(), b.ld(), beta, c.data(), c.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			cblas_dgemm(CblasRowMajor, a.trans(), b.trans(), a.rows(), b.columns(), a.columns(), alpha, a.data(), a.ld(), b.data(), b.ld(), beta, c.data(), c.ld());
		}

		return c;
	}

	// b = op(a)*b;
	template<class X>
	inline matrix<X>& trmm(const matrix<X>& a, matrix<X>& b, X alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		ensure(a.rows() == a.columns());
		ensure(a.uplo());
		
		if constexpr (std::is_same_v<X, float>) {
			cblas_strmm(CblasRowMajor, CblasLeft, a.uplo(), a.trans(), diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			cblas_dtrmm(CblasRowMajor, CblasLeft, a.uplo(), a.trans(), diag, 
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());	
		}

		return b;
	}
	// b = b*op(a);
	template<class X>
	inline matrix<X>& trmm(matrix<X>& b, const matrix<X>& a, X alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		ensure(a.rows() == a.columns());
		ensure(a.uplo());

		if constexpr (std::is_same_v<X, double>) {
			cblas_dtrmm(CblasRowMajor, CblasRight, a.uplo(), a.trans(), diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.columns());
		}

		return b;
	}

	template<class X>
	using upper = matrix<X>::upper;

	template<class X>
	inline int test_mm()
	{
		{
			X _i[] = { 1, 0, 0, 1 };
			X _a[] = { 1, 2, 3, 4, 5, 6 };
			X _c[6];
			matrix<X> id(2, 2, _i);
			matrix<X> a(2, 3, _a); // [1 2 3; 4 5 6]

			auto c = blas::gemm<X>(id, a, _c);
			ensure(c.rows() == 2);
			ensure(c.columns() == 3);
			ensure(vector<X>(6, _a).equal(vector<X>(6, _c)));
			
			a.transpose(); // [1 4; 2 5; 3 6]
			c = gemm<X>(a, id, _c);
			ensure(c.rows() == 3);
			ensure(c.columns() == 2);
			ensure(c.equal(a));
		}
		{
			X _i[] = { 1, 2, 3, 1 };
			X _a[6];
			const matrix<X> i(2, 2, _i); // [1 2; 3 1]
			matrix<X> a(2, 3, _a);

			// [1 2  * [1 2 3
			//  . 1] *  4 5 6]
			// = [1 + 8, 2 + 10, 3 + 12
			//    4      5       6]
			a.assign({ 1, 2, 3, 4, 5, 6 });
			trmm<X>(matrix<X>::upper(i), a);
			ensure(a.rows() == 2);
			ensure(a.columns() == 3);
			ensure(a(0, 0) == 9);
			ensure(a(0, 1) == 12);
			ensure(a(0, 2) == 15);
			ensure(a(1, 0) == 4);
			ensure(a(1, 1) == 5);
			ensure(a(1, 2) == 6);

			// [1 .  * [1 2 3
			//  3 1] *  4 5 6]
			// = [1      2      3
			//    3 + 4, 6 + 5, 9 + 6]
			a.assign({ 1, 2, 3, 4, 5, 6 });
			trmm<X>(matrix<X>::lower(i), a);
			ensure(a.rows() == 2);
			ensure(a.columns() == 3); 
			ensure(a(0, 0) == 1); ensure(a(0, 1) == 2); ensure(a(0, 2) == 3);
			ensure(a(1, 0) == 7); ensure(a(1, 1) == 11); ensure(a(1, 2) == 15);

			// [1 2    [1 2
			//  3 4  *  . 1]
			//  5 6]
			// = [1, 2 + 2
			//    3, 6 + 4
			//    5, 10 + 6]
			a = matrix<X>(3, 2, _a);
			a.assign({ 1, 2, 3, 4, 5, 6 });
			trmm<X>(a, matrix<X>::upper(i));
			ensure(a.rows() == 3);
			ensure(a.columns() == 2);
			ensure(a(0, 0) == 1);
			ensure(a(0, 1) == 4);
			ensure(a(1, 0) == 3);
			ensure(a(1, 1) == 10);
			ensure(a(2, 0) == 5);
			ensure(a(2, 1) == 16);

			// [1 2    [1 .
			//  3 4  *  3 1]
			//  5 6]
			// = [1 + 6,  2
			//    3 + 12, 4
			//    5 + 18, 6]
			a = matrix<X>(3, 2, _a);
			a.assign({ 1, 2, 3, 4, 5, 6 });
			trmm<X>(a, matrix<X>::lower(i));
			ensure(a.rows() == 3);
			ensure(a.columns() == 2);
			ensure(a(0, 0) == 7);
			ensure(a(0, 1) == 2);
			ensure(a(1, 0) == 15);
			ensure(a(1, 1) == 4);
			ensure(a(2, 0) == 23);
			ensure(a(2, 1) == 6);
		}

		return 0;
	}

} // namespace blas

// !!! move to fms_lapack.h
namespace lapack {

	// a = u'u if upper, a = ll' if lower
	template<class X>
	inline int potrf(blas::matrix<X>& a)
	{
		ensure(a.rows() == a.columns());
		
		int ret = INT_MAX;
		CBLAS_UPLO ul = a.uplo();
		ensure(ul);

		if constexpr (std::is_same_v<X, float>) {
			ret = LAPACKE_spotrf(LAPACK_ROW_MAJOR, ul == CblasUpper ? 'U' : 'L', a.rows(), a.data(), a.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			ret = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, ul == CblasUpper ? 'U' : 'L', a.rows(), a.data(), a.ld());
		}

		return ret;
	}

	template<class X>
	inline int potri(blas::matrix<X>& a)
	{
		ensure(a.rows() == a.columns());

		int ret = INT_MAX;
		CBLAS_UPLO ul = a.uplo();
		ensure(ul);

		if constexpr (std::is_same_v<X, float>) {
			ret = LAPACKE_spotri(LAPACK_ROW_MAJOR, ul == CblasUpper ? 'U' : 'L', a.rows(), a.data(), a.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			ret = LAPACKE_dpotri(LAPACK_ROW_MAJOR, ul == CblasUpper ? 'U' : 'L', a.rows(), a.data(), a.ld());
		}

		return ret;
	}

#ifdef _DEBUG
	template<class X>
	inline int potrf_test()
	{
		{
			X _m[4];
			X _a[4];
			blas::matrix<X> m(2, 2, _m);
			blas::matrix<X> a(2, 2, _a);

			m.assign({ 1, 2, 0, 1 });
			a = blas::gemm(blas::matrix<X>::transpose(m), m, a.data()); // [1 2; 2 5]
			potrf<X>(a.upper());
			ensure(a.equal(m.upper()));

			m.assign({ 1, 0, 3, 1 });
			a = blas::gemm(m, blas::matrix<X>::transpose(m), a.data()); // [1 3; 3 10]
			potrf<X>(a.lower());
			ensure(a.equal(m.lower()));
		}
		{
			X _a[4];
			X _b[4];

			blas::matrix<X> a(2, 2, _a);
			blas::matrix<X> b(2, 2, _b);
			a.assign({ X(1), X(2), X(2), X(5) });
			b.assign(a);

			potrf<X>(b.upper());
			potri<X>(b);

			X _c[4];
			blas::matrix<X> c(2, 2, _c);
			c = blas::gemm(a, b, c.data());
			ensure(c(0, 0) == 1);
		}

		return 0;
	}
#endif // _DEBUG

} // namespace lapack

#if 0
template<class X>
class triangular_matrix : public matrix<X> {
public:
	using matrix<X>::c;
	using matrix<X>::columns;
	using matrix<X>::trans;
	using matrix<X>::uplo;

	triangular_matrix(int n, X* a, bool lower = false)
		: matrix<X>(n, lower ? -n : n, a)
	{ }

	int index(int i, int j) const override
	{
		int c_ = c;

		if (trans() == CblasTrans) {
			std::swap(i, j);
			c_ = -c; // up <-> lo
		}

		if (c_ > 0 and i <= j) // upper
			return i + ((2 * columns() - j) * (j + 1)) / 2;
		else if (c_ < 0 and i >= j) // lower
			return i + (j * (j + 1)) / 2;

		return -1;
	}

#ifdef _DEBUG
	static int test()
	{
		{
			X _a[] = { 1,2,3 };
			triangular_matrix<X> t(2, _a);
			ensure(t.rows() == 2);
			ensure(t.columns() == 2);
			ensure(t.trans() == CblasNoTrans);
			ensure(t.uplo() == CblasUpper);
		}

		return 0;
	}
#endif // _DEBUG
};



template<class X>
struct identity_matrix : public matrix<X> {
	static X _a[2] = { 0, 1 };

	identity_matrix(int n)
		: matrix<X>(n, n, _a)
	{ }

	int index_(int i, int j) const override
	{
		return i == j;
	}
};

template<class X>
inline bool upper(const matrix<const X>& a, bool strict = false)
{
	for (int i = 0; i < a.rows(); ++i)
		for (int j = 0; j + strict < i; ++j)
			if (a(i, j) != 0)
				return false;

	return true;
}
template<class X>
inline bool lower(const matrix<const X>& a, bool strict = false)
{
	for (int i = 0; i < a.rows(); ++i)
		for (int j = i + strict; j < a.columns(); ++j)
			if (a(i, j) != 0)
				return false;

	return true;
}
template<class X>
inline bool unit(const matrix<const X>& a)
{
	for (int i = 0; i < a.rows() and i < a.columns(); ++i)
		if (a(i, i) != 1)
			return false;

	return true;
}
template<class X>
inline bool symmetric(const matrix<const X>& a)
{
	for (int i = 0; i < a.rows(); ++i)
		for (int j = 0; j < i; ++j)
			if (a(i, j) != a(j, i))
				return false;

	return true;
}
#endif // 0