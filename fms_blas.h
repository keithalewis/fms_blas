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
#include <numeric>
#include <type_traits>
#include "ensure.h"

namespace blas {

#pragma warning(push)
#pragma warning(disable: 4724)
	// mod returning [0, y) 
	template<typename T>
	//	requires std::is_integral_v<T>
	inline T xmod(T x, T y)
	{
		T z = x % y;

		return z >= 0 ? z : z + y;
	}
#pragma warning(pop)

	// non-owning vector
	template<typename X>
	class vector {
	protected:
		std::size_t n;
		X* v;
	public:
		using type = typename X;

		vector(std::size_t n = 0, X* v = nullptr)
			: n(n), v(v)
		{ }
		template<size_t N>
		vector(X(&v)[N])
			: n(N), v(v)
		{ }
		vector(const vector&) = default;
		vector& operator=(const vector&) = default;
		~vector()
		{ }

		explicit operator bool() const
		{
			return n != 0;
		}
		// size and pointer equality
		auto operator<=>(const vector&) const = default;

		std::size_t size() const
		{
			return n;
		}
		X* data()
		{
			return v;
		}
		const X* data() const
		{
			return v;
		}
		// cyclic index
		X operator[](std::size_t i) const
		{
			return v[i % n];
		}
		X& operator[](std::size_t i)
		{
			return v[i % n];
		}

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
			return v + n;
		}
		const X* end() const
		{
			return v + n;
		}

		// unsafe assignment from iterator
		template<class I>
		vector& copy(const I& i)
		{
			std::copy(i.begin(), i.end(), begin());

			return *this;
		}

		// unsafe content equality
		template<class I>
		bool equal(const I& i) const
		{
			return std::equal(i.begin(), i.end(), begin()/*, end()*/);
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
				X _v[3];
				blas::vector<X> v(_v);
				std::iota(v.begin(), v.end(), X(1));
				ensure(v);
				ensure(v.size() == 3);
				ensure(v[0] == 1);
				ensure(v[1] == 2);
				ensure(v[2] == 3);
				ensure(v[4] == 2);
				ensure(_v[1] == 2);

				// blas::vector<X> v2(&_v[0]);
				blas::vector<X> v2(3, &_v[0]);
				ensure(v2.equal(v));

				v[1] = X(4);
				ensure(v[1] == X(4));
				_v[1] = X(5);
				ensure(v[1] == X(5));
				ensure(v[v.size() + 1] == v[1]);
			}

			return 0;
		}
#endif // _DEBUG
	};

	template<class X, class I>
	class indirect_vector {
		// operator[](I::type j) { return v[i[j]]; }
	};

	template<class I = std::size_t>
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
			return slice(start + size*stride, 0, stride);
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
			{
				X _v1[3];
				X _v2[3];

				auto v1 = vector<X>(_v1);
				vector<X> v2(_v2);

				std::iota(v1.begin(), v1.end(), X(1));
				v2.copy(slice<X>(1, 3));
				ensure(v1.equal(v2));
			}

			return 0;
		}

#endif // _DEBUG
	};

	static constexpr CBLAS_UPLO CblasNoUpLo = static_cast<CBLAS_UPLO>(0);

	// non owning matrix
	template<typename X, CBLAS_TRANSPOSE TRANS = CblasNoTrans, CBLAS_UPLO UPLO = CblasNoUpLo>
	class matrix {
	protected:
		std::size_t r, c;
		X* a;
	public:
		using type = X;

		matrix(std::size_t r = 0, std::size_t c = 0, X* a = nullptr)
			: r(r), c(c), a(a)
		{ }
		matrix(const matrix&) = default;
		matrix& operator=(const matrix&) = default;
		~matrix()
		{ }

		explicit operator bool() const
		{
			return r != 0 and c != 0;
		}
		auto operator<=>(const matrix&) const = default;
		bool equal(const matrix& m) const
		{
			if (r != m.r or c != m.c)
				return false;

			for (std::size_t i = 0; i < r; ++i) {
				std::size_t jlo = (UPLO == CblasUpper) ? i : 0;
				std::size_t jhi = (UPLO == CblasLower) ? i : c;
				for (std::size_t j = jlo; j < jhi; ++j) {
					if (operator()(i, j) != m(i, j))
						return false;
				}
			}

			return true;
		}

		std::size_t index(std::size_t i, std::size_t j) const
		{
			return TRANS == CblasNoTrans ? i * c + j : i + j * r;
		}

		X operator()(std::size_t i, std::size_t j) const
		{
			return a[index(i, j)];
		}
		X& operator()(std::size_t i, std::size_t j)
		{
			return a[index(i, j)];
		}

		// unsafe linear copy
		template<class I>
		matrix& copy(const I& i)
		{
			std::copy(i.begin(), i.end(), a);

			return *this;
		}
		matrix& copy(const matrix& m)
		{
			for (std::size_t i = 0; i < r; ++i) {
				int jlo = (UPLO == CblasUpper) ? i : 0;
				int jhi = (UPLO == CblasLower) ? i : c;
				for (std::size_t j = jlo; j < jhi; ++j) {
					operator()(i, j) == m(i, j);
				}
			}

			return *this;
		}

		// int for BLAS
		int rows() const
		{
			return static_cast<int>(r);
		}
		int columns() const
		{
			return static_cast<int>(c);
		}
		int size() const
		{
			return static_cast<int>(r * c);
		}
		// leading dimension
		int ld() const
		{
			return TRANS == CblasNoTrans ? columns() : rows();
		}
		X* data()
		{
			return a;
		}
		const X* data() const
		{
			return a;
		}
		X* begin()
		{
			return a;
		}
		const X* begin() const
		{
			return a;
		}
		X* end()
		{
			return a + r * c;
		}
		const X* end() const
		{
			return a + r * c;
		}

		auto transpose() const
		{
			return matrix<X, TRANS == CblasNoTrans ? CblasTrans : CblasNoTrans>(c, r, a);
		}
		auto uplo(CBLAS_UPLO uplo) const
		{
			return matrix<X, TRANS, uplo>(r, c, a);
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
				X _a[6]; // = { 1,2,3; 4,5,6 };
				matrix<X> a(2, 3, _a);
				std::iota(a.begin(), a.end(), X(1));
				ensure(a.rows() == 2);
				ensure(a.columns() == 3);
				ensure(a.size() == 6);
				ensure(a(0, 0) == 1);
				ensure(a(0, 1) == 2);
				ensure(a(1, 0) == 4);
				ensure(a(1, 2) == 6);
			}
			{
				X _a[6]; // = { 1,2,3; 4;5,6 }' = {1,4; 2,5; 3,6}
				matrix<X, CblasTrans> a(3, 2, _a);
				std::iota(a.begin(), a.end(), X(1));
				ensure(a.rows() == 3);
				ensure(a.columns() == 2);
				ensure(a(0, 0) == 1);
				ensure(a(0, 1) == 4); // 0 + 3*1
				ensure(a(1, 0) == 2); // 1 + 3*0
				ensure(a(1, 1) == 5); // 1 + 3*1
				ensure(a(2, 0) == 3); // 0 + 3*0
				ensure(a(2, 1) == 6);
				a(0, 1) = 7;
				ensure(a(0, 1) == 7);
				ensure(a.transpose()(1, 0) == 7);
			}

			return 0;
		}

#endif // _DEBUG
	}; // matrix

	template<std::size_t N, typename X, CBLAS_TRANSPOSE TRANS = CblasNoTrans, CBLAS_UPLO UPLO = CblasNoUpLo>
	class identity_matrix : public matrix<X,TRANS,UPLO>
	{
		X id[N*N];
	public:
		identity_matrix()
			: matrix<X,TRANS,UPLO>(N, N)
		{
			matrix<X, TRANS, UPLO>::a = id;
			for (std::size_t i = 0; i < N; ++i)
				matrix<X,TRANS,UPLO>::operator()(i, i) = 1;
		}

#ifdef _DEBUG
		static int test()
		{
			{
				identity_matrix<3,X> id;
				ensure(id.rows() == 3);
				ensure(id.columns() == 3);
				for (std::size_t i = 0; i < 3; ++i) {
					for (std::size_t j = 0; j < 3; ++j) {
						ensure(id(i, j) == X(i == j));
					}
				}
			}

			return 0;
		}
#endif // _DEBUG
	}; // identity_matrix

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
	// general matrix multiplication with preallocated memory in _c
	template<class X, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB>
	inline matrix<X> gemm(const matrix<X,TA, CblasNoUpLo>& a, const matrix<X,TB, CblasNoUpLo>& b, X* _c, X alpha = 1, X beta = 0)
	{
		matrix<X> c(a.rows(), b.columns(), _c);

		if constexpr (std::is_same_v<X, float>) {
			cblas_sgemm(CblasRowMajor, TA, TB, a.rows(), b.columns(), a.columns(), alpha, a.data(), a.ld(), b.data(), b.ld(), beta, c.data(), c.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			cblas_dgemm(CblasRowMajor, TA, TB, a.rows(), b.columns(), a.columns(), alpha, a.data(), a.ld(), b.data(), b.ld(), beta, c.data(), c.ld());
		}

		return c;
	}

#ifdef _DEBUG
	template<class X>
	inline int gemm_test()
	{
		{
			X _a[6];
			matrix<X> a(2, 3, _a); // [1 2 3; 4 5 6]
			std::iota(a.begin(), a.end(), X(1));

			X _c[6];
			matrix<X> c(2, 3, _c);

			c = gemm<X>(identity_matrix<2, X>{}, a, c.data());
			ensure(c.equal(a));

			c = gemm<X>(a, identity_matrix<3, X>{}, c.data());
			ensure(c.equal(a));

			/*
			a.transpose(); // [1 4; 2 5; 3 6]
			c = blas::gemm<X>(transpose(a, id(2), _c);
			ensure(c.rows() == 3);
			ensure(c.columns() == 2);
			ensure(c.equal(a));
			*/
		}

		return 0;
	}

#endif // _DEBUG

} // namespace blas
#if 0


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

	// right multiply by diagonal matrix
	template<class X>
	inline matrix<X>& diag(matrix<X>& a, const vector<X>& d)
	{
		// row vector times scalar
		for (int i = 0; i < d.size(); ++i) {
			if constexpr (std::is_same_v<X, double>) {
				blas_dscal(d.size(), d[i], a.data() + i * a.ld(), 1);
			}
		}

		return a;
	}
	// left multiply by diagonal matrix
	template<class X>
	inline matrix<X>& diag(const vector<X>& d, matrix<X>& a)
	{
		// columns vector times scalar
		for (int i = 0; i < d.size(); ++i) {
			if constexpr (std::is_same_v<X, double>) {
				blas_dscal(d.size(), d[i], a.data() + i, a.ld());
			}
		}

		return a;
	}

	template<class X>
	using upper = matrix<X>::upper;

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