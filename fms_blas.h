// fms_blas.h - BLAS wrappers
#pragma once
//#pragma warning(push)
#pragma warning(disable: 4820)
#pragma warning(disable: 4365)
#include <mkl_cblas.h>
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

	// non-owning strided array of X
	template<typename X>
	class vector {
	protected:
		int n, dn;
		X* v;
	public:
		using type = typename X;

		vector(int n = 0, X* v = nullptr, int dn = 1)
			: n(n), v(v), dn(dn)
		{ }
		template<size_t N>
		vector(X(&v)[N])
			: n(static_cast<int>(N)), v(v), dn(1)
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

		int size() const
		{
			return n;
		}
		int incr() const
		{
			return dn;
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
		X operator[](int i) const
		{
			return v[xmod(i * dn, n * dn)];
		}
		X& operator[](int i)
		{
			return v[xmod(i * dn, n * dn)];
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
			return v + n * abs(dn);
		}
		const X* end() const
		{
			return v + n * abs(dn);
		}

		vector& copy(const vector& w)
		{
			if constexpr (std::is_same_v<X,float>)
				cblas_scopy(n, v, dn, w.data(), w.incr());
			if constexpr (std::is_same_v<X, double>)
				cblas_dcopy(n, v, dn, w.data(), w.incr());
		}
		template<class I>
		vector& copy(const X* pv)
		{
			return copy(vector<X>(n, pv, 1));
		}

		bool equal(const vector& w) const
		{
			int i = 0;

			while (i < n and i < w.size()) {
				if (operator[](i) != w[i])
					return false;
				++i;
			}

			return i == w.size();
		}

		vector& take(int i)
		{
			i = std::clamp(i, -n, n);

			if (i >= 0) {
				n = i;
			}
			else {
				n = -i;
				v += n + i * abs(dn);
			}

			return *this;
		}
		vector& drop(int i)
		{
			i = std::clamp(i, -n, n);

			if (i > 0) {
				n -= i;
				v += i * abs(dn);
			}
			else if (i < 0) {
				n += i;
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
				X _v[3];
				vector<X> v(_v);
				std::iota(v.begin(), v.end(), X(1));
				ensure(v);
				ensure(v.size() == 3);
				ensure(v[0] == 1);
				ensure(v[1] == 2);
				ensure(v[2] == 3);
				ensure(v[4] == 2);
				ensure(_v[1] == 2);

				// blas::vector<X> v2(&_v[0]);
				vector<X> v2(3, &_v[0]);
				ensure(v2.equal(v));

				v[1] = X(4);
				ensure(v[1] == X(4));
				_v[1] = X(5);
				ensure(v[1] == X(5));
				ensure(v[v.size() + 1] == v[1]);

				vector<X> w(v);
				ensure(v == w);
			}
			{
				X _v[] = { 1,2,3 };
				vector<X> v(_v);
				v.take(2); // {1, 2}
				ensure(v.size() == 2);
				ensure(v[0] == 1);
				v.drop(-1); // {1}
				ensure(v.size() == 1);
				ensure(v[0] = 1);
				v.drop(10);
				ensure(v.size() == 0);
			}

			return 0;
		}
#endif // _DEBUG
	};

	template<class X>
	inline vector<X> take(int i, vector<X> v)
	{
		return v.take(i);
	}
	template<class X>
	inline vector<X> drop(int i, vector<X> v)
	{
		return v.drop(i);
	}

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

			return 0;
		}

#endif // _DEBUG
	};

	static constexpr CBLAS_UPLO CblasNoUplo = static_cast<CBLAS_UPLO>(0);

	// non owning matrix
	template<typename X, CBLAS_TRANSPOSE TRANS = CblasNoTrans, CBLAS_UPLO UPLO = CblasNoUplo>
	class matrix {
	protected:
		int r, c;
		X* a;
	public:
		using type = X;

		matrix(int r = 0, int c = 0, X* a = nullptr)
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
		// unsafe underlying data equality
		template<class I>
		bool equal(const I& i) const
		{
			return std::equal(i.begin(), i.end(), begin()/*, end()*/);
		}
		// slow but accurate data and shape equality
		bool equal(const matrix& m) const
		{
			if (r != m.r or c != m.c)
				return false;

			for (int i = 0; i < r; ++i) {
				int jlo = (UPLO == CblasUpper) ? i : 0;
				int jhi = (UPLO == CblasLower) ? i : c;
				for (int j = jlo; j < jhi; ++j) {
					if (operator()(i, j) != m(i, j))
						return false;
				}
			}

			return true;
		}

		int index(int i, int j) const
		{
			return TRANS == CblasNoTrans ? i * c + j : i + j * r;
		}

		X operator()(int i, int j) const
		{
			return a[index(i, j)];
		}
		X& operator()(int i, int j)
		{
			return a[index(i, j)];
		}

		// unsafe linear copy
		matrix& copy(int n, const X* pa)
		{
			std::copy(pa, pa + n, a);

			return *this;
		}
		template<class I>
		matrix& copy(const I& i)
		{
			std::copy(i.begin(), i.end(), begin());

			return *this;
		}
		// careful copy
		matrix& copy(const matrix& m)
		{
			ensure(r*c <= m.size());

			for (int i = 0; i < r; ++i) {
				int jlo = (UPLO == CblasUpper) ? i : 0;
				int jhi = (UPLO == CblasLower) ? i : c;
				for (int j = jlo; j < jhi; ++j) {
					operator()(i, j) = m(i, j);
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
		template<CBLAS_UPLO UPLO>
		auto uplo() const
		{
			return matrix<X, TRANS, UPLO>(r, c, a);
		}

		vector<X> _row(int i) const
		{
			return vector<X>(c, a + i * c, 1);
		}
		vector<X> _column(int j) const
		{
			return vector<X>(r, a + j, c);
		}
		vector<X> row(int i) const
		{
			i = xmod(i, r);

			vector<X> vi = (TRANS == CblasNoTrans) ? _row(i) : transpose()._column(i);

			if constexpr (UPLO == CblasUpper)
				vi.drop(i);
			else if constexpr (UPLO == CblasLower)
				vi.drop(-i);

			return vi;
		}
		vector<X> column(int j) const
		{
			j = xmod(j, c);

			vector<X> vj = (TRANS == CblasNoTrans) ? _column(j) : transpose()._row(j);

			if constexpr (UPLO == CblasUpper)
				vj.drop(j);
			else if constexpr (UPLO == CblasLower)
				vj.drop(-j);

			return vj;
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
				ensure(a2.equal(a));
				ensure(!a2);
				ensure(a2 == a);
				ensure(!(a2 != a));
				ensure(a2 <= a);
				ensure(a2 >= a);
				ensure(!(a2 < a));
				ensure(!(a2 > a));

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

			auto _a0 = std::initializer_list<X>({ 1, 2, 3, 4, 5, 6 });
			X _a[6], _b[6]; // backing store
			X constexpr NaN = std::numeric_limits<X>::quiet_NaN();

			{
				auto a = matrix<X>(2, 3, _a).copy(_a0);
				auto a2{ a };
				ensure(a2);
				ensure(a2 == a);
				ensure(a2.equal(a));
				ensure(a.equal(a.transpose().transpose()));

				ensure(a.rows() == 2);
				ensure(a.columns() == 3);
				ensure(a.size() == 6);
				ensure(a(0, 0) == 1); ensure(a(0, 1) == 2); ensure(a(0, 2) == 3);
				ensure(a(1, 0) == 4); ensure(a(1, 1) == 5); ensure(a(1, 2) == 6);
			}
			{
				auto a = matrix<X, CblasTrans>(3, 2, _a).copy(_a0);
				ensure(a.equal(a.transpose().transpose()));
				
				ensure(a.rows() == 3);
				ensure(a.columns() == 2);
				ensure(a(0, 0) == 1); ensure(a(0, 1) == 4);
				ensure(a(1, 0) == 2); ensure(a(1, 1) == 5);
				ensure(a(2, 0) == 3); ensure(a(2, 1) == 6);

				a = matrix<X>(2, 3, _a).transpose().copy(_a0);
				ensure(a.rows() == 3);
				ensure(a.columns() == 2);
				ensure(a(0, 0) == 1); ensure(a(0, 1) == 4);
				ensure(a(1, 0) == 2); ensure(a(1, 1) == 5);
				ensure(a(2, 0) == 3); ensure(a(2, 1) == 6);

				a(0, 1) = 7;
				ensure(a(0, 1) == 7);
				ensure(a.transpose()(1, 0) == 7);
			}
			// upper
			{
				auto a = matrix<X, CblasNoTrans, CblasUpper>(2, 3, _a).copy(_a0);
				auto b = matrix<X, CblasNoTrans, CblasUpper>(2, 3, _b).copy(_a0);
				ensure(a.equal(b));
				for (int i = 1; i < b.rows(); ++i)
					for (int j = 0; j < i; ++j)
						b(i, j) = NaN;
				ensure(a.equal(b));
			}
			{
				auto a = matrix<X>(2, 3, _a).uplo<CblasUpper>().copy(_a0);
				auto b = matrix<X, CblasNoTrans, CblasUpper>(2, 3, _b).copy(_a0);
				ensure(a.equal(b));
				for (int i = 1; i < b.rows(); ++i)
					for (int j = 0; j < i; ++j)
						b(i, j) = NaN;
				ensure(a.equal(b));
			}
			{
				auto a = matrix<X, CblasTrans, CblasUpper>(3, 2, _a).copy(_a0);
				auto b = matrix<X, CblasTrans, CblasUpper>(3, 2, _b).copy(_a0);
				ensure(a.equal(b));
				for (int i = 1; i < b.rows(); ++i)
					for (int j = 0; j < i; ++j)
						b(i, j) = NaN;
				ensure(a.equal(b));
			}
			// lower
			{
				auto a = matrix<X, CblasNoTrans, CblasLower>(2, 3, _a).copy(_a0);
				auto b = matrix<X, CblasNoTrans, CblasLower>(2, 3, _b).copy(_a0);
				ensure(a.equal(b));
				for (int i = 1; i < b.rows(); ++i)
					for (int j = i + 1; j < b.columns(); ++j)
						b(i, j) = NaN;
				ensure(a.equal(b));
			}
			{
				auto a = matrix<X, CblasTrans, CblasLower>(3, 2, _a).copy(_a0);
				auto b = matrix<X, CblasTrans, CblasLower>(3, 2, _b).copy(_a0);
				ensure(a.equal(b));
				for (int i = 1; i < b.rows(); ++i)
					for (int j = i + 1; j < b.columns(); ++j)
						b(i, j) = NaN;
				ensure(a.equal(b));
			}
			// row/column
			{
				auto a = matrix<X>(2, 3, _a).copy(_a0);
				X _r1[] = { 4, 5, 6 };
				auto r1 = a._row(1);
				ensure(r1.equal(vector<X>(_r1)));

				X _c1[] = { 2, 5 };
				auto c1 = a._column(1);
				ensure(c1.equal(vector<X>(_c1)));
			}
			{
				auto a = matrix<X>(2, 3, _a).copy(_a0);
				X _r1[] = { 4, 5, 6 };
				auto r1 = a.row(1);
				ensure(r1.equal(vector<X>(_r1)));
				ensure(a.uplo<CblasUpper>().row(1).equal(vector<X>(_r1).drop(1)));

				X _c1[] = { 2, 5 };
				auto c1 = a.column(1);
				ensure(c1.equal(vector<X>(_c1)));
				ensure(a.uplo<CblasLower>().column(1).equal(vector<X>(_c1).drop(-1)));

				ensure(a.transpose().row(1).equal(vector<X>(_c1)));
				ensure(a.transpose().column(1).equal(vector<X>(_r1)));
			}
			{
				auto a = matrix<X>(2, 3, _a).copy(_a0);
				X _r1[] = { 4, 5, 6 };
				auto r1 = a._row(1);
				ensure(r1.equal(vector<X>(_r1)));
				X _c1[] = { 2, 5 };
				auto c1 = a._column(1);
				ensure(c1.equal(vector<X>(_c1)));
			}

			return 0;
		}

#endif // _DEBUG
	}; // matrix

	// static???
	template<std::size_t N, typename X, CBLAS_TRANSPOSE TRANS = CblasNoTrans, CBLAS_UPLO UPLO = CblasNoUplo>
	class identity_matrix : public matrix<X,TRANS,UPLO>
	{
		X id[N*N];
	public:
		identity_matrix()
			: matrix<X,TRANS,UPLO>(N, N)
		{
			matrix<X, TRANS, UPLO>::a = id;
			for (int i = 0; i < N; ++i)
				matrix<X,TRANS,UPLO>::operator()(i, i) = 1;
		}

#ifdef _DEBUG
		static int test()
		{
			{
				identity_matrix<3,X> id;
				ensure(id.rows() == 3);
				ensure(id.columns() == 3);
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
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
	
	// sum_i |v_|
	template<class X>
	inline X asum(const vector<X>& v)
	{
		X s = std::numeric_limits<X>::quiet_NaN();

		if constexpr (std::is_same_v <X, float>) {
			s = cblas_sasum(v.size(), v.data(), v.incr());
		}
		if constexpr (std::is_same_v <X, double>) {
			s = cblas_dasum(v.size(), v.data(), v.incr());
		}

		return s;
	}
	// v . w
	template<class X>
	inline X dot(const vector<X>& v, const vector<X>& w)
	{
		X s = std::numeric_limits<X>::quiet_NaN();

		if constexpr (std::is_same_v <X, float>) {
			s = cblas_sdot(v.size(), v.data(), v.incr(), w.data(), w.incr());
		}
		if constexpr (std::is_same_v <X, double>) {
			s = cblas_ddot(v.size(), v.data(), v.incr(), w.data(), w.incr());
		}

		return s;
	}
	// v = a v
	template<class X>
	inline vector<X>& scal(X a, vector<X>& v)
	{
		if constexpr (std::is_same_v <X, float>) {
			cblas_sscal(v.size(), a, v.data(), v.incr());
		}
		if constexpr (std::is_same_v <X, double>) {
			cblas_dscal(v.size(), a, v.data(), v.incr());
		}

		return v;
	}

	//
	// BLAS level 2
	//
	
	// scale rows/cols of m by v
	template<class X, CBLAS_TRANSPOSE TRANS, CBLAS_UPLO UPLO>
	inline matrix<X, TRANS, UPLO>& scal(const vector<X>& v, matrix<X, TRANS, UPLO>& m)
	{
		for (int i = 0; i < v.size(); ++i) {
			vector<X> vi = m.row(i); // row or column i

			if (v[i] != X(1) and vi.size() > 0) {
				scal<X>(v[i], vi);
			}
		}

		return m;
	}

#ifdef _DEBUG

	template<class X>
	inline int scal_test()
	{
		X _v[] = { 1,2,3 };
		{
			X _a[6];
			matrix<X> a(2, 3, _a);
			std::iota(a.begin(), a.end(), X(1));

			scal<X>(vector<X>(2, _v), a); // rows
			ensure(a(0, 0) == 1);   ensure(a(0, 1) == 2);   ensure(a(0, 2) == 3);
			ensure(a(1, 0) == 2*4); ensure(a(1, 1) == 2*5); ensure(a(1, 2) == 2*6);
		}
		{
			X _a[6];
			matrix<X, CblasTrans> a(3, 2, _a);
			std::iota(a.begin(), a.end(), X(1));
			// {1 4; 2 5; 3 6}

			scal<X>(vector<X>(3, _v), a); // columns
			ensure(a(0, 0) == 1);   ensure(a(0, 1) == 4);
			ensure(a(1, 0) == 2*2); ensure(a(1, 1) == 2*5);
			ensure(a(2, 0) == 3*3); ensure(a(2, 1) == 3*6);
		}
		{
			X _a[6];
			auto a = matrix<X>(2, 3, _a).uplo<CblasUpper>();
			std::iota(a.begin(), a.end(), X(1));

			scal<X>(vector<X>(2, _v), a); // rows
			ensure(a(0, 0) == 1); ensure(a(0, 1) == 2);     ensure(a(0, 2) == 3);
			ensure(a(1, 0) == 4); ensure(a(1, 1) == 2 * 5); ensure(a(1, 2) == 2 * 6);
		}
		{
			X _a[6];
			auto a = matrix<X>(2, 3, _a).transpose().uplo<CblasUpper>();
			std::iota(a.begin(), a.end(), X(1));
			// {1 4; 2 5; 3 6}

			scal<X>(vector<X>(3, _v), a); // columns
			ensure(a(0, 0) == 1); ensure(a(0, 1) == 4);
			ensure(a(1, 0) == 1 * 2); ensure(a(1, 1) == 2 * 5);
			ensure(a(2, 0) == 3); ensure(a(2, 1) == 6);
		}

		return 0;
	}

#endif // _DEBUG

	//
	// BLAS level 3
	// 
	// general matrix multiplication with preallocated memory in _c
	template<class X, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB>
	inline matrix<X> gemm(const matrix<X,TA, CblasNoUplo>& a, const matrix<X,TB, CblasNoUplo>& b, X* _c, X alpha = 1, X beta = 0)
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
			identity_matrix<2, X> id2;
			c = gemm<X>(id2, a, c.data());
			ensure(c.equal(a));
			std::fill(c.begin(), c.end(), X(-1));
			// c = gemm<X>(identity_matrix<2, X>{}, a, c.data()); // garbage

			identity_matrix<3, X> id3;
			c = gemm<X>(a, id3, c.data());
			ensure(c.equal(a));

			//auto a_ = a.transpose(); // [1 4; 2 5; 3 6]
			//c = blas::gemm<X>(a.transpose(), id2, _c);
			//ensure(c.equal(a));
			
		}

		return 0;
	}

#endif // _DEBUG


// template<X,T,U>
// inline matrix<X,T,U> operator*(const matrix<X,T,U>& a, const matrix<X,T,U>& b);
// template<>
// inline matrix<X,T,CblasNoUplo> operator*(const matrix<X,T,U>& a, const matrix<X,T,U>& b)
// { return gemm(a, b); }

	// b = op(a)*b;
	template<class X, CBLAS_TRANSPOSE TRANS, CBLAS_UPLO UPLO>
	inline matrix<X>& trmm(const matrix<X,TRANS,UPLO>& a, matrix<X>& b, X alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		static_assert(UPLO != CblasNoUplo);
		
		if constexpr (std::is_same_v<X, float>) {
			cblas_strmm(CblasRowMajor, CblasLeft, UPLO, TRANS, diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());
		}
		if constexpr (std::is_same_v<X, double>) {
			cblas_dtrmm(CblasRowMajor, CblasLeft, UPLO, TRANS, diag, 
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());	
		}

		return b;
	}
	// b = b*op(a);
	template<class X, CBLAS_TRANSPOSE TRANS, CBLAS_UPLO UPLO>
	inline matrix<X>& trmm(matrix<X>& b, const matrix<X,TRANS,UPLO>& a, X alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		static_assert(UPLO != CblasNoUplo);

		if constexpr (std::is_same_v<X, float>) {
			cblas_strmm(CblasRowMajor, CblasRight, UPLO, TRANS, diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.columns());
		}
		if constexpr (std::is_same_v<X, double>) {
			cblas_dtrmm(CblasRowMajor, CblasRight, UPLO, TRANS, diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.columns());
		}

		return b;
	}

#ifdef _DEBUG

	template<class X>
	inline int trmm_test()
	{
		{
			X _i[] = { 1, 2, 3, 1 };
			const matrix<X, CblasNoTrans, CblasUpper> i(2, 2, _i); // [1 2; 3 1]
			X _a[6];
			matrix<X> a(2, 3, _a);
			std::iota(a.begin(), a.end(), X(1));

			// [1 2  * [1 2 3
			//  . 1] *  4 5 6]
			// = [1 + 8, 2 + 10, 3 + 12
			//    4      5       6]
			trmm<X>(i, a);
			ensure(a.rows() == 2);
			ensure(a.columns() == 3);
			ensure(a(0, 0) == 9);
			ensure(a(0, 1) == 12);
			ensure(a(0, 2) == 15);
			ensure(a(1, 0) == 4);
			ensure(a(1, 1) == 5);
			ensure(a(1, 2) == 6);
		}
		{
			X _i[] = { 1, 2, 3, 1 };
			const matrix<X, CblasNoTrans, CblasLower> i(2, 2, _i); // [1 2; 3 1]
			X _a[6];
			matrix<X> a(2, 3, _a);
			std::iota(a.begin(), a.end(), X(1));
			
			// [1 .  * [1 2 3
			//  3 1] *  4 5 6]
			// = [1      2      3
			//    3 + 4, 6 + 5, 9 + 6]
			trmm<X>(i, a);
			ensure(a.rows() == 2);
			ensure(a.columns() == 3);
			ensure(a(0, 0) == 1); ensure(a(0, 1) == 2); ensure(a(0, 2) == 3);
			ensure(a(1, 0) == 7); ensure(a(1, 1) == 11); ensure(a(1, 2) == 15);
		}
		{
			X _i[] = { 1, 2, 3, 1 };
			const matrix<X, CblasNoTrans, CblasUpper> i(2, 2, _i); // [1 2; 3 1]
			X _a[6];
			matrix<X> a(3, 2, _a);
			std::iota(a.begin(), a.end(), X(1));

			// [1 2    [1 2
			//  3 4  *  . 1]
			//  5 6]
			// = [1, 2 + 2
			//    3, 6 + 4
			//    5, 10 + 6]
			trmm<X>(a, i);
			ensure(a.rows() == 3);
			ensure(a.columns() == 2);
			ensure(a(0, 0) == 1);
			ensure(a(0, 1) == 4);
			ensure(a(1, 0) == 3);
			ensure(a(1, 1) == 10);
			ensure(a(2, 0) == 5);
			ensure(a(2, 1) == 16);
		}
		{
			X _i[] = { 1, 2, 3, 1 };
			const matrix<X, CblasNoTrans, CblasLower> i(2, 2, _i); // [1 2; 3 1]
			X _a[6];
			matrix<X> a(3, 2, _a);
			std::iota(a.begin(), a.end(), X(1));

			// [1 2    [1 .
			//  3 4  *  3 1]
			//  5 6]
			// = [1 + 6,  2
			//    3 + 12, 4
			//    5 + 18, 6]
			trmm<X>(a, i);
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

#endif // _DEBUG

	
	/*
	// right multiply by diagonal matrix
	template<class X, CBLAS_TRANSPOSE TRANS, CBLAS_UPLO UPLO>
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
	*/

} // namespace blas

