// fms_blas.h - BLAS wrappers
/*
* Use SCARY blas::matrix class
* matrix_transpose<CBLAS_TRANSPOSE>
* matrix_uplo<CBLAS_UPLO>
*/
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
#include <valarray>
#include "ensure.h"

#define INTEL_ONEMKL "https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/"
#define ONEMKL_CBLAS "top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/"

//                    https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/
//                    top/blas-and-sparse-blas-routines/blas-routines.html
// https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-rot.html

#define INTEL_CBLAS(x) INTEL_ONEMKL ONEMKL_CBLAS "cblas-" x ".html"

namespace blas {

#pragma warning(push)
#pragma warning(disable: 4724)
	// mod returning in range [0, y) 
	template<typename T>
	//	requires std::is_integral_v<T>
	inline T xmod(T x, T y)
	{
		T z = x % y;

		return z >= 0 ? z : z + y;
	}
#pragma warning(pop)

	static inline const char vector_doc[] = R"xyzyx(
	Non-owning strided view of array of T tailored to CBLAS.
)xyzyx";
	template<typename T>
	class vector {
	protected:
		int n, dn;
		T* v;
	public:
		using iterator_category = std::bidirectional_iterator_tag;
		using value_type = T;
		using reference = T&;
		using pointer = T*;
		using difference_type = ptrdiff_t;

		vector()
			: n(0), v(nullptr), dn(1)
		{ }
		
		// Allocation and lifetime of T* managed externally.
		vector(int n, T* v, int dn = 1)
			: n(n), v(v), dn(dn)
		{ }
		
		// T _v[] = {1, ...}; vector<T> v(_v);
		template<size_t N>
		vector(T(&v)[N])
			: n(static_cast<int>(N)), v(v), dn(1)
		{ }
		
		vector(const vector&) = default;
		vector& operator=(const vector&) = default;
		virtual ~vector()
		{ }

		explicit operator bool() const
		{
			return n != 0;
		}
		// size, pointer, and increment equality
		auto operator<=>(const vector&) const = default;

		int size() const
		{
			return n;
		}
		int incr() const
		{
			return dn;
		}
		pointer data()
		{
			return v;
		}
		const pointer data() const
		{
			return v;
		}

		// cyclic index
		T operator[](int i) const
		{
			return v[xmod(i * dn, n * abs(dn))];
		}
		T& operator[](int i)
		{
			return v[xmod(i * dn, n * abs(dn))];
		}

		// usable in range for
		auto begin()
		{
			return *this;
		}
		const auto begin() const
		{
			return *this;;
		}
		auto end()
		{
			return vector(0, v + n * abs(dn), dn);
		}
		const auto end() const
		{
			return vector(0, v + n * abs(dn), dn);
		}
		value_type operator*() const
		{
			return *v;
		}
		reference operator*()
		{
			return *v;
		}
		vector& operator++()
		{
			ensure(dn > 0);

			if (n) {
				--n;
				v += dn;
			}

			return *this;
		}
		vector& operator++(int)
		{
			auto tmp{ *this };

			operator++();

			return *this;
		}

		// equal size and contents, any incr
		bool equal(int _n, const T* _v, int _dn = 1) const
		{
			if (n != _n)
				return false;

			for (int i = 0; i < n; ++i)
				if (operator[](i) != _v[i * _dn])
					return false;

			return true;
		}
		// v.equal({w0, w1, ...})
		bool equal(const std::initializer_list<T>& w)
		{
			return equal((int)w.size(), w.begin(), 1);
		}
		bool equal(const vector& w) const
		{
			return equal(w.size(), w.data(), w.incr());
		}

		// assign values to data
		vector& copy(int n_, const T* v_, int dn_ = 1)
		{
			ensure(n == n_);

			if constexpr (std::is_same_v<T, float>)
				cblas_scopy(n_, v_, dn_, v, dn);
			if constexpr (std::is_same_v<T, double>)
				cblas_dcopy(n_, v_, dn_, v, dn);

			return *this;
		}
		// auto v = vector<T>(n, _v, dn).copy({w0, ...});
		vector& copy(const std::initializer_list<const T>& w)
		{
			return copy((int)w.size(), w.begin(), 1);
		}
		// auto v = vector<T>(n, _v, dn).copy(w);
		vector& copy(const vector& w)
		{
			return copy(w.size(), w.data(), w.incr());
		}

		// set all values to x
		vector& fill(T x)
		{
			for (int i = 0; i < n; ++i) {
				operator[](i) = x;
			}

			return *this;
		}

		// take from front (i > 0) or back (i < 0)
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

		// drop from front (i > 0) or back (i < 0)
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
				blas::vector<T> v;

				ensure(!v);
				ensure(v.size() == 0);
				ensure(v.data() == nullptr);

				blas::vector<T> v2{ v };
				ensure(!v2);
				ensure(v == v2);
				ensure(!(v2 != v));

				v = v2;
				ensure(!v);
				ensure(v.equal(v2));
			}
			{
				T _v[3];
				auto v = vector<T>(_v).copy({ 1, 2, 3 });

				T _v2[3];
				vector<T> v2(_v2);
				std::iota(v2.begin(), v2.end(), T(1));
				ensure(v.equal(v2));

				v2.copy(v);
				ensure(v2.equal(v));

				ensure(v);
				ensure(v.size() == 3);
				ensure(v[0] == 1);
				ensure(v[1] == 2);
				ensure(v[2] == 3);
				ensure(v[4] == 2);
				ensure(_v[1] == 2);

				v[1] = T(4);
				ensure(v[1] == T(4));
				_v[1] = T(5);
				ensure(v[1] == T(5));
				ensure(v[v.size() + 1] == v[1]);

				vector<T> w(v);
				ensure(v == w);
			}
			{
				T _v[3];
				auto v = vector<T>(3, _v).copy({ 1,2,3 });
				ensure(v);
				ensure(v.size() == 3);
				ensure(v[0] == 1);
				ensure(v[1] == 2);
				ensure(v[2] == 3);

				ensure(v.equal(3, _v));
				ensure(v.equal({ 1, 2, 3 }));
				ensure(v.equal(v));
			}
			{
				T _v[] = { 1,2,3 };
				vector<T> v(_v);
				v.take(2); // {1, 2}
				ensure(v.size() == 2);
				ensure(v[0] == 1);
				v.drop(-1); // {1}
				ensure(v.size() == 1);
				ensure(v[0] = 1);
				v.drop(10);
				ensure(v.size() == 0);
			}
			{
				T _v[6];
				auto v = vector<T>(3, _v, 2).copy({ 1,2,3 });
				auto vi = v.begin();
				ensure(*vi == T(1));
				++vi;
				ensure(*vi == T(2));
				vi++;
				ensure(*vi == T(3));
				++vi;
				ensure(vi == v.end());
			}
			{
				T _v[6];
				auto v = vector<T>(3, _v, 2).copy({ 1,2,3 });

				T i = 1;
				for (auto vi : v) {
					ensure(vi == i);
					i += 1;
				}
			}

			return 0;
		}
#endif // _DEBUG
	};

	template<class T>
	inline vector<T> take(int i, vector<T> v)
	{
		return v.take(i);
	}
	template<class T>
	inline vector<T> drop(int i, vector<T> v)
	{
		return v.drop(i);
	}

	//
	// BLAS level 1
	//

	// arg max |x_i|
	template<class T>
	inline auto iamax(const vector<T>& x)
	{
		if constexpr (std::is_same_v <T, float>) {
			return cblas_isamax(x.size(), x.data(), x.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			return cblas_idamax(x.size(), x.data(), x.incr());
		}
	}

	// arg min |x_i|
	template<class T>
	inline auto iamin(const vector<T>& x)
	{
		if constexpr (std::is_same_v <T, float>) {
			return cblas_isamin(x.size(), x.data(), x.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			return cblas_idamin(x.size(), x.data(), x.incr());
		}
	}

	// sum_i |x_i|
	template<class T>
	inline T asum(const vector<T>& x)
	{
		T s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (std::is_same_v <T, float>) {
			s = cblas_sasum(x.size(), x.data(), x.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			s = cblas_dasum(x.size(), x.data(), x.incr());
		}

		return s;
	}

	// y = a x + y
	template<class T>
	inline vector<T> axpy(T a, const vector<T>& x, vector<T> y)
	{
		if constexpr (std::is_same_v <T, float>) {
			cblas_saxpy(x.size(), a, x.data(), x.incr(), y.data(), y.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			cblas_daxpy(x.size(), a, x.data(), x.incr(), y.data(), y.incr());
		}

		return y;
	}

	// x . y
	template<class T>
	inline T dot(const vector<T>& x, const vector<T>& y)
	{
		T s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (std::is_same_v <T, float>) {
			s = cblas_sdot(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			s = cblas_ddot(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}

		return s;
	}

	// sqrt (sum_i v_i^2)
	template<class T>
	inline T nrm2(const vector<T>& x)
	{
		T s = std::numeric_limits<T>::quiet_NaN();

		if constexpr (std::is_same_v <T, float>) {
			s = cblas_snrm2(x.size(), x.data(), x.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			s = cblas_dnrm2(x.size(), x.data(), x.incr());
		}

		return s;
	}

	// x' = c x + y, y' = c y - s x
	template<class T>
	inline void rot(vector<T> x, vector<T> y, T c, T s)
	{
		if constexpr (std::is_same_v <T, float>) {
			cblas_srot(x.size(), x.data(), x.incr(), y.data(), y.incr(), c, s);
		}
		if constexpr (std::is_same_v <T, double>) {
			cblas_drot(x.size(), x.data(), x.incr(), y.data(), y.incr(), c, s);
		}
	}

	template<class T>
	inline void swap(vector<T> x, vector<T> y)
	{
		if constexpr (std::is_same_v <T, float>) {
			cblas_sswap(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
		if constexpr (std::is_same_v <T, double>) {
			cblas_dswap(x.size(), x.data(), x.incr(), y.data(), y.incr());
		}
	}

	// x = a x
	template<class T>
	inline vector<T> scal(T a, vector<T> x)
	{
		if (a != T(1) and x.size() > 0) {
			if constexpr (std::is_same_v <T, float>) {
				cblas_sscal(x.size(), a, x.data(), x.incr());
			}
			if constexpr (std::is_same_v <T, double>) {
				cblas_dscal(x.size(), a, x.data(), x.incr());
			}
		}

		return x;
	}

#ifdef _DEBUG

	template<class T>
	inline int blas1_test()
	{
		{
			T _v[3] = { 1, -2, 3 };
			auto v = vector<T>(_v);

			ensure(2 == iamax<T>(v));
			ensure(0 == iamin<T>(v));
			ensure(6 == asum<T>(v));
			ensure(n2 == sqrt(T(14)));
		}

		return 0;
	}

#endif // _DEBUG

	// non owning view of matrix
	template<typename T>
	class matrix {
	protected:
		int r, c;
		T* a;
		CBLAS_TRANSPOSE t;
	public:
		using value_type = T;

		matrix(int r = 0, int c = 0, T* a = nullptr, CBLAS_TRANSPOSE t = CblasNoTrans)
			: r(r), c(c), a(a), t(t)
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

		// equal contents taking transpose into account
		bool equal(const matrix& m) const
		{
			if (rows() != m.rows() or columns() != m.columns())
				return false;

			for (int i = 0; i < rows(); ++i)
				for (int j = 0; j < columns(); ++j)
					if (operator()(i, j) != m(i, j))
						return false;

			return true;
		}
		// equal to upper or lower component of m only
		bool equal(const matrix& m, CBLAS_UPLO ul) const
		{
			if (rows() != m.rows() or columns() != m.columns())
				return false;

			for (int i = 0; i < rows(); ++i) {
				int jlo = (ul == CblasUpper) ? i : 0;
				int jhi = (ul == CblasLower) ? i : columns();
				for (int j = jlo; j < jhi; ++j)
					if (operator()(i, j) != m(i, j))
						return false;
			}

			return true;
		}

		//!!! virtual ???
		int index(int i, int j) const
		{
			if (t == CblasTrans)
				std::swap(i, j);

			return i * c + j;
		}

		T operator()(int i, int j) const
		{
			return a[index(i, j)];
		}
		T& operator()(int i, int j)
		{
			return a[index(i, j)];
		}

		// linear copy
		matrix& copy(int _n, const T* _v, int _dn = 1)
		{
			to_vector().copy(_n, _v, _dn);

			return *this;
		}
		// auto m = matrix(r, c).copy({a0, ...});
		matrix& copy(const std::initializer_list<T>& v)
		{
			return copy((int)v.size(), v.begin(), 1);
		}
		matrix& copy(const vector<T>& v)
		{
			return copy(v.size(), v.data(), v.incr());
		}
		matrix& copy(const matrix<T>& m)
		{
			ensure(rows() == m.rows() and columns() == m.columns());

			for (int i = 0; i < rows(); ++i)
				row(i).copy(m.row(i));

			return *this;
		}

		vector<T> to_vector() const
		{
			return vector<T>(r * c, a, 1);
		}

		int rows() const
		{
			return t == CblasTrans ? c : r;
		}
		int columns() const
		{
			return t == CblasTrans ? r : c;
		}
		int size() const
		{
			return r * c;
		}
		CBLAS_TRANSPOSE trans() const
		{
			return t;
		}
		T* data()
		{
			return a;
		}
		const T* data() const
		{
			return a;
		}
		T* begin()
		{
			return a;
		}
		const T* begin() const
		{
			return a;
		}
		T* end()
		{
			return a + r * c;
		}
		const T* end() const
		{
			return a + r * c;
		}

		// leading dimension
		int ld() const
		{
			return c;
		}

		matrix transpose() const
		{
			matrix<T> m(*this);

			m.t = (t == CblasTrans) ? CblasNoTrans : CblasTrans;

			return m;
		}
		vector<T> row(int i) const
		{
			return t == CblasTrans ? vector<T>(r, a + i, c) : vector<T>(c, a + i * c, 1);
		}
		vector<T> column(int j) const
		{
			return t == CblasTrans ? vector<T>(c, a + j * c, 1) : vector<T>(r, a + j, c);
		}
		vector<T> row(int i, CBLAS_UPLO uplo) const
		{
			return uplo == CblasUpper ? row(i).drop(i) : row(i).take(i);
		}
		vector<T> column(int j, CBLAS_UPLO uplo) const
		{
			return uplo == CblasUpper ? column(j).drop(j) : column(j).take(j);
		}

#ifdef _DEBUG

		static int test()
		{
			{
				matrix<T> a;
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

			T _a0[] = { 1, 2, 3, 4, 5, 6 };
			vector<T> a0(_a0);
			T _a[6], _b[6]; // backing store
			T constexpr NaN = std::numeric_limits<T>::quiet_NaN();

			{
				auto a = matrix<T>(2, 3, _a).copy(a0);
				auto b = matrix<T>(2, 3, _b).copy(_a0);
				ensure(a.equal(b));

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
				auto a = matrix<T>(2, 3, _a, CblasTrans).copy(a0);
				ensure(a.equal(a.transpose().transpose()));

				ensure(a.rows() == 3);
				ensure(a.columns() == 2);
				ensure(a(0, 0) == 1); ensure(a(0, 1) == 4);
				ensure(a(1, 0) == 2); ensure(a(1, 1) == 5);
				ensure(a(2, 0) == 3); ensure(a(2, 1) == 6);

				a = matrix<T>(2, 3, _a).transpose().copy(a0);
				ensure(a.rows() == 3);
				ensure(a.columns() == 2);
				ensure(a(0, 0) == 1); ensure(a(0, 1) == 4);
				ensure(a(1, 0) == 2); ensure(a(1, 1) == 5);
				ensure(a(2, 0) == 3); ensure(a(2, 1) == 6);

				a(0, 1) = 7;
				ensure(a(0, 1) == 7);
				ensure(a.transpose()(1, 0) == 7);
			}
			{
				auto a = matrix<T>(2, 3, _a).copy(a0);
				ensure(a.row(1).equal({ 4, 5, 6 }));
				ensure(a.column(1).equal({ 2, 5 }));

				a = a.transpose();
				ensure(a.column(1).equal({ 4, 5, 6 }));
				ensure(a.row(1).equal({ 2, 5 }));
			}
			// row/column
			{
				auto a = matrix<T>(2, 3, _a).copy(a0);
				T _r1[] = { 4, 5, 6 };
				auto r1 = a.row(1);
				ensure(r1.equal(vector<T>(_r1)));

				T _c1[] = { 2, 5 };
				ensure(a.column(1).equal(vector<T>(_c1)));
			}

			return 0;
		}

#endif // _DEBUG
	}; // matrix

	template<std::size_t N, typename T>
	class identity_matrix : public matrix<T>
	{
		inline static T _id[N * N];
	public:
		identity_matrix()
			: matrix<T>(N, N)
		{
			matrix<T>::a = _id;
			if (_id[0] != T(1))
				for (int i = 0; i < N; ++i)
					matrix<T>::operator()(i, i) = T(1);
		}

#ifdef _DEBUG
		static int test()
		{
			{
				identity_matrix<3, T> id;
				ensure(id.rows() == 3);
				ensure(id.columns() == 3);
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						ensure(id(i, j) == T(i == j));
					}
				}
			}

			return 0;
		}
#endif // _DEBUG
	}; // identity_matrix

	//
	// BLAS level 2
	//

	// y = alpha op(A)*x + beta y
	template<class T>
	inline vector<T> gemv(const matrix<T>& a, const vector<T>& x, T* _y, T alpha = T(1), T beta = T(0))
	{
		vector<T> y(a.rows(), _y);

		if constexpr (std::is_same_v<T, float>) {
			cblas_sgemv(CblasRowMajor, a.trans(), a.rows(), a.columns(), alpha, a.data(), a.ld(),
				x.data(), x.incr(), beta, y.data(), y.incr());
		}
		if constexpr (std::is_same_v<T, double>) {
			cblas_dgemv(CblasRowMajor, a.trans(), a.rows(), a.columns(), alpha, a.data(), a.ld(),
				x.data(), x.incr(), beta, y.data(), y.incr());
		}

		return y;
	}

	// scale rows of m by v
	template<class T>
	inline matrix<T> scal(const vector<T>& v, matrix<T> m)
	{
		for (int i = 0; i < m.rows(); ++i) {
			scal<T>(v[i], m.row(i));
		}

		return m;
	}

#ifdef _DEBUG

	template<class T>
	inline int scal_test()
	{
		T _v[] = { 1,2,3 };
		{
			T _a[6];
			matrix<T> a(2, 3, _a);
			std::iota(a.begin(), a.end(), T(1));

			scal<T>(vector<T>(2, _v), a); // rows
			ensure(a(0, 0) == 1);   ensure(a(0, 1) == 2);   ensure(a(0, 2) == 3);
			ensure(a(1, 0) == 2 * 4); ensure(a(1, 1) == 2 * 5); ensure(a(1, 2) == 2 * 6);
		}
		{
			T _a[6];
			matrix<T> a(2, 3, _a, CblasTrans);
			std::iota(a.begin(), a.end(), T(1));
			// {1 4; 2 5; 3 6}

			scal<T>(vector<T>(3, _v), a); // columns
			ensure(a(0, 0) == 1);   ensure(a(0, 1) == 4);
			ensure(a(1, 0) == 2 * 2); ensure(a(1, 1) == 2 * 5);
			ensure(a(2, 0) == 3 * 3); ensure(a(2, 1) == 3 * 6);
		}

		return 0;
	}

#endif // _DEBUG

	//
	// BLAS level 3
	// 

	// general matrix multiplication using preallocated memory in _c
	// C = alpha op(A) * op(B) + beta C 
	template<class T>
	inline matrix<T> gemm(const matrix<T>& a, const matrix<T>& b, T* _c, T alpha = 1, T beta = 0)
	{
		int m = a.rows();
		int k = a.columns();
		ensure(k == b.rows());
		int n = b.columns();

		matrix<T> c(m, n, _c);

		int lda = a.ld();
		int ldb = b.ld();
		int ldc = c.ld();

		if constexpr (std::is_same_v<T, float>) {
			cblas_sgemm(CblasRowMajor, a.trans(), b.trans(), m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
		}
		if constexpr (std::is_same_v<T, double>) {
			cblas_dgemm(CblasRowMajor, a.trans(), b.trans(), m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
		}

		return c;
	}

#ifdef _DEBUG
	template<class T>
	inline int gemm_test()
	{
		{
			T _a[6];
			matrix<T> a(2, 3, _a); // [1 2 3; 4 5 6]
			std::iota(a.begin(), a.end(), T(1));

			T _c[6];
			matrix<T> c(2, 3, _c);
			identity_matrix<2, T> id2;
			c = gemm<T>(id2, a, c.data());
			ensure(c.equal(a));
			std::fill(c.begin(), c.end(), T(-1));

			identity_matrix<3, T> id3;
			c = gemm<T>(a, id3, c.data());
			ensure(c.equal(a));

			c = gemm<T>(id3, a.transpose(), c.data());
			ensure(c.equal(a.transpose()));

			c = gemm<T>(a.transpose(), id2, c.data());
			ensure(c.equal(a.transpose()));
		}

		return 0;
	}

#endif // _DEBUG

	// b = alpha op(a)*b or b = alpha b*op(a)
	template<class T>
	inline matrix<T>& trmm(CBLAS_SIDE lr, CBLAS_UPLO ul, const matrix<T>& a, matrix<T>& b, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		if constexpr (std::is_same_v<T, float>) {
			cblas_strmm(CblasRowMajor, lr, ul, a.trans(), diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());
		}
		if constexpr (std::is_same_v<T, double>) {
			cblas_dtrmm(CblasRowMajor, lr, ul, a.trans(), diag,
				b.rows(), b.columns(), alpha, a.data(), a.ld(), b.data(), b.ld());
		}

		return b;
	}
	template<class T>
	inline matrix<T>& trmm(CBLAS_UPLO ul, const matrix<T>& a, matrix<T>& b, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		return trmm(CblasLeft, ul, a, b, alpha, diag);
	}
	template<class T>
	inline matrix<T>& trmm(matrix<T>& b, CBLAS_UPLO ul, const matrix<T>& a, T alpha = 1, CBLAS_DIAG diag = CblasNonUnit)
	{
		return trmm(CblasRight, ul, a, b, alpha, diag);
	}

#ifdef _DEBUG

	template<class T>
	inline int trmm_test()
	{
		{
			T _i[] = { 1, 2, 3, 1 };
			const matrix<T> i(2, 2, _i); // [1 2; 3 1]
			T _a[6];
			matrix<T> a(2, 3, _a);
			std::iota(a.begin(), a.end(), T(1));

			// [1 2  * [1 2 3
			//  . 1] *  4 5 6]
			// = [1 + 8, 2 + 10, 3 + 12
			//    4      5       6]
			trmm<T>(CblasLeft, CblasUpper, i, a);
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
			T _i[] = { 1, 2, 3, 1 };
			const matrix<T> i(2, 2, _i); // [1 2; 3 1]
			T _a[6];
			matrix<T> a(2, 3, _a);
			std::iota(a.begin(), a.end(), T(1));
			
			// [1 .  * [1 2 3
			//  3 1] *  4 5 6]
			// = [1      2      3
			//    3 + 4, 6 + 5, 9 + 6]
			trmm<T>(CblasLeft, CblasLower, i, a);
			ensure(a.rows() == 2);
			ensure(a.columns() == 3);
			ensure(a(0, 0) == 1); ensure(a(0, 1) == 2); ensure(a(0, 2) == 3);
			ensure(a(1, 0) == 7); ensure(a(1, 1) == 11); ensure(a(1, 2) == 15);
		}
		{
			T _i[] = { 1, 2, 3, 1 };
			const matrix<T> i(2, 2, _i); // [1 2; 3 1]
			T _a[6];
			matrix<T> a(3, 2, _a);
			std::iota(a.begin(), a.end(), T(1));

			// [1 2    [1 2
			//  3 4  *  . 1]
			//  5 6]
			// = [1, 2 + 2
			//    3, 6 + 4
			//    5, 10 + 6]
			trmm<T>(CblasRight, CblasUpper,  i, a);
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
			T _i[] = { 1, 2, 3, 1 };
			const matrix<T> i(2, 2, _i); // [1 2; 3 1]
			T _a[6];
			matrix<T> a(3, 2, _a);
			std::iota(a.begin(), a.end(), T(1));

			// [1 2    [1 .
			//  3 4  *  3 1]
			//  5 6]
			// = [1 + 6,  2
			//    3 + 12, 4
			//    5 + 18, 6]
			trmm<T>(CblasRight, CblasLower, i, a);
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

} // namespace blas

