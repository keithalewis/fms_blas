// fms_blas_matrix.h
#pragma once
#include "fms_blas_vector.h"

namespace blas {

	inline CBLAS_TRANSPOSE transpose(CBLAS_TRANSPOSE t)
	{
		return (t == CblasTrans) ? CblasNoTrans : CblasTrans;
	}

	// non owning view of matrix
	template<typename T>
	class matrix {
	public:
		int r, c;
		T* a;
		CBLAS_TRANSPOSE t;
	public:
		using value_type = T;

		matrix(int r = 0, int c = 0, T* a = nullptr, CBLAS_TRANSPOSE t = CblasNoTrans)
			: r(r), c(c), a(a), t(t)
		{ }
		// square
		matrix(int n, T* a, CBLAS_TRANSPOSE t = CblasNoTrans)
			: r(n), c(n), a(a), t(t)
		{ }
		matrix(const matrix&) = default;
		matrix& operator=(const matrix&) = default;
		matrix(matrix&&) = default;
		matrix& operator=(matrix&&) = default;
		virtual ~matrix()
		{ }
		// column vector so ld() works
		matrix(const vector<T>& v)
			: matrix(v.size(), 1, v.data())
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

		//!!! virtual/transparent ???
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

		matrix& fill(T x)
		{
			as_vector().fill(x);

			return *this;
		}

		// linear copy
		matrix& copy(int _n, const T* _v, int _dn = 1)
		{
			as_vector().copy(_n, _v, _dn);

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
			assert(rows() == m.rows() and columns() == m.columns() and trans() == m.trans());

			copy(m.size(), m.data());

			return *this;
		}

		vector<T> as_vector() const
		{
			return vector<T>(r * c, a, 1);
		}
		matrix& reshape(int r_, int c_)
		{
			// assert (r_*c_ < r*c);
			r = r_;
			c = c_;

			return *this;
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
			return c; // (t == CblasNoTrans) ? c : r;
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
				assert(!a);
				assert(a.rows() == 0);
				assert(a.columns() == 0);
				assert(a.size() == 0);
				assert(a.data() == nullptr);

				auto a2{ a };
				assert(a2.equal(a));
				assert(!a2);
				assert(a2 == a);
				assert(!(a2 != a));
				assert(a2 <= a);
				assert(a2 >= a);
				assert(!(a2 < a));
				assert(!(a2 > a));

				assert(a2.rows() == 0);
				assert(a2.columns() == 0);
				assert(a2.size() == 0);
				assert(a2.data() == nullptr);

				a = a2;
				assert(!a);
				assert(a.rows() == 0);
				assert(a.columns() == 0);
				assert(a.size() == 0);
				assert(a.data() == nullptr);
				assert(!(a != a2));
			}

			T _a0[] = { 1, 2, 3, 4, 5, 6 };
			vector<T> a0(_a0);
			T _a[6], _b[6]; // backing store
			T constexpr NaN = std::numeric_limits<T>::quiet_NaN();

			{
				auto a = matrix<T>(2, 3, _a).copy(a0);
				auto b = matrix<T>(2, 3, _b).copy(_a0);
				assert(a.equal(b));

				auto a2{ a };
				assert(a2);
				assert(a2 == a);
				assert(a2.equal(a));
				assert(a.equal(transpose(transpose(a))));

				assert(a.rows() == 2);
				assert(a.columns() == 3);
				assert(a.size() == 6);
				assert(a(0, 0) == 1); assert(a(0, 1) == 2); assert(a(0, 2) == 3);
				assert(a(1, 0) == 4); assert(a(1, 1) == 5); assert(a(1, 2) == 6);
			}
			{
				auto a = matrix<T>(2, 3, _a, CblasTrans).copy(a0);
				assert(a.equal(transpose(transpose(a))));

				assert(a.rows() == 3);
				assert(a.columns() == 2);
				assert(a(0, 0) == 1); assert(a(0, 1) == 4);
				assert(a(1, 0) == 2); assert(a(1, 1) == 5);
				assert(a(2, 0) == 3); assert(a(2, 1) == 6);

				a = transpose(matrix<T>(2, 3, _a)).copy(a0);
				assert(a.rows() == 3);
				assert(a.columns() == 2);
				assert(a(0, 0) == 1); assert(a(0, 1) == 4);
				assert(a(1, 0) == 2); assert(a(1, 1) == 5);
				assert(a(2, 0) == 3); assert(a(2, 1) == 6);

				a(0, 1) = 7;
				assert(a(0, 1) == 7);
				assert(transpose(a)(1, 0) == 7);
			}
			{
				auto a = matrix<T>(2, 3, _a).copy(a0);
				assert(a.row(1).equal({ 4, 5, 6 }));
				assert(a.column(1).equal({ 2, 5 }));

				a = transpose(a);
				assert(a.column(1).equal({ 4, 5, 6 }));
				assert(a.row(1).equal({ 2, 5 }));
			}
			// row/column
			{
				auto a = matrix<T>(2, 3, _a).copy(a0);
				T _r1[] = { 4, 5, 6 };
				auto r1 = a.row(1);
				assert(r1.equal(vector<T>(_r1)));

				T _c1[] = { 2, 5 };
				assert(a.column(1).equal(vector<T>(_c1)));
			}

			return 0;
		}

#endif // _DEBUG

	}; // matrix


	template<std::size_t N, class T>
	class identity : public matrix<T> {
		T _a[N*N];
	public:
		identity()
			: matrix<T>(static_cast<int>(N), nullptr)
		{
			matrix<T>::a = _a;
			memset(_a, 0, N * N);
			for (std::size_t i = 0; i < N; ++i) {
				_a[i*(N + 1)] = 1;
			}
		}
	};

	template<class T>
	inline matrix<T> transpose(matrix<T> m)
	{
		m.t = blas::transpose(m.trans());

		return m;
	}

	// general rectangular matrix
	template<class T>
	using ge = matrix<T>;

	class UpLo {
		CBLAS_UPLO ul;
	public:
		UpLo(CBLAS_UPLO ul = CblasLower)
			: ul(ul)
		{ }
		CBLAS_UPLO uplo() const
		{
			return ul;
		}
		void uplo(CBLAS_UPLO ul_)
		{
			ul = ul_;
		}
	};

	class Diag  {
		CBLAS_DIAG d;
	public:
		Diag(CBLAS_DIAG d = CblasNonUnit)
			: d(d)
		{ }
		CBLAS_DIAG diag() const
		{
			return d;
		}
		void diag(CBLAS_DIAG d_)
		{
			d = d_;
		}
	};

	// matrix with attributes
	template<class T, class... As>
	struct matrixas : public matrix<T>, As...
	{
		matrixas()
		{ }
		matrixas(const matrix<T>& a, As... as)
			: matrix<T>(a), As(as)...
		{ }
		matrixas& operator=(const matrix<T>& a)
		{
			matrix<T>::operator=(a);

			return *this;
		}
	};

	template<class T, class... As>
	inline matrixas<T, As...> transpose(matrixas<T, As...> m)
	{
		m.t = blas::transpose(m.t);

		return m;
	}

	// packed triangular
	template<class T>
	using tp = matrixas<T, UpLo, Diag>;

	// triangular
	template<class T>
	using tr = matrixas<T, UpLo, Diag>;

	// symmetric
	template<class T>
	using sy = matrixas<T, UpLo>;

#ifdef _DEBUG
	template<class T>
	inline int tp_test()
	{
		{
			T a[] = { 1,2,3,4,5,6 };
			tp<T> A(matrix(2, a), CblasLower, CblasNonUnit);
			assert(CblasNoTrans == A.trans());
			assert(CblasLower == A.uplo());
			assert(CblasNonUnit == A.diag());
			
			//assert(1 == t);
		}
		{
			T a[] = { 1,2,3,4,5,6 };
			tp<T> A(matrix<T>(2, 2, a), CblasLower, CblasNonUnit);
			assert(A.uplo() == CblasLower);
		}

		return 0;
	}
#endif // _DEBUG
	
	
} // namespace blas
