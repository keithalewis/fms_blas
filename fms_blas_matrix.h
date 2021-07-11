// fms_blas_matrix.h
#pragma once
#include "fms_blas_vector.h"

namespace blas {

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
		virtual ~matrix()
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
			ensure(rows() == m.rows() and columns() == m.columns());

			for (int i = 0; i < rows(); ++i)
				row(i).copy(m.row(i));

			return *this;
		}

		vector<T> as_vector() const
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

	template<class T>
	struct triangular_matrix : public matrix<T> {
		CBLAS_UPLO uplo;
		CBLAS_DIAG diag;
	public:
		triangular_matrix(const matrix<T>& m, CBLAS_UPLO uplo, CBLAS_DIAG diag)
			: matrix<T>(m), uplo(uplo), diag(diag)
		{ }
	};

} // namespace blas
