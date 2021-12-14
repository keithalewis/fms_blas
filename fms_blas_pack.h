// fms_blas_pack.h - pack and unpack triangular matrices
#pragma once
#include <mkl_cblas.h>

namespace blas {

	// index into packed matrix
	inline int indexp(int i, int j)
	{
		return (i * (i + 1)) / 2 + j;
	}
	// index into upper triangle of rectangular matrix
	inline int indexu(int i, int j, int n)
	{
		return i + n * j;
	}
	// index into lower triangle of rectangular matrix
	inline int indexl(int i, int j, int n)
	{
		return n * i + j;
	}

	// pack upper triangle of n x n matrix a into l
	template<class T>
	inline void packu(int n, const T* a, T* l)
	{
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j <= i; ++j) {
				l[indexp(i, j)] = a[indexu(i, j, n)];
			}
		}
	}
	// pack lower triangle of n x n matrix a into l
	template<class T>
	inline void packl(int n, const T* a, T* l)
	{
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j <= i; ++j) {
				l[(i * (i + 1)) / 2 + j] = a[n * i + j];
			}
		}
	}
	template<class T>
	inline void pack(CBLAS_UPLO uplo, int n, const T* a, T* l)
	{
		if (uplo == CblasLower) {
			packl(n, a, l);
		}
		else if (uplo == CblasUpper) {
			packu(n, a, l);
		}
	}
#ifdef _DEBUG
	template<class T>
	inline int pack_test()
	{
		{
			T a[] = { 1,2,3,
					  4,5,6,
					  7,8,9 };
			T l[6];
			packu(3, a, l);
			assert(1 == l[0]);
			assert(2 == l[1]);
			assert(5 == l[2]);
			assert(3 == l[3]);
			assert(6 == l[4]);
			assert(9 == l[5]);
		}
		{
			T a[] = { 1,2,3,
					  4,5,6,
					  7,8,9 };
			T l[6];
			pack(CblasUpper, 3, a, l);
			assert(1 == l[0]);
			assert(2 == l[1]);
			assert(5 == l[2]);
			assert(3 == l[3]);
			assert(6 == l[4]);
			assert(9 == l[5]);
		}
		{
			T a[] = { 1,2,3,
					  4,5,6,
					  7,8,9 };
			T l[6];
			packl(3, a, l);
			assert(1 == l[0]);
			assert(4 == l[1]);
			assert(5 == l[2]);
			assert(7 == l[3]);
			assert(8 == l[4]);
			assert(9 == l[5]);
		}
		{
			T a[] = { 1,2,3,
					  4,5,6,
					  7,8,9 };
			T l[6];
			pack(CblasLower, 3, a, l);
			assert(1 == l[0]);
			assert(4 == l[1]);
			assert(5 == l[2]);
			assert(7 == l[3]);
			assert(8 == l[4]);
			assert(9 == l[5]);
		}

		return 0;
	}
#endif // _DEBUG

	// unpack l into lower triangle of a
	template<class T>
	inline void unpackl(int n, const T* l, T* a)
	{
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j <= i; ++j) {
				a[n * i + j] = l[(i * (i + 1)) / 2 + j];
			}
		}
	}
	// unpack l into upper triangle of a
	template<class T>
	inline void unpacku(int n, const T* l, T* a)
	{
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j <= i; ++j) {
				a[i + n * j] = l[(i * (i + 1)) / 2 + j];
			}
		}
	}
	// unpack l into symmetric a
	template<class T>
	inline void unpacks(int n, const T* l, T* a)
	{
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j <= i; ++j) {
				a[j + n * i] = a[i + n * j] = l[(i * (i + 1)) / 2 + j];
			}
		}
	}
#ifdef _DEBUG
	template<class T>
	inline int unpack_test()
	{
		{
			T l[] = { 1,2,3,4,5,6 };
			T a[9] = { 0,0,0,0,0,0,0,0,0 };
			unpackl(3, l, a);
			assert(1 == a[0]);
			assert(0 == a[1]);
			assert(0 == a[2]);
			assert(2 == a[3]);
			assert(3 == a[4]);
			assert(0 == a[5]);
			assert(4 == a[6]);
			assert(5 == a[7]);
			assert(6 == a[8]);
		}
		{
			T l[] = { 1,2,3,4,5,6 };
			T a[9] = { 0,0,0,0,0,0,0,0,0 };
			unpacku(3, l, a);
			assert(1 == a[0]);
			assert(2 == a[1]);
			assert(4 == a[2]);
			assert(0 == a[3]);
			assert(3 == a[4]);
			assert(5 == a[5]);
			assert(0 == a[6]);
			assert(0 == a[7]);
			assert(6 == a[8]);
		}
		{
			T l[] = { 1,2,3,4,5,6 };
			T a[9] = { 0,0,0,0,0,0,0,0,0 };
			unpacks(3, l, a);
			assert(1 == a[0]);
			assert(2 == a[1]);
			assert(4 == a[2]);
			assert(2 == a[3]);
			assert(3 == a[4]);
			assert(5 == a[5]);
			assert(4 == a[6]);
			assert(5 == a[7]);
			assert(6 == a[8]);
		}

		return 0;
	}
#endif // _DEBUG


} // namespace blas
