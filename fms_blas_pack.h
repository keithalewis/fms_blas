// fms_blas_pack.h - pack and unpack triangular matrices
#pragma once

namespace blas {

	// pack upper triangle of a into l
	template<class T>
	inline void packu(int n, const T* a, T* l)
	{
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j <= i; ++j) {
				l[(i * (i + 1)) / 2 + j] = a[i + n * j];
			}
		}
	}
	// pack lower triangle of a into l
	template<class T>
	inline void packl(int n, const T* a, T* l)
	{
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j <= i; ++j) {
				l[(i * (i + 1)) / 2 + j] = a[n * i + j];
			}
		}
	}

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
	// unpack l into a
	template<class T>
	inline void unpack(int n, const T* l, T* a)
	{
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j <= i; ++j) {
				a[j + n * i] = a[i + n * j] = l[(i * (i + 1)) / 2 + j];
			}
		}
	}
	/*
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
	*/


} // namespace blas
