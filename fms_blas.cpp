// fms_blas.cpp - BLAS tests
#include <cassert>
#include "fms_lapack.h"
#include "fms_blas.h"

void xerbla(const char* srname, const int* info, const int)
{
	char buf[256];

	if (*info < 0) {
		sprintf_s(buf, "%s: parameter %d had an illegal value", srname, -*info);
	}
	else if (*info == 1001) {
		sprintf_s(buf, "%s: incompatible optional parameters", srname);
	}
	else if (*info == 1000 or *info == 1089) {
		sprintf_s(buf, "%s: insufficient workspace available", srname);
	}
	else if (info > 0) {
		sprintf_s(buf, "%s: returned error code %d", srname, *info);
	}

	puts(buf);
}

void LAPACKE_xerbla(const char* name, lapack_int info)
{
	char buf[256];

	if (info < 0) {
		sprintf_s(buf, "%s: parameter %d had an illegal value", name, -info);
	}
	else if (info == LAPACK_WORK_MEMORY_ERROR) {
		sprintf_s(buf, "%s: not enough memory to allocate work array", name);
	}
	else if (info == LAPACK_TRANSPOSE_MEMORY_ERROR) {
		sprintf_s(buf, "%s: not enough memory to transpose matrix", name);
	}
	else if (info > 0) {
		sprintf_s(buf, "%s: returned error code %d", name, info);
	}

	puts(buf);
}

template<class X>
int fms_blas_test()
{
#ifdef _DEBUG
	blas::pack_test<X>();
	blas::unpack_test<X>();
	blas::iamax_test<X>();
	blas::axpy_test<X>();
	blas::vector<X>::test();
	blas::vector_alloc<X>::test();
	blas::matrix<X>::test();
	blas::matrix_alloc<X>::test();
	blas::blas2_test<X>();
	blas::blas3_test<X>();
	lapack::potr_test<X>();
	lapack::pptr_test<X>();

#endif // _DEBUG

	return 0;
}

int main()
{
	fms_blas_test<double>();
	fms_blas_test<float>();
	/*
	lapack::potrf_test<float>();
	lapack::potrf_test<double>();
*/
	return 0;
}