// fms_blas.cpp - BLAS tests
#include <cassert>
#include "fms_lapack.h"
#include "fms_blas.h"

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
	blas::identity_matrix<3,X>::test();
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