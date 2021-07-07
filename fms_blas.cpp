// fms_blas.cpp - BLAS tests
#include "fms_blas.h"

template<class X>
int fms_blas_test()
{
	blas::vector<X>::test();
	blas::slice<>::test<X>();
	blas::matrix<X>::test();
	blas::identity_matrix<3,X>::test();
	blas::gemm_test<X>();
	blas::trmm_test<X>();

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