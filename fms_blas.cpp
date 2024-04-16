// fms_blas.cpp - BLAS tests
#include <cassert> 
#include "fms_lapack.h"
#include "fms_blas.h"



template<class X>
int fms_blas_test()
{
#ifdef _DEBUG
	blas::constant_test<X>();
	blas::pack_test<X>();
	blas::unpack_test<X>();
	blas::vector<X>::test();
	blas::matrix<X>::test();
	blas::tp_test<X>();
	blas::blas1_test<X>();
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