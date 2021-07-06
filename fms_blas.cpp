// fms_blas.cpp - BLAS tests
#include "fms_blas.h"

int main()
{
	blas::vector<int>::test();
	blas::vector<float>::test();
	blas::vector<double>::test();

	blas::slice<>::test<int>();
	blas::slice<>::test<float>();
	blas::slice<>::test<double>();

	blas::matrix<int>::test();
	blas::matrix<float>::test();
	blas::matrix<double>::test();
/*
	blas::triangular_matrix<int>::test();
	blas::triangular_matrix<float>::test();
	blas::triangular_matrix<double>::test();


	blas::test_mm<double>();

	lapack::potrf_test<float>();
	lapack::potrf_test<double>();
*/
	return 0;
}