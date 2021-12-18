// fms_xerbla.cpp - xerbla throwing std::runtime_error
#pragma once
#include <cstdio>
#include <stdexcept>
#include <mkl_lapacke.h>

void xerbla(const char* srname, const int* info, const int)
{
	char buf[256] = { 0 };

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

	throw std::runtime_error(buf);
}

void LAPACKE_xerbla(const char* name, lapack_int info)
{
	char buf[256] = { 0 };

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

	puts(buf); //  throw std::runtime_error(buf);// declared nothrow somewhere
}