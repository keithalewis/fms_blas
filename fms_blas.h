// fms_blas.h - BLAS wrappers
#pragma once
#include "fms_blas_vector_alloc.h"
#include "fms_blas_matrix_alloc.h"
#include "fms_blas3.h"

#define INTEL_ONEMKL "https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/"
#define ONEMKL_CBLAS "top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/"

//                    https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/
//                    top/blas-and-sparse-blas-routines/blas-routines.html
// https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-rot.html

#define INTEL_CBLAS(x) INTEL_ONEMKL ONEMKL_CBLAS "cblas-" x ".html"
