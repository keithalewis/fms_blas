# C++ wrappers for BLAS and LAPACK

This library consists of C++ wrappers that respect the zen of LAPACK.
Unlike C++, BLAS and LAPACK routines do not allocate memory; it is your job to
provide pointers to sufficient space for return values. Function arguments
are often views on existing memory. The classes `blas::vector` and `blas::matrix`
provide C++ conveniences for this, but you must manage memory lifetimes yourself.
These are returned as values by some `const` member functions but contain non-`const`
pointers that can be used to modify memory so you can't rely on const-correctness 
or value types when using this library.

The design goal is to map as simply as possible to the underlying BLAS and LAPACK functions.
For example `blas::axpy<T>(a, x, y)` calls
`cblas_?axpy(x.size(), a, x.data(), x.incr(), y.data(), y.incr())` where `a` has type `T`,
`x` is `const blas::vector<T>& x`, `y` is `blas::vector<T>&` (where `?` is the BLAS
type corresponding to `T`) and updates `y` to `a * x + y`.

## `blas::vector`

A BLAS vector has a size, pointer to data, and an increment that defaults to `1`.
It defines the default spaceship operator but that compares data pointers. Use
the member function `equal` to compare vector contents even if vectors have different
increments. Use `copy` to move data into a vector, e.g., `v.copy({1,2,3})` will
copy `{1,2,3}` into the array `v` is pointing at. It will throw an exception if the size of `v` is not 3.
Use `v.fill(x)` to set all values to the same number `x`.

The member functions `operator[]` access either the value or reference at the appropriate increment.
It is cyclic so, e.g., `v[-1]` is the last element. BLAS doesn't know anything about this member function.
Other functions invisible to BLAS are all the accoutremnts needed to make a vector
a bidirectional iterator if its increment is positive.
The functons `take(i)` and `drop(i)` take or drop elements from the beginning (i > 0)
or end (i < 0) of the array. Free standing verions are also defined that do not modify
the vector.

A subset of BLAS level 1 (vector-vector) operations are provided.

## `blas::matrix`

A BLAS matrix has rows, columns, data, and a transpose flag. Computing the transpose
of a non-square matrix is non-trivial but BLAS makes affordances for that.
We write `op(a)` for `a` if it is not transposed and `a'` if it is.'
For example, `blas::gemm(a, b, c, alpha = 1, beta = 0)` performs a GEneral Matrix-Matrix
product. It calls
`cblas_?gemm(CblasRowMajor, a.trans(), b.trans(), a.rows(), b.columns(), a.columns(), alpha, a.data(), a.ld(), b.data(), b.ld(), beta, c.data(), c.ld())`,
where `a` and `b` are const references to matrices, `c` has size at least `a.rows() * b.columns()`
and updates `c` to `alpha * op(a) * op(b) + beta * c`.
The `trans` flags of `a` and `b` indicate BLAS should adjust its algorithms appropriately without actually
performing a messy transpose.

If one of the matrices in the product is upper or lower triangular then the product can be
performed using the memory of the other. The function `trmm(uplo, a, b, alpha = 1)`
updates the content of `matrix& b` with `alpha op(a)*b` and the function
`trmm(b, uplo, a, alpha)` updates `b` with `alpha b * op(a)`.
	
## Remarks

vector equal is looser than matrix equal
`vector` is a view of strided memory.