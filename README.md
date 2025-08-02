# algorithms
Set of handy algorithms to perform varied mathematical operations.

## Fast Matrix Multiplication

A multi-threaded algorithm for fast matrix multiplication, accelerated by
CUDA kernels for massively parallel operations.

Performs a single- or double-precision General Matrix Multiplication (GEMM)
as per Level 3 of the BLAS specification:

    C = α*AB + β*C

where

    C is the result matrix,
    A and B are the matrices to be multiplied,
    α and β are scaling constants.

### Compilation and run

- Install all necessary dependencies: CUDA toolkit 12 and a C++ compiler with
support for C++17.
- Compile with: `nvcc -std=c++17 -Wno-deprecated-gpu-targets -o fastMatMult kernels.cu fastMatMult.cu`
- Run with: ./fastMatMult