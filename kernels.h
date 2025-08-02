/*

Header file for CUDA kernels declaration.

*/

#ifndef kernels_H
#define kernels_H

namespace kernels
{

// Helper function for checking CUDA errors
void cudaCheck(cudaError_t error, const char* file, int line);

// Naive row-column matrix multiplication kernel
__global__ void naive_multiply
(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C
);

// Optimized matrix multiplication kernel
template <const int BlockSize>
__global__ void matrix_multiply
(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C
);

}

#include "kernelsT.cu"

#endif