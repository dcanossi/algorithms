/*

Implementation of CUDA kernels for matrix multiplication.

*/

#include <cstdio>
#include "kernels.h"

void kernels::cudaCheck(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess)
    {
        printf
        (
            "[CUDA ERROR] at file %s:%d:\n%s\n",
            file,
            line,
            cudaGetErrorString(error)
        );

        exit(EXIT_FAILURE);
    }
}

__global__ void kernels::naive_multiply
(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
        {
            tmp += A[x * K + i] * B[i * N + y];
        }

        // C = α*(AxB) + β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}