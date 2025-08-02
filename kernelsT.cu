/*

Implementation of CUDA template kernels for matrix multiplication.

*/

template <const int BlockSize>
__global__ void kernels::matrix_multiply
(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C
)
{
    // The output block that we want to compute in this thread block
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // Allocate buffer for current block in fast shared memory.
    // It is shared between all threads in a block.
    __shared__ float As[BlockSize * BlockSize];
    __shared__ float Bs[BlockSize * BlockSize];

    // The inner row & column that we're accessing in this thread
    const uint threadCol = threadIdx.x % BlockSize;
    const uint threadRow = threadIdx.x / BlockSize;

    // Advance pointers to the starting positions
    A += cRow * BlockSize * K;                    // row = cRow, col = 0
    B += cCol * BlockSize;                        // row = 0, col = cCol
    C += cRow * BlockSize * N + cCol * BlockSize; // row = cRow, col = cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BlockSize)
    {
        // Have each thread load one of the elements in A & B.
        // Make the threadCol (== threadIdx.x) the consecutive index
        // to allow global memory access coalescing.
        As[threadRow * BlockSize + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BlockSize + threadCol] = B[threadRow * N + threadCol];

        // Block threads in this block until cache is fully populated
        __syncthreads();
        A += BlockSize;
        B += BlockSize * N;

        // Execute the dot product on the currently cached block
        for (int dotIdx = 0; dotIdx < BlockSize; ++dotIdx)
        {
            tmp +=
                As[threadRow * BlockSize + dotIdx]
              * Bs[dotIdx * BlockSize + threadCol];
        }

        // Need to sync again at the end, to avoid faster threads fetching
        // the next block into the cache before slower threads are done.
        __syncthreads();
    }

    // C = α*(AxB) + β*C
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
}