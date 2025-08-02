/*

Fast Matrix Multiplication:

A multi-threaded algorithm for fast matrix multiplication, accelerated by
CUDA kernels for massively parallel operations.

Performs a single- or double-precision General Matrix Multiplication (GEMM)
as per Level 3 of the BLAS specification:

    C = α*AB + β*C

where

    C is the result matrix,
    A and B are the matrices to be multiplied,
    α and β are scaling constants.

*/

#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cuda/cmath>
#include "kernels.h"

#define cudaCheck(err) (kernels::cudaCheck(err, __FILE__, __LINE__))

// Type of matrix elements
typedef float matType;

template<typename T>
class Matrix
{
    std::vector<std::vector<T>> data_;

    int rows_, cols_;

    constexpr static std::string_view letter = "ABC";

public:

    // Construct from the number of rows and columns
    Matrix(int rows, int cols)
    :
        rows_(rows),
        cols_(cols)
    {
        data_.resize(rows, std::vector<T>(cols, 0.0));
    }

    // Initialize matrix with different random patterns based on thread ID
    void initialize(int threadId)
    {
        // Seed random number generator with thread ID for different values
        srand(time(nullptr) + threadId * 100);

        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                switch (threadId)
                {
                    // Identity-like pattern with some noise
                    case 0:
                        data_[i][j] =
                            (i == j)
                          ? 1.0 + (rand() % 100) / 100.0
                          : (rand() % 50) / 100.0;
                    break;

                    // Diagonal pattern
                    case 1:
                        data_[i][j] =
                            (i + j) % 2 == 0
                          ? (rand() % 200) / 100.0
                          : 0.1;
                    break;

                    // Upper triangular pattern
                    case 2:
                        data_[i][j] = (i <= j) ? (rand() % 500) / 100.0 : 0.0;
                    break;

                    default:
                        std::cerr << "Invalid number of matrices.\n";
                        std::exit(EXIT_FAILURE);

                }
            }
        }
    }

    // Print matrix for a given thread ID in columnated format
    void print(int threadId) const
    {
        std::cout << " Matrix " << letter[threadId]
            << " (" << rows_ << "x" << cols_ << "):\n";

        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2)
                    << data_[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // Get flattened matrix
    T* get() const
    {
        std::vector<T> flatMat;
        for (const auto& vec : data_)
        {
            flatMat.insert(flatMat.end(), vec.begin(), vec.end());
        }

        return flatMat.data();
    }
};

// Global function to create and store matrix per thread
template<typename T>
void createMatrix(int threadId, Matrix<T>& matrix)
{
    matrix.initialize(threadId);
}

template void createMatrix<matType>(int, Matrix<matType>&);

// Global mutex for synchronized printing
std::mutex printMutex;

// Thread function to print matrix (synchronized)
template<typename T>
void printMatrix(int threadId, const Matrix<T>& matrix)
{
    std::lock_guard<std::mutex> lock(printMutex);

    matrix.print(threadId);
}

int main()
{
    // Matrices definition
    constexpr int NUM_THREADS = 3;
    constexpr int NROWS = 16;
    constexpr int NCOLS = 16;
    const int M = NROWS, N = NROWS, K = NROWS;
    float alpha = 0.5;
    float beta = 3.0;

    // Create matrices to be filled by each thread
    std::vector<Matrix<matType>> matrices;
    matrices.reserve(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        matrices.emplace_back(NROWS, NCOLS);
    }

    // Create threads to initialize matrices
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back(createMatrix<matType>, t, std::ref(matrices[t]));
    }

    // Wait for all matrix creation threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    // Print initial A & B matrices
    std::cout << "=== Matrices Initialization ===\n";
    std::cout << "\nNumber of threads: " << NUM_THREADS << "\n";
    std::cout << "Total matrices: " << matrices.size() << "\n\n";

    for (int t = 0; t < NUM_THREADS - 1; ++t)
    {
        matrices[t].print(t);
    }

    // Host and device flattened matrices
    matType *A = nullptr, *B = nullptr, *C = nullptr;
    matType *dA = nullptr, *dB = nullptr, *dC = nullptr;

    A = matrices[0].get();
    B = matrices[1].get();
    C = matrices[2].get();

    cudaCheck(cudaMalloc((void**)&dA, sizeof(float) * NROWS * NCOLS));
    cudaCheck(cudaMalloc((void**)&dB, sizeof(float) * NROWS * NCOLS));
    cudaCheck(cudaMalloc((void**)&dC, sizeof(float) * NROWS * NCOLS));

    cudaCheck
    (
        cudaMemcpy(dA, A, sizeof(float) * NROWS * NCOLS, cudaMemcpyHostToDevice)
    );
    cudaCheck
    (
        cudaMemcpy(dB, B, sizeof(float) * NROWS * NCOLS, cudaMemcpyHostToDevice)
    );
    cudaCheck
    (
        cudaMemcpy(dC, C, sizeof(float) * NROWS * NCOLS, cudaMemcpyHostToDevice)
    );

    // Run CUDA kernel for fast matrix multiplication
    std::cout << "=== Running Matrix Multiplication Kernel ===\n";
    dim3 blockDim(16*16);
    dim3 gridDim(cuda::ceil_div(M, 16), cuda::ceil_div(N, 16));

    /* kernels::naive_multiply
        <<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC); */

    kernels::matrix_multiply<16>
        <<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    std::cout << "\nDone!\n";

    return 0;
}