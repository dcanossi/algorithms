/*
Fast Matrix Multiplication:

A multi-core, multithreaded algorithm for fast matrix multiplication,
leveraged with CUDA kernels for massively parallel operations.
*/

#include <mpi.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class Matrix
{
    std::vector<std::vector<double>> data_;

    int rows_, cols_;

    constexpr static std::string_view letter = "ABCD";

public:

    // Construct from the number of rows and columns
    Matrix(int rows, int cols)
    :
        rows_(rows),
        cols_(cols)
    {
        data_.resize(rows, std::vector<double>(cols, 0.0));
    }

    // Initialize matrix with different random patterns based on process rank
    // and thread ID
    void initialize(int rank, int threadId)
    {
        // Seed random number generator with process rank and thread ID
        // for different values
        srand(time(nullptr) + rank * 1000 + threadId * 100);

        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                if (rank == 0)
                {
                    if (threadId == 0)
                    {
                        // Process 0, thread 0:
                        // Identity-like pattern with some noise
                        data_[i][j] =
                            (i == j)
                          ? 1.0 + (rand() % 100) / 100.0
                          : (rand() % 50) / 100.0;
                    }
                    else
                    {
                        // Process 0, thread 1: Diagonal pattern
                        data_[i][j] =
                            (i + j) % 2 == 0
                          ? (rand() % 200) / 100.0
                          : 0.1;
                    }
                }
                else
                {
                    if (threadId == 0)
                    {
                        // Process 1, thread 0: Upper triangular pattern
                        data_[i][j] = (i <= j) ? (rand() % 500) / 100.0 : 0.0;
                    }
                    else
                    {
                        // Process 1, thread 1: Lower triangular pattern
                        data_[i][j] = (i >= j) ? (rand() % 300) / 100.0 : 0.0;
                    }
                }
            }
        }
    }

    // Print matrix for a given process in columnated format
    void print(int rank, int threadId) const
    {
        const int idx = threadId + 2 * rank;

        std::cout << " Matrix " << letter[idx]
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
};

// Global function to create and store matrix per thread
void createMatrix(int rank, int threadId, Matrix& matrix)
{
    matrix.initialize(rank, threadId);
}

// Global mutex for synchronized printing
std::mutex printMutex;

// Thread function to print matrix (synchronized)
void printMatrix(int rank, int threadId, const Matrix& matrix)
{
    std::lock_guard<std::mutex> lock(printMutex);

    matrix.print(rank, threadId);
}

int main(int argc, char** argv)
{
    // Matrix count and dimensions
    constexpr int NUM_THREADS = 2;
    constexpr int MATRIX_ROWS = 4;
    constexpr int MATRIX_COLS = 4;

    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "Warning: MPI does not fully support multithreading\n";
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if we have exactly 2 processes
    if (size != 2)
    {
        if (rank == 0)
        {
            std::cerr << "This program requires exactly 2 MPI processes.\n";
            std::cerr << "Run with: mpirun -np 2 fastMatMult\n";
        }

        MPI_Finalize();
        return 1;
    }

    // Create matrices for each thread
    std::vector<Matrix> matrices;
    matrices.reserve(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        matrices.emplace_back(MATRIX_ROWS, MATRIX_COLS);
    }

    // Create threads to initialize matrices
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back(createMatrix, rank, t, std::ref(matrices[t]));
    }

    // Wait for all matrix creation threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    // Synchronize MPI processes before printing
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "=== Parallel Matrix Initialization ===\n";
        std::cout << "Number of processes: " << size << "\n";
        std::cout << "Number of threads per process: " << NUM_THREADS << "\n";
        std::cout << "Total matrices: " << size * NUM_THREADS << "\n\n";

        // Process 0 prints its matrices first
        for (int t = 0; t < NUM_THREADS; ++t)
        {
            matrices[t].print(rank, t);
        }

        // Signal process 1 to print its matrices next
        int signal = 1;
        MPI_Send(&signal, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Wait for process 1 to finish printing
        MPI_Recv(&signal, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1)
    {
        // Wait for signal from process 0
        int signal;
        MPI_Recv(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Process 1 prints its matrices
        for (int t = 0; t < NUM_THREADS; ++t)
        {
            matrices[t].print(rank, t);
        }

        // Signal back to process 0 that printing is done
        signal = 1;
        MPI_Send(&signal, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}