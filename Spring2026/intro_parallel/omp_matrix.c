#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Use size_t for the constant to force 64-bit math from the start
const size_t N = 20000; 

double get_duration(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char* argv[]) {
    if (argc > 1) omp_set_num_threads(atoi(argv[1]));

    struct timespec ts_start, ts_init, ts_compute;

    // 1. Correct Allocation using size_t
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    float *A = (float*) malloc(N * N * sizeof(float));
    float *B = (float*) malloc(N * N * sizeof(float));
    float *C = (float*) malloc(N * N * sizeof(float));

    if (!A || !B || !C) {
        printf("Error: Memory allocation failed!\n");
        return 1;
    }

    // 2. Parallel Initialization
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i*N + j] = 1.0f;
            B[i*N + j] = 2.0f;
            C[i*N + j] = 0.0f;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_init);

    // 3. Parallel GEMM
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                // All indices (i, j, k, N) are size_t, preventing overflow
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_compute);

    printf("--- GEMM Report (N=%zu) ---\n", N);
    printf("Compute Time: %.6f sec\n", get_duration(ts_init, ts_compute));

    free(A); free(B); free(C);
    return 0;
}