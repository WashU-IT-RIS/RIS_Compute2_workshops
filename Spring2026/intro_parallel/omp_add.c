#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // Required for omp_get_max_threads()

#define ARRAY_SIZE 100000000

double get_duration(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char* argv[]) {
    // OpenMP usually respects OMP_NUM_THREADS, but we can also set it via argument
    if (argc > 1) {
        omp_set_num_threads(atoi(argv[1]));
    }

    struct timespec ts_start, ts_alloc, ts_init, ts_compute, ts_end;

    // 1. Timing Allocation
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    double *a = malloc(ARRAY_SIZE * sizeof(double));
    double *b = malloc(ARRAY_SIZE * sizeof(double));
    double *res = malloc(ARRAY_SIZE * sizeof(double));
    clock_gettime(CLOCK_MONOTONIC, &ts_alloc);

    // 2. Timing Initialization (Serial)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = 1.0; b[i] = 2.0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_init);

    // 3. Timing Parallel Computation with OpenMP
    // The "pragma" handles thread creation, work-sharing, and synchronization
    #pragma omp parallel for
    for (int i = 0; i < ARRAY_SIZE; i++) {
        res[i] = a[i] + b[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_compute);

    printf("--- OpenMP Performance Report (%d threads) ---\n", omp_get_max_threads());
    printf("Allocation time:    %.6f sec\n", get_duration(ts_start, ts_alloc));
    printf("Init (Serial):      %.6f sec\n", get_duration(ts_alloc, ts_init));
    printf("Compute (Parallel): %.6f sec\n", get_duration(ts_init, ts_compute));
    printf("Total execution:     %.6f sec\n", get_duration(ts_start, ts_compute));
    printf("---------------------------------------------\n");

    free(a); free(b); free(res);
    return 0;
}
