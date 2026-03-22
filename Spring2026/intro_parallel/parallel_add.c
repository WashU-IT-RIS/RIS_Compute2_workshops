#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Problem size increased 100x: 100 Million elements
#define ARRAY_SIZE 100000000

typedef struct {
    int start;
    int end;
    double *a;
    double *b;
    double *result;
} ThreadData;

// Helper to calculate duration in seconds
double get_duration(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void* vector_add(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start; i < data->end; i++) {
        data->result[i] = data->a[i] + data->b[i];
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : 4;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
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

    // 3. Timing Parallel Computation
    int chunk_size = ARRAY_SIZE / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? ARRAY_SIZE : (i + 1) * chunk_size;
        thread_data[i].a = a;
        thread_data[i].b = b;
        thread_data[i].result = res;
        pthread_create(&threads[i], NULL, vector_add, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_compute);

    // Results Display
    printf("--- Performance Report (%d threads) ---\n", num_threads);
    printf("Allocation time:   %.6f sec\n", get_duration(ts_start, ts_alloc));
    printf("Init (Serial):     %.6f sec\n", get_duration(ts_alloc, ts_init));
    printf("Compute (Parallel): %.6f sec\n", get_duration(ts_init, ts_compute));
    printf("Total execution:    %.6f sec\n", get_duration(ts_start, ts_compute));
    printf("---------------------------------------\n");

    free(a); free(b); free(res);
    return 0;
}