#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <omp.h>

#define rdpmc(ecx, eax, edx)    \
    asm volatile (              \
        "rdpmc"                 \
        : "=a"(eax),            \
          "=d"(edx)             \
        : "c"(ecx))

/*
 *  usage - how to run the program
 *      @return: -1
 */
int32_t
usage(void)
{
    printf("\t./mat_mul <N>\n");
    return -1;
}

/*
 *  print_matrix - if you need convincing that it works just fine
 *      @N: square matrix size
 *      @m: pointer to matrix
 */
void
print_matrix(uint32_t N, long *m)
{
    for (uint32_t i=0; i<N; ++i) {
        for (uint32_t j=0; j<N; ++j)
            printf("%3ld ", m[i*N + j]);
        printf("\n");
    }
}

/*
 *  main - program entry point
 *      @argc: number of arguments & program name
 *      @argv: arguments
 */
int32_t
main(int32_t argc, char *argv[])
{
    if (argc != 2)
        return usage();

    /* allocate space for matrices */
    clock_t t;
    uint32_t N   = atoi(argv[1]);
    int64_t  *m1 = malloc(N * N * sizeof(int64_t));
    int64_t  *m2 = malloc(N * N * sizeof(int64_t));
    int64_t  *r  = malloc(N * N * sizeof(int64_t));
    double wc_start, wc_end;

    /* initialize matrices */
    for (uint32_t i=0; i<N*N; ++i) {
        m1[i] = i;
        m2[i] = i;
    }
    printf("%d\n", N);
    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    wc_start = omp_get_wtime();
    t = clock();

    /* TODO: count L2 cache misses for the next block using RDPMC */
    uint32_t eax, edx;
    rdpmc(0, eax, edx);
    int64_t start_cnt = ((int64_t)eax) | ((int64_t)edx << 32);
    /* perform slow multiplication */
    for (uint32_t i=0; i<N; ++i)             /* line   */
        for (uint32_t j=0; j<N; ++j)         /* column */
            for (uint32_t k=0; k<N; ++k)
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    /* clock delta */
    t = clock() - t;
    wc_end = omp_get_wtime();
    /* L2 misses */
    rdpmc(0, eax, edx);
    int64_t end_naive_cnt = ((int64_t)eax) | ((int64_t)edx << 32);
    // printf("L2 Cache Miss: %ld \n", end_naive_cnt - start_cnt);
    // printf("Multiplication 1 finished in %6.2f s\n",
    //        ((float)t)/CLOCKS_PER_SEC);
    printf("%ld \n", end_naive_cnt - start_cnt);
    printf("%.6f \n",
           ((float)t)/CLOCKS_PER_SEC);
    printf("%.6f \n",
           wc_end-wc_start); 

    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    wc_start = omp_get_wtime();
    t = clock();

    /* TODO: count L2 cache misses for the next block using RDPMC */
    rdpmc(0, eax, edx);
    int64_t start_faster_cnt  = ((int64_t)eax) | ((int64_t)edx << 32);

    /* perform fast(er) multiplication */
    for (uint32_t k=0; k<N; ++k)
        for (uint32_t i=0; i<N; ++i)         /* line   */
            for (uint32_t j=0; j<N; ++j)     /* column */
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    /* clock delta */
    t = clock() - t;
    wc_end = omp_get_wtime();
    rdpmc(0, eax, edx);
    int64_t end_faster_cnt = ((int64_t)eax) | ((int64_t)edx << 32);
    printf("%ld \n", end_faster_cnt - start_faster_cnt);
    printf("%.6f \n",
           ((float)t)/CLOCKS_PER_SEC); 
    printf("%.6f \n",
           wc_end-wc_start); 
    return 0;
}