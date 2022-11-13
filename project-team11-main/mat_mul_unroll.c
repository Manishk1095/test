#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */

#define rdpmc(ecx, eax, edx)    \
    asm volatile (              \
        "rdpmc"                 \
        : "=a"(eax),            \
          "=d"(edx)             \
        : "c"(ecx))

#define VERIFY 0

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

void
verify_matrix(uint32_t N, int64_t *m1, int64_t *m2, int64_t *r)
{
    int64_t  *v  = malloc(N * N * sizeof(int64_t));
    for (uint32_t k=0; k<N; ++k)
        for (uint32_t i=0; i<N; ++i)
            for (uint32_t j=0; j<N; ++j)
                v[i*N + j] += m1[i*N + k] * m2[k*N + j];

    int valid = 1;
    for (uint32_t i=0; i<N*N; i++) {
        if (v[i] != r[i]) {
            valid = 0;
            break;
        }
    }

    if (!valid) {
        printf("Matrix verification failed\n");
    } else {
        printf("Matrix verification ok\n");
    }

    free(v);
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

    /* initialize matrices */
    for (uint32_t i=0; i<N*N; ++i) {
        m1[i] = i;
        m2[i] = i;
    }

    uint32_t edx=0, eax=0;
    uint32_t n_edx=0, n_eax=0;
    uint64_t start, end;

    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();

    /* TODO: count L2 cache misses for the next block using RDPMC */
    rdpmc(0, eax, edx);

    /* perform fast(er) multiplication */
    for (uint32_t i=0; i<N; ++i)
        for (uint32_t j=0; j<N; ++j)         /* line   */
            for (uint32_t k=0; k<N; k+=8) {     /* column */
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];
                r[i*N + j] += m1[i*N + (k+1)] * m2[(k+1)*N + j];
                r[i*N + j] += m1[i*N + (k+2)] * m2[(k+2)*N + j];
                r[i*N + j] += m1[i*N + (k+3)] * m2[(k+3)*N + j];
                r[i*N + j] += m1[i*N + (k+4)] * m2[(k+4)*N + j];
                r[i*N + j] += m1[i*N + (k+5)] * m2[(k+5)*N + j];
                r[i*N + j] += m1[i*N + (k+6)] * m2[(k+6)*N + j];
                r[i*N + j] += m1[i*N + (k+7)] * m2[(k+7)*N + j];
            }

    rdpmc(0, n_eax, n_edx);
    /* clock delta */
    t = clock() - t;

    start = (uint64_t) edx << 32 | eax;
    end = (uint64_t) n_edx << 32 | n_eax;

    printf("%d %d\n", edx, eax);
    printf("%d %d\n", n_edx, n_eax);
    printf("%lu %lu\n", start, end);
    printf("l2 cache miss: %lu\n", end-start);

    printf("Multiplication 2 finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC);

#if VERIFY
    verify_matrix(N, m1, m2, r);
#endif

    return 0;
}
