#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <pthread.h>

#define rdpmc(ecx, eax, edx)    \
    asm volatile (              \
        "rdpmc"                 \
        : "=a"(eax),            \
          "=d"(edx)             \
        : "c"(ecx))

#define N_THREADS 4
#define VERIFY 0

struct targ {
    uint32_t N;
    int64_t *m1;
    int64_t *m2;
    int id;
};

struct targ targs[N_THREADS];

void *worker(void *args)
{
    struct targ *tdata = (struct targ *) args;
    //printf("%d\n", tdata->id);

    uint32_t N = tdata->N;
    uint32_t block = N/N_THREADS;
    int64_t *m1 = tdata->m1;
    int64_t *m2 = tdata->m2;
    int64_t *r  = malloc(N * block * sizeof(int64_t));

    uint32_t start = tdata->id * block;

    //printf("%d %d\n", tdata->id, start);

    for (uint32_t i=0;i<block;i++) {
        for (uint32_t j=0;j<N;j++) {
            for (uint32_t k=0;k<N;k++) {
                r[i*N + j] += m1[(i+start)*N + k] * m2[k*N + j];
            }
        }
    }

    return (void *)r;
}

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
    int64_t *v  = malloc(N * N * sizeof(int64_t));
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

    pthread_t pthreads[N_THREADS];
    for (int i=0;i<N_THREADS;i++) {
        targs[i].m1 = m1;
        targs[i].m2 = m2;
        targs[i].N = N;
        targs[i].id = i;

        pthread_create(&pthreads[i], NULL, worker, (void *)&targs[i]);
    }

    for (int i=0;i<N_THREADS;i++) {
        int64_t *res;
        pthread_join(pthreads[i], (void **)&res);

        int block = N/N_THREADS;
        for (int j=0;j<N*block;j++) {
            r[i*N*block+j] = res[j];
        }

        free(res);
    }


    t = clock() - t;

    printf("Multiplication 2 finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC);

#if VERIFY
    verify_matrix(N, m1, m2, r);
#endif

    return 0;
}
