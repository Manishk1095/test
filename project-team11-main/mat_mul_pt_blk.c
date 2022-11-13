#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <pthread.h>
#include <omp.h>        /* for timing functions */
#include <math.h>

// Number of threads have to be a multiple of 2
// Block ratio is sqrt(N_THREADS), which is how many sub matrices per row/col
#define N_THREADS 8
#define BLOCK_RATIO 2

#define VERIFY 1

struct targ {
    uint32_t N;
    int64_t *m1;
    int64_t *m2;
    int64_t *r;

    uint32_t i;
    uint32_t j;
    uint32_t nrow;
    uint32_t ncol;
};

struct targ targs[N_THREADS];

void *worker(void *args)
{
    struct targ *tdata = (struct targ *) args;
    //printf("%d\n", tdata->id);

    uint32_t N = tdata->N;
    uint32_t block_size = N/BLOCK_RATIO;

    int64_t *m1 = tdata->m1;
    int64_t *m2 = tdata->m2;
    int64_t *r  = malloc(block_size * block_size * sizeof(int64_t));
    uint32_t nrow = tdata->nrow;
    uint32_t ncol = tdata->ncol;
    uint32_t ii = tdata->i;
    uint32_t jj = tdata->j;
    uint32_t min_i = min(((ii+1)*nrow), N);
    uint32_t min_j = min(((jj+1)*ncol), N);
    // uint32_t min_k = min(((kk+1)*b), N); 

    uint32_t start_i = (tdata->i % tdata->nrow) * block_size;
    uint32_t start_j = tdata->j * tdata->ncol;

    //printf("%d %d %d\n", tdata->id, start_i, start_j);

    // Matrix multiplication
    for (uint32_t i=ii*nrow;i<min_i;i++) {
        for (uint32_t j=ii*ncol;j<min_j;j++) {
            for (uint32_t k=0;k<N;k++) {
                r[i*block_size + j] += m1[(i+start_i)*N + k] * m2[k*N + (j+start_j)];
            }
        }
    }

    // Copy to final array
    for (uint32_t i=0;i<block_size;i++) {
        for (uint32_t j=0;j<block_size;j++) {
            tdata->r[(start_i+i)*N+(start_j+j)] = r[i*block_size+j];
        }
    }

    free(r);
    return NULL;
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

    double wc_start, wc_end;
    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    wc_start = omp_get_wtime();
    t = clock();
    int ncolblk = floor(sqrt(N_THREADS));
    int nrowblk = N_THREADS/ncolblk;
    int nrow = 1.0 * (N + nrowblk - 1) / nrowblk;
    int ncol = 1.0 * (N + ncolblk - 1) / ncolblk;
    pthread_t pthreads[N_THREADS];

    for (int i=0;i<nrow;i++) {
        for (int j=0; j<ncol; j++) {
            targs[i*ncol+ j].m1 = m1;
            targs[i*ncol+ j].m2 = m2;
            targs[i*ncol+ j].r = r;
            targs[i*ncol+ j].N = N;
            targs[i*ncol+ j].i = i;
            targs[i*ncol+ j].j = j;
            targs[i*ncol+ j].nrow = nrow;
            targs[i*ncol+ j].ncol = ncol;
            pthread_create(&pthreads[i*ncol + j], NULL, worker, (void *)&targs[i*ncol + j]);
        }
    }

    for (int t=0;t<N_THREADS;t++) {
        pthread_join(pthreads[t], NULL);
    }


    t = clock() - t;
    wc_end = omp_get_wtime();

    printf("Multiplication finished in %6.2f s %.6f s\n",
           ((float)t)/CLOCKS_PER_SEC,
           wc_end-wc_start);

#if VERIFY
    verify_matrix(N, m1, m2, r);
#endif

    return 0;
}
