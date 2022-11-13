#define _GNU_SOURCE

#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <pthread.h>
#include <omp.h>        /* for timing functions */
#include <errno.h>

#define handle_error_en(en, msg) \
        do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

#define N_THREADS 8
#define BLOCK_RATIO_W 4
#define BLOCK_RATIO_H 2

#define VERIFY 1
#define THREAD_AFFINITY 1
#define THREAD_AFFINITY_CORE_OFFSET 0

struct targ {
    uint32_t N;
    int64_t *m1;
    int64_t *m2;
    int64_t *r;

    uint32_t id;
};

struct targ targs[N_THREADS];

void *worker(void *args)
{
    struct targ *tdata = (struct targ *) args;
    //printf("%d\n", tdata->id);

    uint32_t N = tdata->N;
    uint32_t block_size_w = N/BLOCK_RATIO_W;
    uint32_t block_size_h = N/BLOCK_RATIO_H;

    int64_t *m1 = malloc(block_size_h * N * sizeof(int64_t));
    int64_t *m2 = malloc(block_size_w * N * sizeof(int64_t));
    int64_t *r  = malloc(block_size_w * block_size_h * sizeof(int64_t));

    memset(r, 0, block_size_w * block_size_h * sizeof(int64_t));

    uint32_t start_i = (tdata->id / BLOCK_RATIO_W) * block_size_h;
    uint32_t start_j = (tdata->id % BLOCK_RATIO_W) * block_size_w;

    // Copy out data that is needed
    // This also implicitly transpose m2
    for (uint32_t i=0;i<block_size_h;i++) {
        for (uint32_t k=0;k<N;k++) {
            m1[i*N+k] = tdata->m1[(start_i+i)*N+k];
        }
    }

    for (uint32_t j=0;j<block_size_w;j++) {
        for (uint32_t k=0;k<N;k++) {
            m2[j*N+k] = tdata->m2[k*N + (start_j+j)];
        }
    }

    //printf("%d %d %d\n", tdata->id, start_i, start_j);
    //printf("%d [%d %d] [%d %d]\n", tdata->id, start_i, start_i+block_size_h, start_j, start_j+block_size_w);

    // Matrix multiplication
    for (uint32_t i=0;i<block_size_h;i++) {
        for (uint32_t j=0;j<block_size_w;j++) {
            for (uint32_t k=0;k<N;k++) {
                r[i*block_size_w + j] += m1[i*N + k] * m2[j*N + k];
            }
        }
    }

    // Copy to final array
    for (uint32_t i=0;i<block_size_h;i++) {
        for (uint32_t j=0;j<block_size_w;j++) {
            tdata->r[(start_i+i)*N+(start_j+j)] = r[i*block_size_w+j];
        }
    }

    free(m1);
    free(m2);
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
    } 
    // else {
    //     printf("Matrix verification ok\n");
    // }

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
    if (argc != 3)
        return usage();

    /* allocate space for matrices */
    clock_t t;
    uint32_t N   = atoi(argv[1]);
    uint32_t VERIFY  = atoi(argv[2]);
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

    pthread_t pthreads[N_THREADS];
    for (int i=0;i<N_THREADS;i++) {
        targs[i].m1 = m1;
        targs[i].m2 = m2;
        targs[i].r = r;
        targs[i].N = N;
        targs[i].id = i;

#if THREAD_AFFINITY
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i+THREAD_AFFINITY_CORE_OFFSET, &cpuset);
#endif

        pthread_create(&pthreads[i], NULL, worker, (void *)&targs[i]);

#if THREAD_AFFINITY
    int s = pthread_setaffinity_np(pthreads[i], sizeof(cpu_set_t), &cpuset);
    if (s != 0)
        handle_error_en(s, "pthread_set_affinity_np, s");
#endif
    }

    for (int t=0;t<N_THREADS;t++) {
        pthread_join(pthreads[t], NULL);
    }


    t = clock() - t;
    wc_end = omp_get_wtime();

    printf("precopy\n%d\n%.6f\n%.6f\n",
            N, 
           ((float)t)/CLOCKS_PER_SEC,
           wc_end-wc_start);
           
    if (VERIFY)
        verify_matrix(N, m1, m2, r);

    return 0;
}
