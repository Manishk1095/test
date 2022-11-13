#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <math.h>
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
    printf("\t./mat_mul <N> <b>\n");
    return -1;
}

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


/*
 *  print_matrix - if you need convincing that it works just fine
 *      @N: square matrix size
 *      @m: pointer to matrix
 */
void
print_matrix(uint32_t N, int64_t *m)
{
    for (uint32_t i=0; i<N; i++) {
        for (uint32_t j=0; j<N; j++)
            printf("%3ld ", m[i*N + j]);
        printf("\n");
    }
}

/*
 *  block matrix multiplication
 *      @jj: block col index
 *      @numBlk: number of blocks
 *      @b: block size
 *      @N: number of rows and cols
 *      @m1: pointer to matrix 1
 *      @m2: pointer to matrix 2
 *      @rBlk: resultant matrix
 */
void
blk_mat_mul(uint32_t ii, uint32_t jj, uint32_t kk, int b, uint32_t N, int64_t *m1, int64_t *m2, int64_t *rBlk)
{
    uint32_t min_i = min(((ii+1)*b), N);
    uint32_t min_j = min(((jj+1)*b), N);
    uint32_t min_k = min(((kk+1)*b), N);
    for (uint32_t k=kk*b; k<min_k; k++)
        for (uint32_t i=ii*b; i<min_i; i++)         /* line   */
            for (uint32_t j=jj*b; j<min_j; j++)     /* column */
                rBlk[i*N + j] += m1[i*N + k] * m2[k*N + j]; 
}


void
naive_blk_mat_mul(uint32_t ii, uint32_t jj, uint32_t kk, int b, uint32_t N, int64_t *m1, int64_t *m2, int64_t *rBlk)
{
    uint32_t min_i = min(((ii+1)*b), N);
    uint32_t min_j = min(((jj+1)*b), N);
    uint32_t min_k = min(((kk+1)*b), N);
    for (uint32_t i=ii*b; i<min_i; i++)         /* line   */
        for (uint32_t j=jj*b; j<min_j; j++)     /* column */
            for (uint32_t k=kk*b; k<min_k; k++)
                rBlk[i*N + j] += m1[i*N + k] * m2[k*N + j]; 
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
    // int b   = atoi(argv[2]);
    int64_t  *m1 = malloc(N * N * sizeof(int64_t));
    int64_t  *m2 = malloc(N * N * sizeof(int64_t));
    int64_t  *r  = malloc(N * N * sizeof(int64_t));
    uint32_t eax, edx;

    /* initialize matrices */
    for (uint32_t i=0; i<N*N; ++i) {
        m1[i] = i;
        m2[i] = i;
    }

    int64_t  *rTruth  = malloc(N * N * sizeof(int64_t));
    memset(rTruth, 0, N * N * sizeof(int64_t));
    for (uint32_t k=0; k<N; k++)
        for (uint32_t i=0; i<N; i++)         /* line   */
            for (uint32_t j=0; j<N; j++)     /* column */
                rTruth[i*N + j] += m1[i*N + k] * m2[k*N + j];


    //////////////////////////////////////////////////////////////////////////////////////////
    // OMP2 (IJK)
    //////////////////////////////////////////////////////////////////////////////////////////
    printf("%d\n", omp_get_num_procs());
    omp_set_num_threads(omp_get_num_procs());
    printf("OMP (Naive)\n");
    /* result matrix clear; clock init */
    int64_t  *rBlk  = malloc(N * N * sizeof(int64_t));
    memset(rBlk, 0, N * N * sizeof(int64_t));
    double wc_start, wc_end;
    wc_start = omp_get_wtime();
    t = clock();

    /* perform parallel naive multiplication */
    int64_t total;

    #pragma omp parallel for reduction (+: rBlk[:(N*N)])
    for (uint32_t i=0; i<N; i++)
        for (uint32_t j=0; j<N; j++){
            for (uint32_t k=0; k<N; k++){
                rBlk[i*N + j] += m1[i*N + k] * m2[k*N + j];
            }
        }

    /* clock delta */
    t = clock() - t;
    wc_end = omp_get_wtime();
    printf("%.6f \n",
           ((float)t)/CLOCKS_PER_SEC);
    printf("%.6f \n",
           wc_end-wc_start);

    //////////////////////////////////////////////////////////////////////////////////////////
    // OMP2 (KIJ)
    //////////////////////////////////////////////////////////////////////////////////////////

    printf("OMP (Faster)\n");
    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    wc_start = omp_get_wtime();
    t = clock();
    /* perform parallel faster multiplication */
    #pragma omp parallel for reduction (+: r[:(N*N)])
    for (uint32_t k=0; k<N; k++)
        for (uint32_t i=0; i<N; i++)         /* line   */
            for (uint32_t j=0; j<N; j++) {   /* column */
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];
            }

    /* clock delta */
    t = clock() - t;
    wc_end = omp_get_wtime();
    printf("%.6f \n",
           ((float)t)/CLOCKS_PER_SEC);
    printf("%.6f \n",
           wc_end-wc_start);


    /* check matrix results*/
    for (int i=0; i<N*N; i++){
        if (rBlk[i] != rTruth[i]) {
            printf("rBlk Matrix not same @ %d\n", i);
            break;
        }
    }

    /* check matrix results*/
    for (int i=0; i<N*N; i++){
        if (r[i] != rTruth[i]) {
            printf("r Matrix not same @ %d\n", i);
            break;
        }
    }
    // printf("\n");
    // print_matrix(N, rTruth);
    // printf("\n");
    // print_matrix(N, r);
    // printf("\n");
    // print_matrix(N, rBlk);
    // printf("\n");
    return 0;
}
