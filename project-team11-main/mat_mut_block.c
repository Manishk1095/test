#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <math.h>

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
 *  block matrix multiplication - if you need convincing that it works just fine
 *      @ii: block row index 
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
                rBlk[i*N + j] += m1[i*N + k] * m2[k*N + j]; // printf("(%d, %d, %d)\t(%d, %d, %d)\n", ii, jj, kk, i, j, k); // 
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
                rBlk[i*N + j] += m1[i*N + k] * m2[k*N + j]; // printf("(%d, %d, %d)\t(%d, %d, %d)\n", ii, jj, kk, i, j, k); // 
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
    int b   = atoi(argv[2]);
    int64_t  *m1 = malloc(N * N * sizeof(int64_t));
    int64_t  *m2 = malloc(N * N * sizeof(int64_t));
    int64_t  *r  = malloc(N * N * sizeof(int64_t));

    /* initialize matrices */
    for (uint32_t i=0; i<N*N; ++i) {
        m1[i] = i;
        m2[i] = i;
    }
    //////////////////////////////////////////////////////////////////////////////////////////
    // Naive
    //////////////////////////////////////////////////////////////////////////////////////////
    printf("%d\n", b);
    printf("Naive\n");
    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();

    /* TODO: count L2 cache misses for the next block using RDPMC */
    uint32_t eax, edx;
    rdpmc(0, eax, edx);
    int64_t start_cnt = ((int64_t)eax) | ((int64_t)edx << 32);
    /* perform slow multiplication */
    for (uint32_t i=0; i<N; i++)             /* line   */
        for (uint32_t j=0; j<N; j++)         /* column */
            for (uint32_t k=0; k<N; k++)
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    /* clock delta */
    t = clock() - t;

    /* L2 misses */
    rdpmc(0, eax, edx);
    int64_t end_cnt = ((int64_t)eax) | ((int64_t)edx << 32);
    printf("L2 Cache Miss: %ld \n", end_cnt - start_cnt);
    printf("Multiplication finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC);


    //////////////////////////////////////////////////////////////////////////////////////////
    // Faster
    //////////////////////////////////////////////////////////////////////////////////////////
    printf("\nFaster\n");
    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();

    /* TODO: count L2 cache misses for the next block using RDPMC */
    rdpmc(0, eax, edx);
    start_cnt  = ((int64_t)eax) | ((int64_t)edx << 32);

    /* perform fast(er) multiplication */
    for (uint32_t k=0; k<N; k++)
        for (uint32_t i=0; i<N; i++)         /* line   */
            for (uint32_t j=0; j<N; j++)     /* column */
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    /* clock delta */
    t = clock() - t;

    rdpmc(0, eax, edx);
    end_cnt = ((int64_t)eax) | ((int64_t)edx << 32);
    printf("L2 Cache Miss: %ld \n", end_cnt - start_cnt);
    printf("Multiplication finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC); 


    //////////////////////////////////////////////////////////////////////////////////////////
    // Block
    //////////////////////////////////////////////////////////////////////////////////////////
    printf("\nBlock\n");

    /* assumes rows/cols are perfectly divisible into numBlk*/
    double L2_cache_size = 256.0 * 1024;
    int b1 = (int) floor(sqrt(L2_cache_size) / sizeof(int64_t) / 3); // from https://marek.ai/matrix-multiplication-on-cpu.html
    // int b = (int) floor(sqrt(L2_cache_size / 3) / sizeof(int64_t)); 
    int c = (int) floor(sqrt(L2_cache_size / 3 / sizeof(int64_t))); 
    int numBlk = 1.0 * (N + b - 1) / b;
    printf("b: %d, b1: %d, c: %d, numBlk: %d\n", b, b1, c, numBlk);

    /* result matrix and clock init */
    int64_t  *rBlk  = malloc(N * N * sizeof(int64_t));
    t = clock();
    rdpmc(0, eax, edx);
    start_cnt  = ((int64_t)eax) | ((int64_t)edx << 32);

    /* perform block multiplication */
    for (uint32_t ii=0; ii<numBlk; ii++)
        for (uint32_t jj=0; jj<numBlk; jj++) 
            for (uint32_t kk=0; kk<numBlk; kk++) /*Block multiplications below*/
                blk_mat_mul(ii, jj, kk, b, N, m1, m2, rBlk);

    /* clock delta */
    t = clock() - t;
    rdpmc(0, eax, edx);
    end_cnt = ((int64_t)eax) | ((int64_t)edx << 32);
    printf("L2 Cache Miss: %ld \n", end_cnt - start_cnt);
    printf("Multiplication finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC); 

    //////////////////////////////////////////////////////////////////////////////////////////////////////////    

    /* assumes rows/cols are perfectly divisible into numBlk*/
    printf("\n");

    /* result matrix and clock init */
    memset(rBlk, 0, N * N * sizeof(int64_t));
    t = clock();
    rdpmc(0, eax, edx);
    start_cnt  = ((int64_t)eax) | ((int64_t)edx << 32);

    /* perform block multiplication */
    for (uint32_t ii=0; ii<numBlk; ii++)
        for (uint32_t jj=0; jj<numBlk; jj++) 
            for (uint32_t kk=0; kk<numBlk; kk++) /*Block multiplications below*/
                naive_blk_mat_mul(ii, jj, kk, b, N, m1, m2, rBlk);

    /* clock delta */
    t = clock() - t;
    rdpmc(0, eax, edx);
    end_cnt = ((int64_t)eax) | ((int64_t)edx << 32);
    printf("L2 Cache Miss: %ld \n", end_cnt - start_cnt);
    printf("Multiplication finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC); 

    /* check matrix results*/
    for (int i=0; i<N*N; i++){
        if (r[i] != rBlk[i]) 
            printf("Matrix not same @ %d\n", i);
            break;
    }
    printf("\n");
    // print_matrix(N, m1);
    // printf("\n");
    // print_matrix(N, r);
    // printf("\n");
    // print_matrix(N, rBlk);
    // printf("\n");
    return 0;
}
