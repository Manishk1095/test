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
    //////////////////////////////////////////////////////////////////////////////////////////
    // Naive
    //////////////////////////////////////////////////////////////////////////////////////////
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
    // transpose
    //////////////////////////////////////////////////////////////////////////////////////////
    printf("\nTranspose\n");

    /* result matrix and clock init */
    int64_t  *rBlk  = malloc(N * N * sizeof(int64_t));
    int64_t  *m2_t  = malloc(N * N * sizeof(int64_t));
    t = clock();
    rdpmc(0, eax, edx);
    start_cnt  = ((int64_t)eax) | ((int64_t)edx << 32);

    /* Transpose M2 */
    for (uint32_t i=0; i<N; i++)         /* line   */
        for (uint32_t j=0; j<N; j++)     /* column */
            m2_t[j*N + i] = m2[i*N + j];

    /* perform transpose multiplication */
    for (uint32_t i=0; i<N; i++)         /* line   */
        for (uint32_t j=0; j<N; j++)     /* column */
            for (uint32_t k=0; k<N; k++)
                rBlk[i*N + j] += m1[i*N + k] * m2_t[j*N + k];

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
