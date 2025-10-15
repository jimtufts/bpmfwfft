#ifndef NNLS_CUH
#define NNLS_CUH

#include <cuda_runtime.h>

// Utility macros
#define MIN(a, b) ((a)>(b)?(b):(a))
#define MAX(a, b) ((a)>(b)?(a):(b))
#define SIGN(a) ((a)>0?1:-1)

// Index macros
#define X_I(a, b) ((a)*(N)+(b))
#define B_I(a, b) ((a)*(M)+(b))
#define R_I(a, b, c) ((a)*((M)*(M))+(b)*(M)+(c))
#define A_I(a, b, c) ((a)*((N)*(M))+(b)*(M)+(c))

// Function prototypes
__global__ void NNLS_MGS_GR_512(double *d_A, double *d_At, double *d_x, double *d_b,
                                double *d_R, int *nIters, int *lsIters, 
                                int NSYS, int N, int M,
				int MAX_ITER_LS, int MAX_ITER_NNLS,
                                double TOL_TERMINATION, double TOL_0);

// template utility functions 
template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ double reduce(double *smem, unsigned int tID);

template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ int maxIndex(double *smem, double *smemC, unsigned int tID);

template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ double norml2(double *smem, double v, unsigned int tID);

template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ double min(double *smem, unsigned int tID);


#endif // NNLS_CUH
