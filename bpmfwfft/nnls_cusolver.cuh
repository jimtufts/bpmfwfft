#ifndef NNLS_SMALL_CUSOLVER_CUH
#define NNLS_SMALL_CUSOLVER_CUH

// NNLS solver using cuSOLVER QR factorization
// More numerically stable than normal equations
extern "C" void cuda_nnls_10x10_cusolver(
    const double* d_A,           // [num_systems, 10, 10] input matrices
    const double* d_b,           // [num_systems, 10] input RHS vectors
    double* d_x,                 // [num_systems, 10] output solutions
    int num_systems,             // Number of systems to solve
    int max_outer_iter           // Maximum outer iterations
);

#endif // NNLS_SMALL_CUSOLVER_CUH
