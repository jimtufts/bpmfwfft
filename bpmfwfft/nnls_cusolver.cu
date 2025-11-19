#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <float.h>
#include <cmath>
#include <cstdio>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t err = call; \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            printf("cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            return; \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            return; \
        } \
    } while(0)

// Kernel: Compute dual variables w = A^T * (b - A*x)
__global__ void compute_dual_kernel(
    const double* A_global,  // [num_systems, 10, 10]
    const double* b_global,  // [num_systems, 10]
    const double* x_global,  // [num_systems, 10]
    double* w_global,        // [num_systems, 10] output
    const bool* active,      // [num_systems, 10]
    int num_systems
) {
    int sys_idx = blockIdx.x;
    if (sys_idx >= num_systems) return;

    int lane_id = threadIdx.x;

    const double* A = A_global + sys_idx * 100;
    const double* b = b_global + sys_idx * 10;
    const double* x = x_global + sys_idx * 10;
    double* w = w_global + sys_idx * 10;
    const bool* act = active + sys_idx * 10;

    __shared__ double residual[10];  // b - A*x

    // Compute residual = b - A*x
    if (lane_id < 10) {
        double Ax = 0.0;
        for (int j = 0; j < 10; j++) {
            Ax += A[lane_id * 10 + j] * x[j];
        }
        residual[lane_id] = b[lane_id] - Ax;
    }
    __syncthreads();

    // Compute w = A^T * residual (only for inactive variables)
    if (lane_id < 10) {
        if (!act[lane_id]) {
            double sum = 0.0;
            for (int i = 0; i < 10; i++) {
                sum += A[i * 10 + lane_id] * residual[i];
            }
            w[lane_id] = sum;
        } else {
            w[lane_id] = -DBL_MAX;  // Active variables don't participate
        }
    }
}

// Kernel: Find maximum dual variable for each system
__global__ void find_max_dual_kernel(
    const double* w_global,   // [num_systems, 10]
    int* max_idx_global,      // [num_systems] output
    double* max_val_global,   // [num_systems] output
    bool* converged_global,   // [num_systems] output
    int num_systems,
    double tolerance
) {
    int sys_idx = blockIdx.x;
    if (sys_idx >= num_systems) return;

    int lane_id = threadIdx.x;
    const double* w = w_global + sys_idx * 10;

    __shared__ double max_val;
    __shared__ int max_idx;

    // Each thread finds max in its subset
    double local_max = -DBL_MAX;
    int local_idx = -1;

    for (int i = lane_id; i < 10; i += 32) {
        if (w[i] > local_max) {
            local_max = w[i];
            local_idx = i;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        double other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    if (lane_id == 0) {
        max_val = local_max;
        max_idx = local_idx;
        max_val_global[sys_idx] = max_val;
        max_idx_global[sys_idx] = max_idx;
        converged_global[sys_idx] = (max_val <= tolerance);
    }
}

// Kernel: Add variable to active set
__global__ void add_to_active_set_kernel(
    bool* active,           // [num_systems, 10]
    const int* max_idx,     // [num_systems]
    const bool* converged,  // [num_systems]
    int num_systems
) {
    int sys_idx = blockIdx.x;
    if (sys_idx >= num_systems) return;
    if (converged[sys_idx]) return;

    int idx = max_idx[sys_idx];
    if (idx >= 0 && idx < 10) {
        active[sys_idx * 10 + idx] = true;
    }
}

// Kernel: Build reduced systems for all active sets
__global__ void build_reduced_systems_kernel(
    const double* A_global,      // [num_systems, 10, 10]
    const double* b_global,      // [num_systems, 10]
    const bool* active,          // [num_systems, 10]
    double* A_reduced_global,    // [num_systems, 10, 10] - will store 10 x active_count
    double* b_reduced_global,    // [num_systems, 10]
    int* active_counts,          // [num_systems] output
    int* active_maps,            // [num_systems, 10] maps reduced idx to full idx
    int num_systems
) {
    int sys_idx = blockIdx.x;
    if (sys_idx >= num_systems) return;

    int lane_id = threadIdx.x;

    const double* A = A_global + sys_idx * 100;
    const double* b = b_global + sys_idx * 10;
    const bool* act = active + sys_idx * 10;
    double* A_red = A_reduced_global + sys_idx * 100;
    double* b_red = b_reduced_global + sys_idx * 10;
    int* act_count_ptr = active_counts + sys_idx;
    int* act_map = active_maps + sys_idx * 10;

    __shared__ int active_count;
    __shared__ int local_map[10];

    // Build active map
    if (lane_id == 0) {
        active_count = 0;
        for (int i = 0; i < 10; i++) {
            if (act[i]) {
                local_map[active_count++] = i;
            }
        }
        *act_count_ptr = active_count;
    }
    __syncthreads();

    // Copy active map to global
    if (lane_id < 10) {
        act_map[lane_id] = (lane_id < active_count) ? local_map[lane_id] : -1;
    }

    if (active_count == 0) return;

    // Build reduced system: 10 rows (all equations) x active_count columns
    for (int elem = lane_id; elem < 10 * active_count; elem += 32) {
        int row = elem / active_count;
        int col = elem % active_count;
        int orig_col = local_map[col];
        A_red[row * active_count + col] = A[row * 10 + orig_col];
    }

    // Copy b (all 10 equations)
    if (lane_id < 10) {
        b_red[lane_id] = b[lane_id];
    }
}

// Kernel: Check for negatives in active set after solving
__global__ void check_negatives_kernel(
    const double* x_global,     // [num_systems, 10]
    const bool* active,         // [num_systems, 10]
    bool* has_negatives,        // [num_systems] output
    int num_systems,
    double tolerance
) {
    int sys_idx = blockIdx.x;
    if (sys_idx >= num_systems) return;

    int lane_id = threadIdx.x;
    const double* x = x_global + sys_idx * 10;
    const bool* act = active + sys_idx * 10;

    bool local_has_neg = false;
    for (int i = lane_id; i < 10; i += 32) {
        if (act[i] && x[i] < -tolerance) {
            local_has_neg = true;
        }
    }

    // Warp reduction
    unsigned int ballot = __ballot_sync(0xffffffff, local_has_neg);
    if (lane_id == 0) {
        has_negatives[sys_idx] = (ballot != 0);
    }
}

// Kernel: Apply line search and remove variables that hit zero
__global__ void line_search_kernel(
    double* x_global,           // [num_systems, 10]
    const double* x_old_global, // [num_systems, 10]
    bool* active,               // [num_systems, 10]
    const bool* has_negatives,  // [num_systems]
    int num_systems,
    double tolerance
) {
    int sys_idx = blockIdx.x;
    if (sys_idx >= num_systems) return;
    if (!has_negatives[sys_idx]) return;

    int lane_id = threadIdx.x;
    double* x = x_global + sys_idx * 10;
    const double* x_old = x_old_global + sys_idx * 10;
    bool* act = active + sys_idx * 10;

    __shared__ double alpha_min;

    // Find minimum alpha
    if (lane_id == 0) {
        alpha_min = 1.0;
        for (int i = 0; i < 10; i++) {
            if (act[i] && x[i] < -tolerance) {
                double alpha = x_old[i] / (x_old[i] - x[i]);
                if (alpha >= 0.0 && alpha < alpha_min) {
                    alpha_min = alpha;
                }
            }
        }
    }
    __syncthreads();

    // Apply line search and remove zeros
    if (lane_id < 10 && act[lane_id]) {
        x[lane_id] = x_old[lane_id] + alpha_min * (x[lane_id] - x_old[lane_id]);
        if (fabs(x[lane_id]) < tolerance) {
            act[lane_id] = false;
            x[lane_id] = 0.0;
        }
    }
}

// Kernel: Map reduced solution back to full x
__global__ void map_solution_kernel(
    double* x_global,              // [num_systems, 10]
    const double* x_reduced_global, // [num_systems, 10] - solution in b_reduced
    const int* active_maps,        // [num_systems, 10]
    const int* active_counts,      // [num_systems]
    int num_systems
) {
    int sys_idx = blockIdx.x;
    if (sys_idx >= num_systems) return;

    int lane_id = threadIdx.x;
    double* x = x_global + sys_idx * 10;
    const double* x_red = x_reduced_global + sys_idx * 10;
    const int* map = active_maps + sys_idx * 10;
    int count = active_counts[sys_idx];

    // Zero out all x first
    if (lane_id < 10) {
        x[lane_id] = 0.0;
    }
    __syncthreads();

    // Map reduced solution back
    if (lane_id < count) {
        int orig_idx = map[lane_id];
        if (orig_idx >= 0 && orig_idx < 10) {
            x[orig_idx] = x_red[lane_id];
        }
    }
}

// Host function: NNLS solver using cuSOLVER QR
extern "C" void cuda_nnls_10x10_cusolver(
    const double* d_A,
    const double* d_b,
    double* d_x,
    int num_systems,
    int max_outer_iter
) {
    const double TOLERANCE = 1e-5;
    const int MAX_INNER_ITER = 50;

    // Allocate device memory
    bool *d_active, *d_converged, *d_has_negatives;
    double *d_w, *d_max_val, *d_x_old;
    double *d_A_reduced, *d_b_reduced, *d_tau;
    int *d_max_idx, *d_active_counts, *d_active_maps, *d_info;

    CHECK_CUDA(cudaMalloc(&d_active, num_systems * 10 * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_converged, num_systems * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_has_negatives, num_systems * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_w, num_systems * 10 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_max_val, num_systems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_max_idx, num_systems * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x_old, num_systems * 10 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_A_reduced, num_systems * 100 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b_reduced, num_systems * 10 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_tau, num_systems * 10 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_active_counts, num_systems * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_active_maps, num_systems * 10 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_info, num_systems * sizeof(int)));

    // Initialize
    CHECK_CUDA(cudaMemset(d_active, 0, num_systems * 10 * sizeof(bool)));
    CHECK_CUDA(cudaMemset(d_x, 0, num_systems * 10 * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_converged, 0, num_systems * sizeof(bool)));

    // Create cuSOLVER and cuBLAS handles
    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    printf("Starting cuSOLVER-based NNLS for %d systems\n", num_systems);

    // Outer loop
    for (int iter = 0; iter < max_outer_iter; iter++) {
        // Compute dual variables
        compute_dual_kernel<<<num_systems, 32>>>(d_A, d_b, d_x, d_w, d_active, num_systems);
        CHECK_CUDA(cudaGetLastError());

        // Find maximum dual
        find_max_dual_kernel<<<num_systems, 32>>>(d_w, d_max_idx, d_max_val, d_converged, num_systems, TOLERANCE);
        CHECK_CUDA(cudaGetLastError());

        // Check convergence
        bool h_all_converged = true;
        bool* h_converged = new bool[num_systems];
        CHECK_CUDA(cudaMemcpy(h_converged, d_converged, num_systems * sizeof(bool), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_systems; i++) {
            if (!h_converged[i]) {
                h_all_converged = false;
                break;
            }
        }
        delete[] h_converged;

        if (h_all_converged) {
            printf("All systems converged at iteration %d\n", iter);
            break;
        }

        // Add variable to active set
        add_to_active_set_kernel<<<num_systems, 32>>>(d_active, d_max_idx, d_converged, num_systems);
        CHECK_CUDA(cudaGetLastError());

        // Inner loop: solve with current active set
        for (int inner_iter = 0; inner_iter < MAX_INNER_ITER; inner_iter++) {
            // Save old x
            CHECK_CUDA(cudaMemcpy(d_x_old, d_x, num_systems * 10 * sizeof(double), cudaMemcpyDeviceToDevice));

            // Build reduced systems
            build_reduced_systems_kernel<<<num_systems, 32>>>(
                d_A, d_b, d_active, d_A_reduced, d_b_reduced,
                d_active_counts, d_active_maps, num_systems
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            // Get active counts to host for QR
            int* h_active_counts = new int[num_systems];
            CHECK_CUDA(cudaMemcpy(h_active_counts, d_active_counts, num_systems * sizeof(int), cudaMemcpyDeviceToHost));

            // Solve each system's reduced problem using cuSOLVER QR
            for (int sys = 0; sys < num_systems; sys++) {
                int m = 10;  // Always 10 equations
                int n = h_active_counts[sys];  // Number of active variables

                if (n == 0) continue;

                double* A_sys = d_A_reduced + sys * 100;
                double* b_sys = d_b_reduced + sys * 10;
                double* tau_sys = d_tau + sys * 10;
                int* info_sys = d_info + sys;

                // QR factorization workspace query
                int lwork;
                CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolver_handle, m, n, A_sys, n, &lwork));

                double* d_work;
                CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));

                // QR factorization: A = Q * R
                CHECK_CUSOLVER(cusolverDnDgeqrf(cusolver_handle, m, n, A_sys, n, tau_sys, d_work, lwork, info_sys));

                // Apply Q^T to b: b = Q^T * b
                CHECK_CUSOLVER(cusolverDnDormqr(
                    cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                    m, 1, n, A_sys, n, tau_sys, b_sys, m,
                    d_work, lwork, info_sys
                ));

                // Solve R * x = b[0:n] using triangular solve
                double alpha = 1.0;
                CHECK_CUBLAS(cublasDtrsm(
                    cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    n, 1, &alpha, A_sys, n, b_sys, m
                ));

                cudaFree(d_work);
            }

            // Map reduced solutions back to full x
            map_solution_kernel<<<num_systems, 32>>>(d_x, d_b_reduced, d_active_maps, d_active_counts, num_systems);
            CHECK_CUDA(cudaGetLastError());

            delete[] h_active_counts;

            // Check for negatives
            check_negatives_kernel<<<num_systems, 32>>>(d_x, d_active, d_has_negatives, num_systems, TOLERANCE);
            CHECK_CUDA(cudaGetLastError());

            // Check if any system has negatives
            bool h_any_negatives = false;
            bool* h_has_negatives = new bool[num_systems];
            CHECK_CUDA(cudaMemcpy(h_has_negatives, d_has_negatives, num_systems * sizeof(bool), cudaMemcpyDeviceToHost));
            for (int i = 0; i < num_systems; i++) {
                if (h_has_negatives[i]) {
                    h_any_negatives = true;
                    break;
                }
            }
            delete[] h_has_negatives;

            if (!h_any_negatives) break;

            // Apply line search
            line_search_kernel<<<num_systems, 32>>>(d_x, d_x_old, d_active, d_has_negatives, num_systems, TOLERANCE);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    printf("cuSOLVER NNLS completed\n");

    // Cleanup
    cudaFree(d_active);
    cudaFree(d_converged);
    cudaFree(d_has_negatives);
    cudaFree(d_w);
    cudaFree(d_max_val);
    cudaFree(d_max_idx);
    cudaFree(d_x_old);
    cudaFree(d_A_reduced);
    cudaFree(d_b_reduced);
    cudaFree(d_tau);
    cudaFree(d_active_counts);
    cudaFree(d_active_maps);
    cudaFree(d_info);

    cusolverDnDestroy(cusolver_handle);
    cublasDestroy(cublas_handle);
}
