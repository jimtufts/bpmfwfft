#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cusolverDn.h>

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << " : " \
                  << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
}

// Error checking macro for cuBLAS calls
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in file '" << __FILE__ << "' in line " << __LINE__ << " : " \
                  << cublasGetStatusString(status) << std::endl; \
        cudaError_t cudaStatus = cudaGetLastError(); \
        if (cudaStatus != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl; \
        } \
        throw std::runtime_error("cuBLAS error"); \
    } \
}

#define CHECK_CUSOLVER(call) \
    { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// Platform-specific includes and second() function definition
#if defined(__linux__)
#include <stddef.h>
#include <sys/resource.h>
#include <sys/time.h>
double second(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#elif defined(__APPLE__)
#include <stddef.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/types.h>
double second(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif

__global__ void createPointerArray(double** pointerArray, double* data, int stride, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        pointerArray[idx] = data + idx * stride;
    }
}

__global__ void computeTenCornersKernel(
    const double* atom_coordinates,
    const double* origin_crd,
    const double* upper_most_corner_crd,
    const int64_t* upper_most_corner,
    const double* spacing,
    const double* grid_x,
    const double* grid_y,
    const double* grid_z,
    int num_atoms,
    int64_t* ten_corners)
{
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= num_atoms) return;

    double atom_coordinate[3] = {
        atom_coordinates[atom_idx * 3],
        atom_coordinates[atom_idx * 3 + 1],
        atom_coordinates[atom_idx * 3 + 2]
    };

    if (atom_idx < 5) {
        printf("Atom %d: coordinates = (%f, %f, %f)\n", 
               atom_idx, atom_coordinate[0], atom_coordinate[1], atom_coordinate[2]);
        printf("Grid bounds: (%f, %f, %f) to (%f, %f, %f)\n",
               origin_crd[0], origin_crd[1], origin_crd[2],
               upper_most_corner_crd[0], upper_most_corner_crd[1], upper_most_corner_crd[2]);
    }

    int64_t lower_corner[3];
    for (int i = 0; i < 3; ++i) {
        lower_corner[i] = static_cast<int64_t>((atom_coordinate[i] - origin_crd[i]) / spacing[i]);
        lower_corner[i] = max(int64_t(0), min(lower_corner[i], upper_most_corner[i] - 1));
    }
    
    if (atom_idx < 5) {
        printf("Atom %d: Lower corner calculated: (%ld, %ld, %ld)\n", 
               atom_idx, lower_corner[0], lower_corner[1], lower_corner[2]);
    }

    int64_t eight_corners[8][3];
    double distances[8];
    
    if (atom_idx < 5) printf("Atom %d: Calculating eight corners\n", atom_idx);
    
    for (int i = 0; i < 8; ++i) {
        eight_corners[i][0] = lower_corner[0] + (i & 1);
        eight_corners[i][1] = lower_corner[1] + ((i >> 1) & 1);
        eight_corners[i][2] = lower_corner[2] + ((i >> 2) & 1);
        
        double corner_crd[3] = {
            grid_x[eight_corners[i][0]],
            grid_y[eight_corners[i][1]],
            grid_z[eight_corners[i][2]]
        };
        
        distances[i] = sqrt(
            (corner_crd[0] - atom_coordinate[0]) * (corner_crd[0] - atom_coordinate[0]) +
            (corner_crd[1] - atom_coordinate[1]) * (corner_crd[1] - atom_coordinate[1]) +
            (corner_crd[2] - atom_coordinate[2]) * (corner_crd[2] - atom_coordinate[2])
        );
        
        if (atom_idx < 5) {
            printf("  Corner %d: (%ld, %ld, %ld), distance: %f\n", 
                   i, eight_corners[i][0], eight_corners[i][1], eight_corners[i][2], distances[i]);
        }
    }

    int nearest_ind = 0;
    int furthest_ind = 0;
    for (int i = 1; i < 8; ++i) {
        if (distances[i] < distances[nearest_ind]) nearest_ind = i;
        if (distances[i] > distances[furthest_ind]) furthest_ind = i;
    }
    
    if (atom_idx < 5) {
        printf("Atom %d: Nearest corner index: %d, Furthest corner index: %d\n", 
               atom_idx, nearest_ind, furthest_ind);
    }

    int64_t six_corners[6][3];
    const int64_t six_corner_shifts[6][3] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}
    };
    
    if (atom_idx < 5) printf("Atom %d: Calculating six additional corners\n", atom_idx);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) {
            six_corners[i][j] = min(upper_most_corner[j], max(int64_t(0), 
                eight_corners[nearest_ind][j] + six_corner_shifts[i][j]));
        }
        if (atom_idx < 5) {
            printf("  Additional Corner %d: (%ld, %ld, %ld)\n", 
                   i, six_corners[i][0], six_corners[i][1], six_corners[i][2]);
        }
    }

    if (atom_idx < 5) printf("Atom %d: Selecting ten corners\n", atom_idx);
    int corner_count = 0;
    for (int i = 0; i < 8; ++i) {
        if (i != furthest_ind) {
            for (int j = 0; j < 3; ++j) {
                ten_corners[atom_idx * 30 + corner_count * 3 + j] = eight_corners[i][j];
            }
            if (atom_idx < 5) {
                printf("  Selected Corner %d: (%ld, %ld, %ld)\n", 
                       corner_count, eight_corners[i][0], eight_corners[i][1], eight_corners[i][2]);
            }
            ++corner_count;
        }
    }

    for (int i = 0; i < 6 && corner_count < 10; ++i) {
        bool is_new = true;
        for (int j = 0; j < 8; ++j) {
            if (six_corners[i][0] == eight_corners[j][0] &&
                six_corners[i][1] == eight_corners[j][1] &&
                six_corners[i][2] == eight_corners[j][2]) {
                is_new = false;
                break;
            }
        }
        if (is_new) {
            for (int j = 0; j < 3; ++j) {
                ten_corners[atom_idx * 30 + corner_count * 3 + j] = six_corners[i][j];
            }
            if (atom_idx < 5) {
                printf("  Selected Additional Corner %d: (%ld, %ld, %ld)\n", 
                       corner_count, six_corners[i][0], six_corners[i][1], six_corners[i][2]);
            }
            ++corner_count;
        }
    }

    while (corner_count < 10) {
        for (int j = 0; j < 3; ++j) {
            ten_corners[atom_idx * 30 + corner_count * 3 + j] = eight_corners[nearest_ind][j];
        }
        if (atom_idx < 5) {
            printf("  Filled Remaining Corner %d: (%ld, %ld, %ld)\n", 
                   corner_count, eight_corners[nearest_ind][0], eight_corners[nearest_ind][1], eight_corners[nearest_ind][2]);
        }
        ++corner_count;
    }

    if (atom_idx < 5) {
        printf("Atom %d: Final ten corners\n", atom_idx);
        for (int i = 0; i < 10; ++i) {
            printf("  Corner %d: (%ld, %ld, %ld)\n", i,
                   ten_corners[atom_idx * 30 + i * 3],
                   ten_corners[atom_idx * 30 + i * 3 + 1],
                   ten_corners[atom_idx * 30 + i * 3 + 2]);
        }
    }
}

__global__ void setupLinearSystemKernel(
    const int64_t* ten_corners,
    const double* atom_coordinates,
    const double* charges,
    const double* grid_x,
    const double* grid_y,
    const double* grid_z,
    int num_atoms,
    int i_max, int j_max, int k_max,
    double* a_matrices,
    double* b_vectors
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= num_atoms) return;

    // Set up b vector
    for (int i = 0; i < 10; ++i) {
        b_vectors[atom_idx * 10 + i] = (i == 0) ? charges[atom_idx] : 0.0;
    }

    // Set up A matrix
    double* a_matrix = &a_matrices[atom_idx * 100];

    // First row of A is all ones
    for (int j = 0; j < 10; ++j) {
        a_matrix[j] = 1.0;
    }

    // Compute delta vectors and fill the rest of A
    for (int i = 0; i < 10; ++i) {
        int64_t x = ten_corners[atom_idx * 30 + i * 3];
        int64_t y = ten_corners[atom_idx * 30 + i * 3 + 1];
        int64_t z = ten_corners[atom_idx * 30 + i * 3 + 2];

        double corner_crd[3] = {grid_x[x], grid_y[y], grid_z[z]};
        double delta[3];
        for (int d = 0; d < 3; ++d) {
            delta[d] = corner_crd[d] - atom_coordinates[atom_idx * 3 + d];
        }

        a_matrix[10 + i] = delta[0];
        a_matrix[20 + i] = delta[1];
        a_matrix[30 + i] = delta[2];
        a_matrix[40 + i] = delta[0] * delta[0];
        a_matrix[50 + i] = delta[1] * delta[1];
        a_matrix[60 + i] = delta[2] * delta[2];
        a_matrix[70 + i] = delta[0] * delta[1];
        a_matrix[80 + i] = delta[1] * delta[2];
        a_matrix[90 + i] = delta[2] * delta[0];
    }

    if (atom_idx < 5) {
        printf("CUDA Atom %d: Linear System\n", atom_idx);
        printf("A matrix:\n");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                printf("%.4e ", a_matrix[i * 10 + j]);
            }
            printf("\n");
        }
        printf("b vector: ");
        for (int i = 0; i < 10; i++) {
            printf("%.4e ", b_vectors[atom_idx * 10 + i]);
        }
        printf("\n");
    }
}

__global__ void distributeChargesKernel(
    const double* distributed_charges,
    const int64_t* ten_corners,
    int num_atoms,
    int i_max, int j_max, int k_max,
    double* grid
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= num_atoms) return;

    if (atom_idx < 5) {
        printf("CUDA Atom %d: Grid Addition\n", atom_idx);
    }

    for (int i = 0; i < 10; ++i) {
        int64_t x = ten_corners[atom_idx * 30 + i * 3];
        int64_t y = ten_corners[atom_idx * 30 + i * 3 + 1];
        int64_t z = ten_corners[atom_idx * 30 + i * 3 + 2];

        if (x >= 0 && x < i_max && y >= 0 && y < j_max && z >= 0 && z < k_max) {
            double charge = distributed_charges[atom_idx * 10 + i];
            atomicAdd(&grid[x * j_max * k_max + y * k_max + z], charge);
            
            if (atom_idx < 5) {
                printf("CUDA Atom %d: Corner %d: (%.4e) added to grid[%ld][%ld][%ld]\n", 
                       atom_idx, i, charge, x, y, z);
            }
        }
    }
}

// Transpose matrices kernel
__global__ void transposeMatricesKernel(const double* input, double* output, int num_matrices, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = rows * cols;

    if (idx < num_matrices * matrix_size) {
        int matrix_idx = idx / matrix_size;
        int element_idx = idx % matrix_size;
        int i = element_idx / cols;
        int j = element_idx % cols;

        output[matrix_idx * matrix_size + j * rows + i] = input[matrix_idx * matrix_size + i * cols + j];
    }
}

extern "C" void cuda_cal_charge_grid(
    const double* d_atom_coordinates,
    const double* d_charges,
    const double* d_grid_x,
    const double* d_grid_y,
    const double* d_grid_z,
    const double* d_origin_crd,
    const double* d_upper_most_corner_crd,
    const int64_t* d_upper_most_corner,
    const double* d_spacing,
    const int64_t* d_eight_corner_shifts,
    const int64_t* d_six_corner_shifts,
    int num_atoms,
    int i_max, int j_max, int k_max,
    double* d_grid,
    bool use_nnls_solver
) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device memory
    int64_t *d_ten_corners;
    double *d_a_matrices, *d_b_vectors;
    int *d_pivots, *d_info;
    double **d_a_array, **d_b_array;
    CHECK_CUDA(cudaMalloc(&d_ten_corners, num_atoms * 10 * 3 * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_a_matrices, num_atoms * 10 * 10 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b_vectors, num_atoms * 10 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_pivots, num_atoms * 10 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_info, num_atoms * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_a_array, num_atoms * sizeof(double*)));
    CHECK_CUDA(cudaMalloc(&d_b_array, num_atoms * sizeof(double*)));

    // Compute ten corners
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_atoms + threadsPerBlock - 1) / threadsPerBlock;
    computeTenCornersKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_atom_coordinates, d_origin_crd, d_upper_most_corner_crd,
        d_upper_most_corner, d_spacing, d_grid_x, d_grid_y, d_grid_z,
        num_atoms, d_ten_corners
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Setup linear system
    setupLinearSystemKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_ten_corners, d_atom_coordinates, d_charges,
        d_grid_x, d_grid_y, d_grid_z,
        num_atoms, i_max, j_max, k_max,
        d_a_matrices, d_b_vectors
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (use_nnls_solver) {
        // Allocate additional memory for NNLS solver
        double *d_At, *d_x, *d_R;
        int *d_nnlsIters, *d_lsIters;
        CHECK_CUDA(cudaMalloc(&d_At, num_atoms * 10 * 10 * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_x, num_atoms * 10 * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_R, num_atoms * 10 * 10 * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_nnlsIters, num_atoms * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_lsIters, num_atoms * sizeof(int)));
        
	size_t size_A = num_atoms * 10 * 10 * sizeof(double);
        size_t size_x = num_atoms * 10 * sizeof(double);
        size_t size_b = num_atoms * 10 * sizeof(double);
        size_t size_R = num_atoms * 10 * 10 * sizeof(double);
        
        printf("Allocated sizes:\n");
        printf("d_At: %zu bytes\n", size_A);
        printf("d_x: %zu bytes\n", size_x);
        printf("d_b_vectors: %zu bytes\n", size_b);
        printf("d_R: %zu bytes\n", size_R);
        // Implementation not yet working
        	
    } else {
        // Setup array of pointers for cuBLAS
        createPointerArray<<<blocksPerGrid, threadsPerBlock>>>(
            d_a_array, d_a_matrices, 10 * 10, num_atoms
        );
        createPointerArray<<<blocksPerGrid, threadsPerBlock>>>(
            d_b_array, d_b_vectors, 10, num_atoms
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Perform batched LU factorization
        CHECK_CUBLAS(cublasDgetrfBatched(handle, 10,
                            d_a_array, 10,
                            d_pivots, d_info, num_atoms));

        // Check for LU factorization success
        int h_info;
        CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            std::cerr << "LU factorization failed for atom: " << h_info << std::endl;
            // Handle the error appropriately
        }

        // Solve the systems using the LU factorization
        CHECK_CUBLAS(cublasDgetrsBatched(handle, CUBLAS_OP_T, 10, 1,
                            (const double**)d_a_array, 10,
                            d_pivots,
                            d_b_array, 10,
                            &h_info,
                            num_atoms));
        if (h_info != 0) {
            std::cerr << "Solving the system failed for atom: " << h_info << std::endl;
            // Handle the error appropriately
        }
    }

    // Initialize the grid to zero
    CHECK_CUDA(cudaMemset(d_grid, 0, i_max * j_max * k_max * sizeof(double)));

    // Distribute charges to grid
    distributeChargesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_b_vectors, d_ten_corners,
        num_atoms, i_max, j_max, k_max, d_grid
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clean up
    cudaFree(d_ten_corners);
    cudaFree(d_a_matrices);
    cudaFree(d_b_vectors);
    cudaFree(d_pivots);
    cudaFree(d_info);
    cudaFree(d_a_array);
    cudaFree(d_b_array);
    cublasDestroy(handle);
}
