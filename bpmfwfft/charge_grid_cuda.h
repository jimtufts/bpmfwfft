#ifndef CHARGE_GRID_CUDA_H
#define CHARGE_GRID_CUDA_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_cal_charge_grid(
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
);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
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
    int64_t* ten_corners
);

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
);

__global__ void createPointerArray(double** pointerArray, double* data, int stride, int count);
#endif // __CUDACC__

#endif // CHARGE_GRID_CUDA_H
