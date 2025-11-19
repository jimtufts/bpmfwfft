// charge_grid_cuda_handler.h
#ifndef CHARGE_GRID_CUDA_HANDLER_H
#define CHARGE_GRID_CUDA_HANDLER_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

class ChargeGridCUDAHandler {
public:
    ChargeGridCUDAHandler();
    ~ChargeGridCUDAHandler();

    void initialize(std::size_t num_atoms, std::size_t grid_size);
    void cleanup();

    void calculateChargeGrid(
        const double* atom_coordinates,
        const double* charges,
        const double* grid_x,
        const double* grid_y,
        const double* grid_z,
        const double* origin_crd,
        const double* upper_most_corner_crd,
        const std::int64_t* upper_most_corner,
        const double* spacing,
        const std::int64_t* eight_corner_shifts,
        const std::int64_t* six_corner_shifts,
        double* output_grid,
        bool use_nnls_solver = false
    );

private:
    // GPU memory pointers
    double *d_atom_coordinates, *d_charges, *d_grid_x, *d_grid_y, *d_grid_z;
    double *d_origin_crd, *d_upper_most_corner_crd, *d_spacing;
    std::int64_t *d_upper_most_corner, *d_eight_corner_shifts, *d_six_corner_shifts;
    double *d_output_grid;

    std::size_t num_atoms;
    std::size_t grid_size;

    bool is_initialized;

    void allocateMemory();
    void freeMemory();
    void copyToDevice(
        const double* atom_coordinates,
        const double* charges,
        const double* grid_x,
        const double* grid_y,
        const double* grid_z,
        const double* origin_crd,
        const double* upper_most_corner_crd,
        const std::int64_t* upper_most_corner,
        const double* spacing,
        const std::int64_t* eight_corner_shifts,
        const std::int64_t* six_corner_shifts
    );
    void copyFromDevice(double* output_grid);
};

#endif // CHARGE_GRID_CUDA_HANDLER_H
