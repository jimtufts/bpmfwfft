#ifndef SASA_CUDA_HANDLER_H
#define SASA_CUDA_HANDLER_H

#include <cstddef>
#include <string>
#include <cuda_runtime.h>

class SASACUDAHandler {
public:
    SASACUDAHandler();
    ~SASACUDAHandler();

    void initialize(
        std::size_t n_atoms,
        std::size_t n_sphere_points,
        int grid_count_x,
        int grid_count_y,
        int grid_count_z
    );

    void cleanup();

    void calculateSASAGrid(
        const float* atom_coords,
        const float* atom_radii,
        const float* sphere_points,
        const int* atom_selection_mask,
        float* output_grid,
        float grid_spacing,
        bool use_ten_corners
    );

private:
    // Device memory pointers
    float *d_atom_coords;
    float *d_atom_radii;
    float *d_sphere_points;
    int *d_atom_selection_mask;
    float *d_output_grid;

    // Dimensions
    std::size_t n_atoms;
    std::size_t n_sphere_points;
    int grid_count_x;
    int grid_count_y;
    int grid_count_z;
    std::size_t grid_total_size;

    bool is_initialized;

    void allocateMemory();
    void freeMemory();

    void copyToDevice(
        const float* atom_coords,
        const float* atom_radii,
        const float* sphere_points,
        const int* atom_selection_mask
    );

    void copyFromDevice(float* output_grid);
};

#endif // SASA_CUDA_HANDLER_H
