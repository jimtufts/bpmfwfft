#include "sasa_cuda_handler.h"
#include "sasa_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
}

SASACUDAHandler::SASACUDAHandler()
    : d_atom_coords(nullptr)
    , d_atom_radii(nullptr)
    , d_sphere_points(nullptr)
    , d_atom_selection_mask(nullptr)
    , d_output_grid(nullptr)
    , n_atoms(0)
    , n_sphere_points(0)
    , grid_count_x(0)
    , grid_count_y(0)
    , grid_count_z(0)
    , grid_total_size(0)
    , is_initialized(false)
{
}

SASACUDAHandler::~SASACUDAHandler() {
    cleanup();
}

void SASACUDAHandler::initialize(
    std::size_t n_atoms_,
    std::size_t n_sphere_points_,
    int grid_count_x_,
    int grid_count_y_,
    int grid_count_z_
) {
    if (is_initialized) {
        cleanup();
    }

    n_atoms = n_atoms_;
    n_sphere_points = n_sphere_points_;
    grid_count_x = grid_count_x_;
    grid_count_y = grid_count_y_;
    grid_count_z = grid_count_z_;
    grid_total_size = grid_count_x * grid_count_y * grid_count_z;

    allocateMemory();
    is_initialized = true;
}

void SASACUDAHandler::allocateMemory() {
    try {
        CHECK_CUDA(cudaMalloc(&d_atom_coords, n_atoms * 3 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_atom_radii, n_atoms * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sphere_points, n_sphere_points * 3 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_atom_selection_mask, n_atoms * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_output_grid, grid_total_size * sizeof(float)));
    } catch (...) {
        freeMemory();
        throw;
    }
}

void SASACUDAHandler::freeMemory() {
    if (d_atom_coords) { cudaFree(d_atom_coords); d_atom_coords = nullptr; }
    if (d_atom_radii) { cudaFree(d_atom_radii); d_atom_radii = nullptr; }
    if (d_sphere_points) { cudaFree(d_sphere_points); d_sphere_points = nullptr; }
    if (d_atom_selection_mask) { cudaFree(d_atom_selection_mask); d_atom_selection_mask = nullptr; }
    if (d_output_grid) { cudaFree(d_output_grid); d_output_grid = nullptr; }
}

void SASACUDAHandler::cleanup() {
    if (is_initialized) {
        freeMemory();
        is_initialized = false;
    }
}

void SASACUDAHandler::copyToDevice(
    const float* atom_coords,
    const float* atom_radii,
    const float* sphere_points,
    const int* atom_selection_mask
) {
    CHECK_CUDA(cudaMemcpy(d_atom_coords, atom_coords, n_atoms * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_atom_radii, atom_radii, n_atoms * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sphere_points, sphere_points, n_sphere_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_atom_selection_mask, atom_selection_mask, n_atoms * sizeof(int), cudaMemcpyHostToDevice));
}

void SASACUDAHandler::copyFromDevice(float* output_grid) {
    CHECK_CUDA(cudaMemcpy(output_grid, d_output_grid, grid_total_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void SASACUDAHandler::calculateSASAGrid(
    const float* atom_coords,
    const float* atom_radii,
    const float* sphere_points,
    const int* atom_selection_mask,
    float* output_grid,
    float grid_spacing,
    bool use_ten_corners
) {
    if (!is_initialized) {
        throw std::runtime_error("SASACUDAHandler not initialized");
    }

    // Copy data to device
    copyToDevice(atom_coords, atom_radii, sphere_points, atom_selection_mask);

    // Launch SASA kernel
    launch_sasa_kernel(
        d_atom_coords,
        d_atom_radii,
        d_sphere_points,
        d_atom_selection_mask,
        d_output_grid,
        n_atoms,
        n_sphere_points,
        grid_count_x,
        grid_count_y,
        grid_count_z,
        grid_spacing,
        use_ten_corners
    );

    // Copy result back to host
    copyFromDevice(output_grid);
}
