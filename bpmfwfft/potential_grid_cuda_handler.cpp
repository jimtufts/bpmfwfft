#include "potential_grid_cuda_handler.h"
#include "potential_grid_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <iostream>

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
}

PotentialGridCUDAHandler::PotentialGridCUDAHandler()
    : d_atom_coordinates(nullptr)
    , d_charges(nullptr)
    , d_lj_sigma(nullptr)
    , d_vdw_radii(nullptr)
    , d_clash_radii(nullptr)
    , d_molecule_sasa(nullptr)
    , d_grid_x(nullptr)
    , d_grid_y(nullptr)
    , d_grid_z(nullptr)
    , d_output_grid(nullptr)
    , natoms(0)
    , grid_x_size(0)
    , grid_y_size(0)
    , grid_z_size(0)
    , grid_total_size(0)
    , is_initialized(false)
{
}

PotentialGridCUDAHandler::~PotentialGridCUDAHandler() {
    cleanup();
}

void PotentialGridCUDAHandler::initialize(
    std::size_t natoms_,
    std::size_t grid_x_size_,
    std::size_t grid_y_size_,
    std::size_t grid_z_size_
) {
    if (is_initialized) {
        cleanup();
    }

    natoms = natoms_;
    grid_x_size = grid_x_size_;
    grid_y_size = grid_y_size_;
    grid_z_size = grid_z_size_;
    grid_total_size = grid_x_size * grid_y_size * grid_z_size;

    allocateMemory();
    is_initialized = true;
}

void PotentialGridCUDAHandler::allocateMemory() {
    try {
        CHECK_CUDA(cudaMalloc(&d_atom_coordinates, natoms * 3 * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_charges, natoms * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_lj_sigma, natoms * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_vdw_radii, natoms * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_clash_radii, natoms * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_molecule_sasa, natoms * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_grid_x, grid_x_size * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_grid_y, grid_y_size * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_grid_z, grid_z_size * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_output_grid, grid_total_size * sizeof(double)));
    } catch (...) {
        freeMemory();
        throw;
    }
}

void PotentialGridCUDAHandler::freeMemory() {
    if (d_atom_coordinates) { cudaFree(d_atom_coordinates); d_atom_coordinates = nullptr; }
    if (d_charges) { cudaFree(d_charges); d_charges = nullptr; }
    if (d_lj_sigma) { cudaFree(d_lj_sigma); d_lj_sigma = nullptr; }
    if (d_vdw_radii) { cudaFree(d_vdw_radii); d_vdw_radii = nullptr; }
    if (d_clash_radii) { cudaFree(d_clash_radii); d_clash_radii = nullptr; }
    if (d_molecule_sasa) { cudaFree(d_molecule_sasa); d_molecule_sasa = nullptr; }
    if (d_grid_x) { cudaFree(d_grid_x); d_grid_x = nullptr; }
    if (d_grid_y) { cudaFree(d_grid_y); d_grid_y = nullptr; }
    if (d_grid_z) { cudaFree(d_grid_z); d_grid_z = nullptr; }
    if (d_output_grid) { cudaFree(d_output_grid); d_output_grid = nullptr; }
}

void PotentialGridCUDAHandler::cleanup() {
    if (is_initialized) {
        freeMemory();
        is_initialized = false;
    }
}

void PotentialGridCUDAHandler::copyToDevice(
    const double* atom_coordinates,
    const double* grid_x,
    const double* grid_y,
    const double* grid_z,
    const double* charges,
    const double* lj_sigma,
    const double* vdw_radii,
    const double* clash_radii,
    const double* molecule_sasa
) {
    CHECK_CUDA(cudaMemcpy(d_atom_coordinates, atom_coordinates, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_charges, charges, natoms * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lj_sigma, lj_sigma, natoms * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vdw_radii, vdw_radii, natoms * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_clash_radii, clash_radii, natoms * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_molecule_sasa, molecule_sasa, natoms * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid_x, grid_x, grid_x_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid_y, grid_y, grid_y_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid_z, grid_z, grid_z_size * sizeof(double), cudaMemcpyHostToDevice));
}

void PotentialGridCUDAHandler::copyFromDevice(double* output_grid) {
    CHECK_CUDA(cudaMemcpy(output_grid, d_output_grid, grid_total_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void PotentialGridCUDAHandler::calculatePotentialGrid(
    const std::string& grid_name,
    const double* atom_coordinates,
    const double* grid_x,
    const double* grid_y,
    const double* grid_z,
    const double* charges,
    const double* lj_sigma,
    const double* vdw_radii,
    const double* clash_radii,
    const double* molecule_sasa,
    double* output_grid
) {
    if (!is_initialized) {
        throw std::runtime_error("PotentialGridCUDAHandler not initialized");
    }

    // Copy data to device
    copyToDevice(atom_coordinates, grid_x, grid_y, grid_z,
                 charges, lj_sigma, vdw_radii, clash_radii, molecule_sasa);

    // Initialize output grid to zero
    CHECK_CUDA(cudaMemset(d_output_grid, 0, grid_total_size * sizeof(double)));

    // Determine exponent and grid type based on grid name
    double exponent = 0.0;
    int grid_type = 0;  // 0 = electrostatic/LJ, 1 = water, 2 = occupancy/sasa

    if (grid_name == "electrostatic") {
        exponent = 0.5;
        grid_type = 0;
    } else if (grid_name == "LJa") {
        exponent = 3.0;
        grid_type = 0;
    } else if (grid_name == "LJr") {
        exponent = 6.0;
        grid_type = 0;
    } else if (grid_name == "water") {
        exponent = 1.0;
        grid_type = 1;
    } else if (grid_name == "occupancy" || grid_name == "sasa") {
        exponent = 1.0;  // Not used for occupancy/sasa
        grid_type = 2;
    } else {
        throw std::runtime_error("Unknown grid type: " + grid_name);
    }

    // Launch potential grid kernel (now includes per-atom clash masking for electrostatic/LJ)
    launch_potential_grid_kernel(
        d_atom_coordinates,
        d_charges,
        d_lj_sigma,
        d_vdw_radii,
        d_clash_radii,
        d_molecule_sasa,
        d_grid_x,
        d_grid_y,
        d_grid_z,
        d_output_grid,
        natoms,
        grid_x_size,
        grid_y_size,
        grid_z_size,
        exponent,
        grid_type
    );

    // Copy result back to host
    copyFromDevice(output_grid);
}
