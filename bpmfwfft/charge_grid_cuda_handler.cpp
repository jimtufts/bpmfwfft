#include "charge_grid_cuda_handler.h"
#include "charge_grid_cuda.h"
#include <stdexcept>
#include <iostream>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

ChargeGridCUDAHandler::ChargeGridCUDAHandler() : 
    d_atom_coordinates(nullptr), d_charges(nullptr), 
    d_grid_x(nullptr), d_grid_y(nullptr), d_grid_z(nullptr),
    d_origin_crd(nullptr), d_upper_most_corner_crd(nullptr), d_spacing(nullptr),
    d_upper_most_corner(nullptr), d_eight_corner_shifts(nullptr), d_six_corner_shifts(nullptr),
    d_output_grid(nullptr), num_atoms(0), grid_size(0), is_initialized(false) {}

ChargeGridCUDAHandler::~ChargeGridCUDAHandler() {
    cleanup();
}

void ChargeGridCUDAHandler::initialize(std::size_t num_atoms, std::size_t grid_size) {
    if (is_initialized) {
        cleanup();
    }
    this->num_atoms = num_atoms;
    this->grid_size = grid_size;
    allocateMemory();
    is_initialized = true;
}

void ChargeGridCUDAHandler::cleanup() {
    if (is_initialized) {
        freeMemory();
        is_initialized = false;
    }
}

void ChargeGridCUDAHandler::calculateChargeGrid(
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
    double* output_grid)
{
    if (!is_initialized) {
        throw std::runtime_error("ChargeGridCUDAHandler not initialized");
    }

    try {
        copyToDevice(atom_coordinates, charges, grid_x, grid_y, grid_z,
                     origin_crd, upper_most_corner_crd, upper_most_corner,
                     spacing, eight_corner_shifts, six_corner_shifts);

        cuda_cal_charge_grid(
            d_atom_coordinates, d_charges, d_grid_x, d_grid_y, d_grid_z,
            d_origin_crd, d_upper_most_corner_crd, d_upper_most_corner,
            d_spacing, d_eight_corner_shifts, d_six_corner_shifts,
            static_cast<int>(num_atoms), static_cast<int>(grid_size), 
            static_cast<int>(grid_size), static_cast<int>(grid_size),
            d_output_grid, false
        );

        copyFromDevice(output_grid);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in calculateChargeGrid: " << e.what() << std::endl;
        throw;
    }
}

void ChargeGridCUDAHandler::allocateMemory() {
    CHECK_CUDA(cudaMalloc(&d_atom_coordinates, num_atoms * 3 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_charges, num_atoms * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_grid_x, grid_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_grid_y, grid_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_grid_z, grid_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_origin_crd, 3 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_upper_most_corner_crd, 3 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_spacing, 3 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_upper_most_corner, 3 * sizeof(std::int64_t)));
    CHECK_CUDA(cudaMalloc(&d_eight_corner_shifts, 24 * sizeof(std::int64_t)));
    CHECK_CUDA(cudaMalloc(&d_six_corner_shifts, 18 * sizeof(std::int64_t)));
    CHECK_CUDA(cudaMalloc(&d_output_grid, grid_size * grid_size * grid_size * sizeof(double)));
}

void ChargeGridCUDAHandler::freeMemory() {
    cudaFree(d_atom_coordinates);
    cudaFree(d_charges);
    cudaFree(d_grid_x);
    cudaFree(d_grid_y);
    cudaFree(d_grid_z);
    cudaFree(d_origin_crd);
    cudaFree(d_upper_most_corner_crd);
    cudaFree(d_spacing);
    cudaFree(d_upper_most_corner);
    cudaFree(d_eight_corner_shifts);
    cudaFree(d_six_corner_shifts);
    cudaFree(d_output_grid);
}

void ChargeGridCUDAHandler::copyToDevice(
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
    const std::int64_t* six_corner_shifts)
{
    CHECK_CUDA(cudaMemcpy(d_atom_coordinates, atom_coordinates, num_atoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_charges, charges, num_atoms * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid_x, grid_x, grid_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid_y, grid_y, grid_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid_z, grid_z, grid_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_origin_crd, origin_crd, 3 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_upper_most_corner_crd, upper_most_corner_crd, 3 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_spacing, spacing, 3 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_upper_most_corner, upper_most_corner, 3 * sizeof(std::int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_eight_corner_shifts, eight_corner_shifts, 24 * sizeof(std::int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_six_corner_shifts, six_corner_shifts, 18 * sizeof(std::int64_t), cudaMemcpyHostToDevice));
}

void ChargeGridCUDAHandler::copyFromDevice(double* output_grid) {
    CHECK_CUDA(cudaMemcpy(output_grid, d_output_grid, grid_size * grid_size * grid_size * sizeof(double), cudaMemcpyDeviceToHost));
}
