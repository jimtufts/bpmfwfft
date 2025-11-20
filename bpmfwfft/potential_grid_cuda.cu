#include "potential_grid_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * CUDA kernel to compute potential grid values
 * Each thread computes potential for one grid point from all atoms
 */
__global__ void compute_potential_kernel(
    const double* __restrict__ atom_coords,
    const double* __restrict__ charges,
    const double* __restrict__ lj_sigma,
    const double* __restrict__ vdw_radii,
    const double* __restrict__ clash_radii,
    const double* __restrict__ molecule_sasa,
    const double* __restrict__ grid_x,
    const double* __restrict__ grid_y,
    const double* __restrict__ grid_z,
    double* __restrict__ output_grid,
    std::size_t natoms,
    std::size_t grid_x_size,
    std::size_t grid_y_size,
    std::size_t grid_z_size,
    double exponent,
    int grid_type  // 0 = electrostatic/LJ, 1 = water, 2 = occupancy/sasa
) {
    // Calculate grid point indices
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= grid_x_size || idy >= grid_y_size || idz >= grid_z_size) {
        return;
    }

    std::size_t grid_idx = idx * grid_y_size * grid_z_size + idy * grid_z_size + idz;

    // Get grid point coordinates
    double gx = grid_x[idx];
    double gy = grid_y[idy];
    double gz = grid_z[idz];

    double potential = 0.0;

    if (grid_type == 1) {
        // Water grid: accumulate contributions from atoms within surface layer
        for (std::size_t atom_idx = 0; atom_idx < natoms; ++atom_idx) {
            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double dist_sq = dx * dx + dy * dy + dz * dz;
            double dist = sqrt(dist_sq);

            double vdw_radius = vdw_radii[atom_idx];
            double surface_layer = vdw_radius + 1.4;  // probe radius

            if (dist <= surface_layer) {
                potential += 1.0;
            }
        }
    } else if (grid_type == 2) {
        // Occupancy/sasa grid: boolean OR - check if within any atom's clash radius
        for (std::size_t atom_idx = 0; atom_idx < natoms; ++atom_idx) {
            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double dist_sq = dx * dx + dy * dy + dz * dz;
            double dist = sqrt(dist_sq);

            double clash_radius = clash_radii[atom_idx];

            if (dist <= clash_radius) {
                potential = 1.0;
                break;  // Found one, no need to check others
            }
        }
    } else {
        // Electrostatic/LJ grids: compute distance-based potential with per-atom clash masking
        for (std::size_t atom_idx = 0; atom_idx < natoms; ++atom_idx) {
            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double dist_sq = dx * dx + dy * dy + dz * dz;
            double dist = sqrt(dist_sq);

            // Avoid division by zero
            if (dist_sq < 1e-10) {
                continue;
            }

            double charge = charges[atom_idx];
            double d = pow(dist_sq, exponent);
            double contribution = 0.0;

            if (d > 1e-10) {
                contribution = charge / d;
            }

            // Apply per-atom clash masking (like CPU implementation)
            double clash_radius = clash_radii[atom_idx];
            if (dist <= clash_radius) {
                contribution = 0.0;
            }

            potential += contribution;
        }
    }

    output_grid[grid_idx] = potential;
}

/**
 * CUDA kernel to apply clash radius masking
 * Sets grid points to zero if within clash radius of any atom
 */
__global__ void apply_clash_mask_kernel(
    const double* __restrict__ atom_coords,
    const double* __restrict__ clash_radii,
    const double* __restrict__ grid_x,
    const double* __restrict__ grid_y,
    const double* __restrict__ grid_z,
    double* __restrict__ output_grid,
    std::size_t natoms,
    std::size_t grid_x_size,
    std::size_t grid_y_size,
    std::size_t grid_z_size
) {
    // Calculate grid point indices
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= grid_x_size || idy >= grid_y_size || idz >= grid_z_size) {
        return;
    }

    std::size_t grid_idx = idx * grid_y_size * grid_z_size + idy * grid_z_size + idz;

    // Get grid point coordinates
    double gx = grid_x[idx];
    double gy = grid_y[idy];
    double gz = grid_z[idz];

    // Check if within clash radius of any atom
    for (std::size_t atom_idx = 0; atom_idx < natoms; ++atom_idx) {
        double ax = atom_coords[atom_idx * 3 + 0];
        double ay = atom_coords[atom_idx * 3 + 1];
        double az = atom_coords[atom_idx * 3 + 2];

        double dx = gx - ax;
        double dy = gy - ay;
        double dz = gz - az;
        double dist_sq = dx * dx + dy * dy + dz * dz;
        double dist = sqrt(dist_sq);

        double clash_radius = clash_radii[atom_idx];

        if (dist <= clash_radius) {
            output_grid[grid_idx] = 0.0;
            return;  // Already masked, no need to check other atoms
        }
    }
}

void launch_potential_grid_kernel(
    const double* d_atom_coordinates,
    const double* d_charges,
    const double* d_lj_sigma,
    const double* d_vdw_radii,
    const double* d_clash_radii,
    const double* d_molecule_sasa,
    const double* d_grid_x,
    const double* d_grid_y,
    const double* d_grid_z,
    double* d_output_grid,
    std::size_t natoms,
    std::size_t grid_x_size,
    std::size_t grid_y_size,
    std::size_t grid_z_size,
    double exponent,
    int grid_type
) {
    // Configure kernel launch parameters
    dim3 block_size(8, 8, 8);
    dim3 grid_size(
        (grid_x_size + block_size.x - 1) / block_size.x,
        (grid_y_size + block_size.y - 1) / block_size.y,
        (grid_z_size + block_size.z - 1) / block_size.z
    );

    compute_potential_kernel<<<grid_size, block_size>>>(
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

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_clash_mask_kernel(
    const double* d_atom_coordinates,
    const double* d_clash_radii,
    const double* d_grid_x,
    const double* d_grid_y,
    const double* d_grid_z,
    double* d_output_grid,
    std::size_t natoms,
    std::size_t grid_x_size,
    std::size_t grid_y_size,
    std::size_t grid_z_size
) {
    // Configure kernel launch parameters
    dim3 block_size(8, 8, 8);
    dim3 grid_size(
        (grid_x_size + block_size.x - 1) / block_size.x,
        (grid_y_size + block_size.y - 1) / block_size.y,
        (grid_z_size + block_size.z - 1) / block_size.z
    );

    apply_clash_mask_kernel<<<grid_size, block_size>>>(
        d_atom_coordinates,
        d_clash_radii,
        d_grid_x,
        d_grid_y,
        d_grid_z,
        d_output_grid,
        natoms,
        grid_x_size,
        grid_y_size,
        grid_z_size
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
