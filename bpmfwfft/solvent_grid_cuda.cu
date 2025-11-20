// solvent_grid_cuda.cu
// GPU implementation of solvent (occupancy) grid calculation

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            throw std::runtime_error(cudaGetErrorString(error)); \
        } \
    } while(0)

__device__ float compute_distance(float x1, float y1, float z1,
                                   float x2, float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

// Kernel: For each grid point, check all atoms to see if any are within VDW radius
__global__ void compute_solvent_grid_kernel(
    const float* __restrict__ atom_coords,  // [n_atoms, 3]
    const float* __restrict__ vdw_radii,    // [n_atoms]
    const float* __restrict__ grid_x,       // [nx]
    const float* __restrict__ grid_y,       // [ny]
    const float* __restrict__ grid_z,       // [nz]
    float* __restrict__ grid,               // [nz, ny, nx]
    int n_atoms,
    int nx, int ny, int nz)
{
    // Each thread handles one grid point
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = nx * ny * nz;

    if (idx >= total_points) return;

    // Convert linear index to 3D grid indices
    int k = idx % nx;
    int j = (idx / nx) % ny;
    int i = idx / (nx * ny);

    // Get grid point coordinates
    float gx = grid_x[k];
    float gy = grid_y[j];
    float gz = grid_z[i];

    // Check all atoms
    float grid_value = 0.0f;
    for (int atom_idx = 0; atom_idx < n_atoms; ++atom_idx) {
        float ax = atom_coords[atom_idx * 3 + 0];
        float ay = atom_coords[atom_idx * 3 + 1];
        float az = atom_coords[atom_idx * 3 + 2];
        float radius = vdw_radii[atom_idx];

        float dist = compute_distance(gx, gy, gz, ax, ay, az);

        if (dist <= radius) {
            grid_value = 1.0f;
            break;  // No need to check other atoms
        }
    }

    // Write result
    grid[i * ny * nx + j * nx + k] = grid_value;
}

extern "C" {

void compute_solvent_grid_cuda(
    const float* atom_coords,
    const float* vdw_radii,
    const float* grid_x,
    const float* grid_y,
    const float* grid_z,
    float* grid_output,
    int n_atoms,
    int nx, int ny, int nz)
{
    // Device pointers
    float *d_atom_coords = nullptr;
    float *d_vdw_radii = nullptr;
    float *d_grid_x = nullptr;
    float *d_grid_y = nullptr;
    float *d_grid_z = nullptr;
    float *d_grid = nullptr;

    try {
        // Allocate device memory
        CHECK_CUDA(cudaMalloc(&d_atom_coords, n_atoms * 3 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_vdw_radii, n_atoms * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grid_x, nx * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grid_y, ny * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grid_z, nz * sizeof(float)));

        int total_points = nx * ny * nz;
        CHECK_CUDA(cudaMalloc(&d_grid, total_points * sizeof(float)));

        // Copy input data to device
        CHECK_CUDA(cudaMemcpy(d_atom_coords, atom_coords, n_atoms * 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vdw_radii, vdw_radii, n_atoms * sizeof(float),
                             cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_grid_x, grid_x, nx * sizeof(float),
                             cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_grid_y, grid_y, ny * sizeof(float),
                             cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_grid_z, grid_z, nz * sizeof(float),
                             cudaMemcpyHostToDevice));

        // Initialize grid to zero
        CHECK_CUDA(cudaMemset(d_grid, 0, total_points * sizeof(float)));

        // Launch kernel
        int threads_per_block = 256;
        int num_blocks = (total_points + threads_per_block - 1) / threads_per_block;

        compute_solvent_grid_kernel<<<num_blocks, threads_per_block>>>(
            d_atom_coords, d_vdw_radii,
            d_grid_x, d_grid_y, d_grid_z,
            d_grid,
            n_atoms, nx, ny, nz
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(grid_output, d_grid, total_points * sizeof(float),
                             cudaMemcpyDeviceToHost));

        // Cleanup
        cudaFree(d_atom_coords);
        cudaFree(d_vdw_radii);
        cudaFree(d_grid_x);
        cudaFree(d_grid_y);
        cudaFree(d_grid_z);
        cudaFree(d_grid);

    } catch (...) {
        // Cleanup on error
        if (d_atom_coords) cudaFree(d_atom_coords);
        if (d_vdw_radii) cudaFree(d_vdw_radii);
        if (d_grid_x) cudaFree(d_grid_x);
        if (d_grid_y) cudaFree(d_grid_y);
        if (d_grid_z) cudaFree(d_grid_z);
        if (d_grid) cudaFree(d_grid);
        throw;
    }
}

} // extern "C"
