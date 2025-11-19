#include "fft_kernels.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

namespace bpmfwfft {

// ============================================================================
// Tile Extraction/Insertion Kernels
// ============================================================================

__global__ void extractTileKernel(
    const float* full_grid,
    float* tile,
    int grid_x, int grid_y, int grid_z,
    int start_x, int start_y, int start_z,
    int tile_size_x, int tile_size_y, int tile_size_z
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;

    if (tx >= tile_size_x || ty >= tile_size_y || tz >= tile_size_z) {
        return;
    }

    // Position in full grid
    int gx = start_x + tx;
    int gy = start_y + ty;
    int gz = start_z + tz;

    // Index in tile (row-major: z * Y * X + y * X + x)
    int tile_idx = tz * tile_size_y * tile_size_x + ty * tile_size_x + tx;

    // Check if position is within full grid bounds
    if (gx >= 0 && gx < grid_x && gy >= 0 && gy < grid_y && gz >= 0 && gz < grid_z) {
        // Within bounds: copy value
        int grid_idx = gz * grid_y * grid_x + gy * grid_x + gx;
        tile[tile_idx] = full_grid[grid_idx];
    } else {
        // Outside bounds: zero-pad
        tile[tile_idx] = 0.0f;
    }
}

__global__ void accumulateTileKernel(
    const float* tile,
    float* full_grid,
    int grid_x, int grid_y, int grid_z,
    int tile_size_x, int tile_size_y, int tile_size_z,
    int valid_start_x, int valid_start_y, int valid_start_z,
    int valid_size_x, int valid_size_y, int valid_size_z,
    int output_start_x, int output_start_y, int output_start_z
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z * blockDim.z + threadIdx.z;

    if (vx >= valid_size_x || vy >= valid_size_y || vz >= valid_size_z) {
        return;
    }

    // Position within tile
    int tx = valid_start_x + vx;
    int ty = valid_start_y + vy;
    int tz = valid_start_z + vz;

    // Position in output grid
    int gx = output_start_x + vx;
    int gy = output_start_y + vy;
    int gz = output_start_z + vz;

    // Bounds check
    if (gx >= 0 && gx < grid_x && gy >= 0 && gy < grid_y && gz >= 0 && gz < grid_z) {
        int tile_idx = tz * tile_size_y * tile_size_x + ty * tile_size_x + tx;
        int grid_idx = gz * grid_y * grid_x + gy * grid_x + gx;

        // Atomic add for thread safety (in case of overlapping writes)
        atomicAdd(&full_grid[grid_idx], tile[tile_idx]);
    }
}

void launchExtractTileKernel(
    const float* full_grid,
    float* tile,
    int grid_x, int grid_y, int grid_z,
    int start_x, int start_y, int start_z,
    int tile_size_x, int tile_size_y, int tile_size_z,
    cudaStream_t stream
) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (tile_size_x + block.x - 1) / block.x,
        (tile_size_y + block.y - 1) / block.y,
        (tile_size_z + block.z - 1) / block.z
    );

    extractTileKernel<<<grid, block, 0, stream>>>(
        full_grid, tile,
        grid_x, grid_y, grid_z,
        start_x, start_y, start_z,
        tile_size_x, tile_size_y, tile_size_z
    );
}

void launchAccumulateTileKernel(
    const float* tile,
    float* full_grid,
    int grid_x, int grid_y, int grid_z,
    int start_x, int start_y, int start_z,
    int valid_start_x, int valid_start_y, int valid_start_z,
    int valid_size_x, int valid_size_y, int valid_size_z,
    int output_start_x, int output_start_y, int output_start_z,
    cudaStream_t stream
) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (valid_size_x + block.x - 1) / block.x,
        (valid_size_y + block.y - 1) / block.y,
        (valid_size_z + block.z - 1) / block.z
    );

    // Calculate tile dimensions from valid region
    // Tile size is at least the valid region plus its starting offset
    int tile_size_x_calc = valid_start_x + valid_size_x;
    int tile_size_y_calc = valid_start_y + valid_size_y;
    int tile_size_z_calc = valid_start_z + valid_size_z;

    // Use maximum for safe upper bound
    int max_tile_size = tile_size_x_calc;
    if (tile_size_y_calc > max_tile_size) max_tile_size = tile_size_y_calc;
    if (tile_size_z_calc > max_tile_size) max_tile_size = tile_size_z_calc;

    accumulateTileKernel<<<grid, block, 0, stream>>>(
        tile, full_grid,
        grid_x, grid_y, grid_z,
        max_tile_size, max_tile_size, max_tile_size,  // Safe upper bound
        valid_start_x, valid_start_y, valid_start_z,
        valid_size_x, valid_size_y, valid_size_z,
        output_start_x, output_start_y, output_start_z
    );
}

// ============================================================================
// Complex Arithmetic Kernels
// ============================================================================

__global__ void complexMultiplyConjugateKernel(
    const cufftComplex* a,
    const cufftComplex* b,
    cufftComplex* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // result = a * conj(b)
        // conj(b) = (b.x, -b.y)
        // (a.x + i*a.y) * (b.x - i*b.y) = (a.x*b.x + a.y*b.y) + i*(a.y*b.x - a.x*b.y)
        float real = a[idx].x * b[idx].x + a[idx].y * b[idx].y;
        float imag = a[idx].y * b[idx].x - a[idx].x * b[idx].y;

        result[idx].x = real;
        result[idx].y = imag;
    }
}

void launchComplexMultiplyConjugateKernel(
    const cufftComplex* a,
    const cufftComplex* b,
    cufftComplex* result,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    complexMultiplyConjugateKernel<<<grid_size, block_size, 0, stream>>>(
        a, b, result, size
    );
}

// ============================================================================
// Type Conversion Kernels
// ============================================================================

__global__ void extractRealKernel(
    const cufftComplex* complex_array,
    float* real_array,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        real_array[idx] = complex_array[idx].x;
    }
}

void launchExtractRealKernel(
    const cufftComplex* complex_array,
    float* real_array,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    extractRealKernel<<<grid_size, block_size, 0, stream>>>(
        complex_array, real_array, size
    );
}

__global__ void realToComplexKernel(
    const float* real_array,
    cufftComplex* complex_array,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        complex_array[idx].x = real_array[idx];
        complex_array[idx].y = 0.0f;
    }
}

void launchRealToComplexKernel(
    const float* real_array,
    cufftComplex* complex_array,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    realToComplexKernel<<<grid_size, block_size, 0, stream>>>(
        real_array, complex_array, size
    );
}

// ============================================================================
// Utility Kernels
// ============================================================================

__global__ void zeroFloatArrayKernel(float* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = 0.0f;
    }
}

void launchZeroFloatArrayKernel(
    float* array,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    zeroFloatArrayKernel<<<grid_size, block_size, 0, stream>>>(array, size);
}

__global__ void zeroComplexArrayKernel(cufftComplex* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx].x = 0.0f;
        array[idx].y = 0.0f;
    }
}

void launchZeroComplexArrayKernel(
    cufftComplex* array,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    zeroComplexArrayKernel<<<grid_size, block_size, 0, stream>>>(array, size);
}

} // namespace bpmfwfft
