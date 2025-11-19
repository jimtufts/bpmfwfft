#ifndef FFT_KERNELS_CUH
#define FFT_KERNELS_CUH

#include <cufft.h>
#include <cuda_runtime.h>

namespace bpmfwfft {

/**
 * Extract a tile from a full 3D grid with optional zero-padding
 *
 * @param full_grid: Input full grid [grid_z][grid_y][grid_x]
 * @param tile: Output tile [tile_z][tile_y][tile_x]
 * @param grid_x, grid_y, grid_z: Full grid dimensions
 * @param start_x, start_y, start_z: Starting position in full grid
 * @param tile_size_x, tile_size_y, tile_size_z: Tile dimensions
 */
void launchExtractTileKernel(
    const float* full_grid,
    float* tile,
    int grid_x, int grid_y, int grid_z,
    int start_x, int start_y, int start_z,
    int tile_size_x, int tile_size_y, int tile_size_z,
    cudaStream_t stream = 0
);

/**
 * Accumulate a tile into a full 3D grid
 * Only the valid region (excluding overlap) is accumulated
 *
 * @param tile: Input tile [tile_z][tile_y][tile_x]
 * @param full_grid: Output grid to accumulate into [grid_z][grid_y][grid_x]
 * @param grid_x, grid_y, grid_z: Full grid dimensions
 * @param start_x, start_y, start_z: Tile position in full grid
 * @param valid_start_x, valid_start_y, valid_start_z: Valid region start within tile
 * @param valid_size_x, valid_size_y, valid_size_z: Valid region size
 * @param output_start_x, output_start_y, output_start_z: Where to place in output grid
 */
void launchAccumulateTileKernel(
    const float* tile,
    float* full_grid,
    int grid_x, int grid_y, int grid_z,
    int start_x, int start_y, int start_z,
    int valid_start_x, int valid_start_y, int valid_start_z,
    int valid_size_x, int valid_size_y, int valid_size_z,
    int output_start_x, int output_start_y, int output_start_z,
    cudaStream_t stream = 0
);

/**
 * Multiply two complex arrays: result = a * conj(b)
 * Used for correlation in frequency domain
 *
 * @param a: First complex array
 * @param b: Second complex array (will be conjugated)
 * @param result: Output array
 * @param size: Number of complex elements
 */
void launchComplexMultiplyConjugateKernel(
    const cufftComplex* a,
    const cufftComplex* b,
    cufftComplex* result,
    int size,
    cudaStream_t stream = 0
);

/**
 * Extract real part from complex array
 *
 * @param complex_array: Input complex array
 * @param real_array: Output real array
 * @param size: Number of elements
 */
void launchExtractRealKernel(
    const cufftComplex* complex_array,
    float* real_array,
    int size,
    cudaStream_t stream = 0
);

/**
 * Convert real array to complex (imaginary part = 0)
 *
 * @param real_array: Input real array
 * @param complex_array: Output complex array
 * @param size: Number of elements
 */
void launchRealToComplexKernel(
    const float* real_array,
    cufftComplex* complex_array,
    int size,
    cudaStream_t stream = 0
);

/**
 * Zero out a float array
 *
 * @param array: Array to zero
 * @param size: Number of elements
 */
void launchZeroFloatArrayKernel(
    float* array,
    int size,
    cudaStream_t stream = 0
);

/**
 * Zero out a complex array
 *
 * @param array: Array to zero
 * @param size: Number of complex elements
 */
void launchZeroComplexArrayKernel(
    cufftComplex* array,
    int size,
    cudaStream_t stream = 0
);

} // namespace bpmfwfft

#endif // FFT_KERNELS_CUH
