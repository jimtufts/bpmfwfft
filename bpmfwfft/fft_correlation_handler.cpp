#include "fft_correlation_handler.h"
#include "fft_kernels.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace bpmfwfft {

// CUDA error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
}

FFTCorrelationHandler::FFTCorrelationHandler()
    : initialized_(false),
      grid_nx_(0), grid_ny_(0), grid_nz_(0), grid_size_(0),
      num_tiles_(0),
      d_tile_real_(nullptr),
      d_tile_complex_(nullptr),
      d_correlation_grid_(nullptr),
      d_product_(nullptr) {
}

FFTCorrelationHandler::~FFTCorrelationHandler() {
    cleanup();
}

void FFTCorrelationHandler::initialize(
    int grid_nx, int grid_ny, int grid_nz,
    int max_tile_size,
    int overlap
) {
    if (initialized_) {
        cleanup();
    }

    grid_nx_ = grid_nx;
    grid_ny_ = grid_ny;
    grid_nz_ = grid_nz;
    grid_size_ = grid_nx * grid_ny * grid_nz;

    // Initialize tile manager
    tile_manager_ = std::make_unique<TileManager>();

    // If max_tile_size is specified, use it; otherwise auto-detect
    if (max_tile_size > 0) {
        // Manual configuration
        tile_config_.grid_size_x = grid_nx;
        tile_config_.grid_size_y = grid_ny;
        tile_config_.grid_size_z = grid_nz;
        tile_config_.tile_size = max_tile_size;

        // Check if this is a single-tile configuration
        if (max_tile_size >= grid_nx && max_tile_size >= grid_ny && max_tile_size >= grid_nz) {
            // Single tile - no overlap needed
            tile_config_.overlap = 0;
            tile_config_.num_tiles_x = 1;
            tile_config_.num_tiles_y = 1;
            tile_config_.num_tiles_z = 1;
        } else {
            // Multiple tiles - use overlap
            tile_config_.overlap = overlap;
            int effective_size = max_tile_size - 2 * overlap;
            if (effective_size <= 0) {
                throw std::runtime_error("Tile size too small for specified overlap");
            }
            tile_config_.num_tiles_x = (grid_nx + effective_size - 1) / effective_size;
            tile_config_.num_tiles_y = (grid_ny + effective_size - 1) / effective_size;
            tile_config_.num_tiles_z = (grid_nz + effective_size - 1) / effective_size;
        }
    } else {
        // Auto-detect based on available memory
        tile_config_ = tile_manager_->determineTiling(grid_nx, grid_ny, grid_nz);
    }

    num_tiles_ = tile_config_.num_tiles_x *
                 tile_config_.num_tiles_y *
                 tile_config_.num_tiles_z;

    std::cout << "FFTCorrelationHandler initialized:" << std::endl;
    std::cout << "  Grid: " << grid_nx << "x" << grid_ny << "x" << grid_nz << std::endl;
    std::cout << "  Tile size: " << tile_config_.tile_size << std::endl;
    std::cout << "  Overlap: " << tile_config_.overlap << std::endl;
    std::cout << "  Num tiles: " << num_tiles_ << " ("
              << tile_config_.num_tiles_x << "x"
              << tile_config_.num_tiles_y << "x"
              << tile_config_.num_tiles_z << ")" << std::endl;

    // Initialize FFT plan manager for tile size
    fft_manager_ = std::make_unique<FFTPlanManager>();
    fft_manager_->createPlansC2C(
        tile_config_.tile_size,
        tile_config_.tile_size,
        tile_config_.tile_size,
        1,  // Single tile at a time
        false  // float32
    );

    // Allocate buffers
    allocateBuffers();

    initialized_ = true;
}

void FFTCorrelationHandler::allocateBuffers() {
    int tile_size = tile_config_.tile_size;
    int tile_elements = tile_size * tile_size * tile_size;

    // Per-tile working buffers
    CHECK_CUDA(cudaMalloc(&d_tile_real_, tile_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tile_complex_, tile_elements * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_product_, tile_elements * sizeof(cufftComplex)));

    // Receptor FFT storage (one buffer per tile)
    d_receptor_fft_tiles_.resize(num_tiles_);
    for (int i = 0; i < num_tiles_; i++) {
        CHECK_CUDA(cudaMalloc(&d_receptor_fft_tiles_[i], tile_elements * sizeof(cufftComplex)));
    }

    // Full correlation grid on GPU
    CHECK_CUDA(cudaMalloc(&d_correlation_grid_, grid_size_ * sizeof(float)));

    std::cout << "  Allocated " << (getMemoryUsage() / (1024.0 * 1024.0))
              << " MB on GPU" << std::endl;
}

void FFTCorrelationHandler::freeBuffers() {
    if (d_tile_real_) {
        cudaFree(d_tile_real_);
        d_tile_real_ = nullptr;
    }
    if (d_tile_complex_) {
        cudaFree(d_tile_complex_);
        d_tile_complex_ = nullptr;
    }
    if (d_product_) {
        cudaFree(d_product_);
        d_product_ = nullptr;
    }
    if (d_correlation_grid_) {
        cudaFree(d_correlation_grid_);
        d_correlation_grid_ = nullptr;
    }

    for (auto* ptr : d_receptor_fft_tiles_) {
        if (ptr) cudaFree(ptr);
    }
    d_receptor_fft_tiles_.clear();
}

void FFTCorrelationHandler::cleanup() {
    freeBuffers();
    fft_manager_.reset();
    tile_manager_.reset();
    initialized_ = false;
}

void FFTCorrelationHandler::precomputeReceptorFFT(const float* receptor_grid) {
    if (!initialized_) {
        throw std::runtime_error("Handler not initialized");
    }

    std::cout << "Precomputing receptor FFT (" << num_tiles_ << " tiles)..." << std::endl;

    // Allocate host buffer for full grid on GPU
    float* d_full_grid;
    CHECK_CUDA(cudaMalloc(&d_full_grid, grid_size_ * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_full_grid, receptor_grid, grid_size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Process each tile
    for (int tile_idx = 0; tile_idx < num_tiles_; tile_idx++) {
        // Get tile bounds
        int tx, ty, tz;
        getTileIndices3D(tile_idx, tx, ty, tz);
        TileManager::TileBounds bounds = tile_manager_->getTileBounds(tile_config_, tx, ty, tz);

        // Extract tile from full grid
        launchExtractTileKernel(
            d_full_grid,
            d_tile_real_,
            grid_nx_, grid_ny_, grid_nz_,
            bounds.start_x, bounds.start_y, bounds.start_z,
            tile_config_.tile_size, tile_config_.tile_size, tile_config_.tile_size
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // Convert to complex
        int tile_elements = tile_config_.tile_size * tile_config_.tile_size * tile_config_.tile_size;
        launchRealToComplexKernel(d_tile_real_, d_tile_complex_, tile_elements);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Forward FFT
        fft_manager_->executeForwardInPlace(d_tile_complex_);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy to receptor FFT storage
        CHECK_CUDA(cudaMemcpy(
            d_receptor_fft_tiles_[tile_idx],
            d_tile_complex_,
            tile_elements * sizeof(cufftComplex),
            cudaMemcpyDeviceToDevice
        ));

        if ((tile_idx + 1) % 100 == 0 || tile_idx == num_tiles_ - 1) {
            std::cout << "  Processed " << (tile_idx + 1) << "/" << num_tiles_ << " tiles" << std::endl;
        }
    }

    cudaFree(d_full_grid);
    std::cout << "Receptor FFT precomputation complete!" << std::endl;
}

double FFTCorrelationHandler::computeCorrelationEnergy(const float* ligand_grid) {
    if (!initialized_) {
        throw std::runtime_error("Handler not initialized");
    }

    // Zero out correlation grid
    launchZeroFloatArrayKernel(d_correlation_grid_, grid_size_);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocate device memory for full ligand grid
    float* d_ligand_full;
    CHECK_CUDA(cudaMalloc(&d_ligand_full, grid_size_ * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ligand_full, ligand_grid, grid_size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    int tile_elements = tile_config_.tile_size * tile_config_.tile_size * tile_config_.tile_size;

    // Process each tile
    for (int tile_idx = 0; tile_idx < num_tiles_; tile_idx++) {
        int tx, ty, tz;
        getTileIndices3D(tile_idx, tx, ty, tz);
        TileManager::TileBounds bounds = tile_manager_->getTileBounds(tile_config_, tx, ty, tz);

        // Extract ligand tile
        launchExtractTileKernel(
            d_ligand_full,
            d_tile_real_,
            grid_nx_, grid_ny_, grid_nz_,
            bounds.start_x, bounds.start_y, bounds.start_z,
            tile_config_.tile_size, tile_config_.tile_size, tile_config_.tile_size
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // Convert to complex
        launchRealToComplexKernel(d_tile_real_, d_tile_complex_, tile_elements);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Forward FFT of ligand tile
        fft_manager_->executeForwardInPlace(d_tile_complex_);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Multiply: receptor_FFT * conj(ligand_FFT)
        launchComplexMultiplyConjugateKernel(
            d_receptor_fft_tiles_[tile_idx],
            d_tile_complex_,
            d_product_,
            tile_elements
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // Inverse FFT to get correlation
        fft_manager_->executeInverseInPlace(d_product_);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Extract real part
        launchExtractRealKernel(d_product_, d_tile_real_, tile_elements);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Apply normalization on CPU side (temporary - should be done with kernel)
        float norm_factor = fft_manager_->getNormalizationFactor();
        std::vector<float> h_tile(tile_elements);
        CHECK_CUDA(cudaMemcpy(h_tile.data(), d_tile_real_, tile_elements * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Debug: Check values before normalization
        if (tile_idx == 0) {
            float tile_sum = 0.0f, tile_max = 0.0f;
            for (float val : h_tile) {
                tile_sum += val;
                tile_max = std::max(tile_max, std::abs(val));
            }
            std::cout << "  Tile 0 before norm - sum: " << tile_sum << ", max: " << tile_max << std::endl;
        }

        for (float& val : h_tile) {
            val *= norm_factor;
        }

        // Debug: Check values after normalization
        if (tile_idx == 0) {
            float tile_sum = 0.0f, tile_max = 0.0f;
            for (float val : h_tile) {
                tile_sum += val;
                tile_max = std::max(tile_max, std::abs(val));
            }
            std::cout << "  Tile 0 after norm - sum: " << tile_sum << ", max: " << tile_max << std::endl;
        }

        CHECK_CUDA(cudaMemcpy(d_tile_real_, h_tile.data(), tile_elements * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Debug: Print bounds for tile 0
        if (tile_idx == 0) {
            std::cout << "  Tile 0 bounds:" << std::endl;
            std::cout << "    Grid: " << grid_nx_ << "x" << grid_ny_ << "x" << grid_nz_ << std::endl;
            std::cout << "    Tile size: " << tile_config_.tile_size << std::endl;
            std::cout << "    Valid start: (" << bounds.valid_start_x << ", " << bounds.valid_start_y << ", " << bounds.valid_start_z << ")" << std::endl;
            std::cout << "    Valid size: (" << bounds.valid_size_x << ", " << bounds.valid_size_y << ", " << bounds.valid_size_z << ")" << std::endl;
            std::cout << "    Output start: (" << bounds.output_start_x << ", " << bounds.output_start_y << ", " << bounds.output_start_z << ")" << std::endl;
        }

        // Accumulate tile into full correlation grid
        launchAccumulateTileKernel(
            d_tile_real_,
            d_correlation_grid_,
            grid_nx_, grid_ny_, grid_nz_,
            tile_config_.tile_size, tile_config_.tile_size, tile_config_.tile_size,  // Tile dimensions
            bounds.valid_start_x, bounds.valid_start_y, bounds.valid_start_z,
            bounds.valid_size_x, bounds.valid_size_y, bounds.valid_size_z,
            bounds.output_start_x, bounds.output_start_y, bounds.output_start_z
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cudaFree(d_ligand_full);

    // Sum correlation grid to get total energy
    double total_energy = sumGridEnergy(d_correlation_grid_, grid_size_);

    return total_energy;
}

void FFTCorrelationHandler::getCorrelationGrid(float* correlation_grid) {
    if (!initialized_) {
        throw std::runtime_error("Handler not initialized");
    }

    CHECK_CUDA(cudaMemcpy(
        correlation_grid,
        d_correlation_grid_,
        grid_size_ * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

double FFTCorrelationHandler::sumGridEnergy(const float* d_grid, int size) {
    // Use cuBLAS for efficient sum (via dot product with vector of ones)
    // For now, use a simple CPU-based sum
    std::vector<float> h_grid(size);
    CHECK_CUDA(cudaMemcpy(h_grid.data(), d_grid, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (float val : h_grid) {
        sum += val;
    }

    return sum;
}

size_t FFTCorrelationHandler::getMemoryUsage() const {
    if (!initialized_) return 0;

    int tile_elements = tile_config_.tile_size * tile_config_.tile_size * tile_config_.tile_size;

    size_t total = 0;

    // Per-tile buffers
    total += tile_elements * sizeof(float);           // d_tile_real_
    total += tile_elements * sizeof(cufftComplex);    // d_tile_complex_
    total += tile_elements * sizeof(cufftComplex);    // d_product_

    // Receptor FFT tiles
    total += num_tiles_ * tile_elements * sizeof(cufftComplex);

    // Full correlation grid
    total += grid_size_ * sizeof(float);

    return total;
}

void FFTCorrelationHandler::getTileIndices3D(int tile_idx, int& tx, int& ty, int& tz) const {
    // Convert linear tile index to 3D indices
    // Layout: z * (ny * nx) + y * nx + x
    int nxy = tile_config_.num_tiles_x * tile_config_.num_tiles_y;

    tz = tile_idx / nxy;
    int remainder = tile_idx % nxy;
    ty = remainder / tile_config_.num_tiles_x;
    tx = remainder % tile_config_.num_tiles_x;
}

} // namespace bpmfwfft
