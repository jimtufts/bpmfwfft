#include "fft_tile_manager.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace bpmfwfft {

TileManager::TileManager() {
}

TileManager::~TileManager() {
}

size_t TileManager::queryAvailableGPUMemory() {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "Warning: Failed to query GPU memory: "
                  << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return free_mem;
}

size_t TileManager::estimateMemoryRequired(
    int tile_size,
    int num_grid_types,
    bool use_float64,
    bool include_receptor_ffts
) {
    size_t element_size = use_float64 ? sizeof(double) : sizeof(float);
    size_t complex_element_size = element_size * 2;  // Complex numbers

    size_t tile_points = static_cast<size_t>(tile_size) * tile_size * tile_size;

    // Memory components:
    // 1. One ligand spatial grid (current tile being processed)
    size_t ligand_spatial = tile_points * element_size;

    // 2. One ligand FFT buffer (complex)
    size_t ligand_fft = tile_points * complex_element_size;

    // 3. Receptor FFTs (all 6 types, persistent, complex)
    size_t receptor_ffts = include_receptor_ffts
        ? (tile_points * complex_element_size * num_grid_types)
        : 0;

    // 4. Correlation FFT buffer (complex, intermediate)
    size_t correlation_fft = tile_points * complex_element_size;

    // 5. Correlation result (real)
    size_t correlation_result = tile_points * element_size;

    // 6. Energy accumulation grid (for full grid or tiles)
    // For tiled case, this could be the full grid size, but we'll
    // estimate based on tile size for now (can page to CPU if needed)
    size_t energy_grid = tile_points * element_size;

    // 7. Workspace and scratch buffers (~20% overhead)
    size_t workspace = (ligand_spatial + ligand_fft + correlation_fft) / 5;

    size_t total = ligand_spatial + ligand_fft + receptor_ffts +
                   correlation_fft + correlation_result +
                   energy_grid + workspace;

    return total;
}

int TileManager::calculateOverlap(int tile_size) const {
    // For correlation operations, we need overlap to handle boundaries
    // Use 10% of tile size or minimum 16 points, whichever is larger
    int overlap = std::max(tile_size / 10, 16);

    // Cap at 32 points for efficiency
    overlap = std::min(overlap, 32);

    return overlap;
}

int TileManager::findOptimalTileSize(
    int max_dimension,
    size_t available_vram,
    int num_grid_types,
    bool use_float64
) const {
    // Try standard tile sizes in descending order
    // Use powers of 2 for FFT efficiency
    std::vector<int> candidate_sizes = {512, 256, 128, 64, 32};

    // If grid is smaller than largest candidate, try the grid size itself
    if (max_dimension < candidate_sizes[0]) {
        candidate_sizes.insert(candidate_sizes.begin(), max_dimension);
    }

    // Reserve 30% of memory for safety margin and other allocations
    size_t usable_vram = static_cast<size_t>(available_vram * 0.7);

    for (int tile_size : candidate_sizes) {
        if (tile_size > max_dimension) {
            continue;  // Tile can't be larger than grid
        }

        size_t required = estimateMemoryRequired(
            tile_size, num_grid_types, use_float64, true
        );

        if (required <= usable_vram) {
            return tile_size;
        }
    }

    // If even smallest tile doesn't fit, return minimum and warn
    std::cerr << "Warning: Even minimum tile size (32) exceeds available memory. "
              << "Proceeding anyway, but may encounter OOM errors." << std::endl;
    return 32;
}

TileManager::TileConfig TileManager::determineTiling(
    int grid_x, int grid_y, int grid_z,
    size_t available_vram,
    int num_grid_types,
    bool use_float64
) {
    TileConfig config;

    config.grid_size_x = grid_x;
    config.grid_size_y = grid_y;
    config.grid_size_z = grid_z;

    // Auto-detect available memory if not specified
    if (available_vram == 0) {
        available_vram = queryAvailableGPUMemory();
        if (available_vram == 0) {
            // Fallback to conservative estimate if query failed
            available_vram = 8ULL * 1024 * 1024 * 1024;  // 8 GB
            std::cerr << "Warning: Using conservative 8GB memory estimate" << std::endl;
        }
    }

    std::cout << "Available GPU memory: " << (available_vram / (1024.0 * 1024.0))
              << " MB" << std::endl;

    // Find maximum dimension
    int max_dim = std::max({grid_x, grid_y, grid_z});

    // Find optimal tile size
    config.tile_size = findOptimalTileSize(
        max_dim, available_vram, num_grid_types, use_float64
    );

    std::cout << "Selected tile size: " << config.tile_size << "³" << std::endl;

    // Calculate overlap
    config.overlap = calculateOverlap(config.tile_size);

    // Calculate number of tiles needed in each dimension
    // We need to cover the entire grid, accounting for overlap
    auto calc_num_tiles = [](int grid_size, int tile_size, int overlap) -> int {
        if (grid_size <= tile_size) {
            return 1;  // Single tile covers entire dimension
        }

        // Effective tile size (excluding overlap that will be reused)
        int effective_tile_size = tile_size - 2 * overlap;

        // Number of tiles needed
        int num_tiles = (grid_size + effective_tile_size - 1) / effective_tile_size;

        return num_tiles;
    };

    config.num_tiles_x = calc_num_tiles(grid_x, config.tile_size, config.overlap);
    config.num_tiles_y = calc_num_tiles(grid_y, config.tile_size, config.overlap);
    config.num_tiles_z = calc_num_tiles(grid_z, config.tile_size, config.overlap);

    std::cout << "Tiling configuration: "
              << config.num_tiles_x << " × "
              << config.num_tiles_y << " × "
              << config.num_tiles_z << " = "
              << config.total_tiles() << " tiles" << std::endl;

    if (config.is_single_tile()) {
        std::cout << "Using single tile (no tiling overhead)" << std::endl;
    } else {
        std::cout << "Using overlap of " << config.overlap << " points per side" << std::endl;
    }

    // Estimate total memory usage
    size_t estimated_mem = estimateMemoryRequired(
        config.tile_size, num_grid_types, use_float64, true
    );
    std::cout << "Estimated memory usage: "
              << (estimated_mem / (1024.0 * 1024.0)) << " MB" << std::endl;

    return config;
}

TileManager::TileBounds TileManager::getTileBounds(
    const TileConfig& config,
    int tile_idx_x,
    int tile_idx_y,
    int tile_idx_z
) const {
    TileBounds bounds;

    // Check for single-tile configuration
    bool is_single_tile = config.is_single_tile();

    // Effective tile size (portion that doesn't overlap with neighbors)
    int effective_tile_size = is_single_tile ? config.tile_size : (config.tile_size - 2 * config.overlap);

    // Calculate start position in full grid
    bounds.start_x = tile_idx_x * effective_tile_size;
    bounds.start_y = tile_idx_y * effective_tile_size;
    bounds.start_z = tile_idx_z * effective_tile_size;

    // For tiles after the first, we need to include overlap before
    if (tile_idx_x > 0) bounds.start_x -= config.overlap;
    if (tile_idx_y > 0) bounds.start_y -= config.overlap;
    if (tile_idx_z > 0) bounds.start_z -= config.overlap;

    // Ensure start is not negative
    bounds.start_x = std::max(0, bounds.start_x);
    bounds.start_y = std::max(0, bounds.start_y);
    bounds.start_z = std::max(0, bounds.start_z);

    // Calculate tile size (with overlap)
    bounds.size_x = std::min(config.tile_size, config.grid_size_x - bounds.start_x);
    bounds.size_y = std::min(config.tile_size, config.grid_size_y - bounds.start_y);
    bounds.size_z = std::min(config.tile_size, config.grid_size_z - bounds.start_z);

    // Valid region within tile
    bool is_first_x = (tile_idx_x == 0);
    bool is_first_y = (tile_idx_y == 0);
    bool is_first_z = (tile_idx_z == 0);

    // Valid region starts after overlap (unless first tile)
    bounds.valid_start_x = is_first_x ? 0 : config.overlap;
    bounds.valid_start_y = is_first_y ? 0 : config.overlap;
    bounds.valid_start_z = is_first_z ? 0 : config.overlap;

    // Output position in final grid
    bounds.output_start_x = tile_idx_x * effective_tile_size;
    bounds.output_start_y = tile_idx_y * effective_tile_size;
    bounds.output_start_z = tile_idx_z * effective_tile_size;

    // Valid size is what we need to cover from output_start
    int remaining_x = config.grid_size_x - bounds.output_start_x;
    int remaining_y = config.grid_size_y - bounds.output_start_y;
    int remaining_z = config.grid_size_z - bounds.output_start_z;

    bounds.valid_size_x = std::min(effective_tile_size, remaining_x);
    bounds.valid_size_y = std::min(effective_tile_size, remaining_y);
    bounds.valid_size_z = std::min(effective_tile_size, remaining_z);

    return bounds;
}

} // namespace bpmfwfft
