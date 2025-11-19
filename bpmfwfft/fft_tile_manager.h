#ifndef FFT_TILE_MANAGER_H
#define FFT_TILE_MANAGER_H

#include <cstddef>
#include <vector>

namespace bpmfwfft {

/**
 * Manages tiling strategy for large FFT operations
 * Determines optimal tile sizes based on available GPU memory
 * Handles tile bounds calculation with overlap regions
 */
class TileManager {
public:
    /**
     * Configuration for a tiled grid
     */
    struct TileConfig {
        // Tile dimensions (cubic tiles for simplicity)
        int tile_size;

        // Overlap per side (for handling boundaries in convolution)
        int overlap;

        // Number of tiles in each dimension
        int num_tiles_x;
        int num_tiles_y;
        int num_tiles_z;

        // Original grid dimensions
        int grid_size_x;
        int grid_size_y;
        int grid_size_z;

        // Total number of tiles
        int total_tiles() const {
            return num_tiles_x * num_tiles_y * num_tiles_z;
        }

        // Check if this is a single-tile configuration (no tiling needed)
        bool is_single_tile() const {
            return num_tiles_x == 1 && num_tiles_y == 1 && num_tiles_z == 1;
        }
    };

    /**
     * Bounds for a specific tile (including overlap and valid regions)
     */
    struct TileBounds {
        // Start position in full grid (including overlap)
        int start_x, start_y, start_z;

        // Size of tile (including overlap)
        int size_x, size_y, size_z;

        // Valid region within tile (excluding overlap)
        // These are offsets within the tile
        int valid_start_x, valid_start_y, valid_start_z;
        int valid_size_x, valid_size_y, valid_size_z;

        // Position where valid region goes in output grid
        int output_start_x, output_start_y, output_start_z;

        // Total points in tile and valid region
        size_t total_points() const {
            return static_cast<size_t>(size_x) * size_y * size_z;
        }

        size_t valid_points() const {
            return static_cast<size_t>(valid_size_x) * valid_size_y * valid_size_z;
        }
    };

    TileManager();
    ~TileManager();

    /**
     * Determine optimal tiling configuration based on available memory
     *
     * @param grid_x, grid_y, grid_z: Full grid dimensions
     * @param available_vram: Available GPU memory in bytes (0 = auto-detect)
     * @param num_grid_types: Number of grid types to process (affects memory)
     * @param use_float64: Whether to use double precision (affects memory)
     * @return TileConfig with optimal settings
     */
    TileConfig determineTiling(
        int grid_x, int grid_y, int grid_z,
        size_t available_vram = 0,
        int num_grid_types = 6,
        bool use_float64 = false
    );

    /**
     * Get bounds for a specific tile
     *
     * @param config: Tile configuration
     * @param tile_idx_x, tile_idx_y, tile_idx_z: Tile indices
     * @return TileBounds for the specified tile
     */
    TileBounds getTileBounds(
        const TileConfig& config,
        int tile_idx_x,
        int tile_idx_y,
        int tile_idx_z
    ) const;

    /**
     * Estimate memory required for a given tile configuration
     *
     * @param tile_size: Size of tiles
     * @param num_grid_types: Number of grid types
     * @param use_float64: Whether using double precision
     * @param include_receptor_ffts: Whether to include persistent receptor FFTs
     * @return Estimated memory in bytes
     */
    static size_t estimateMemoryRequired(
        int tile_size,
        int num_grid_types = 6,
        bool use_float64 = false,
        bool include_receptor_ffts = true
    );

    /**
     * Query available GPU memory
     *
     * @return Available GPU memory in bytes
     */
    static size_t queryAvailableGPUMemory();

private:
    /**
     * Calculate overlap size for a given tile size
     * Conservative estimate for correlation operations
     */
    int calculateOverlap(int tile_size) const;

    /**
     * Find largest tile size that fits in memory
     */
    int findOptimalTileSize(
        int max_dimension,
        size_t available_vram,
        int num_grid_types,
        bool use_float64
    ) const;
};

} // namespace bpmfwfft

#endif // FFT_TILE_MANAGER_H
