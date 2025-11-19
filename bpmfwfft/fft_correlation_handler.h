#ifndef BPMFWFFT_FFT_CORRELATION_HANDLER_H
#define BPMFWFFT_FFT_CORRELATION_HANDLER_H

#include "fft_plan_manager.h"
#include "fft_tile_manager.h"
#include <cufft.h>
#include <vector>
#include <memory>

namespace bpmfwfft {

/**
 * Handles FFT-based correlation between receptor and ligand grids.
 *
 * Workflow:
 * 1. Precompute receptor FFT (once per receptor, tiled if needed)
 * 2. For each ligand:
 *    a. Compute ligand FFT (tiled if needed)
 *    b. Multiply: receptor_FFT * conj(ligand_FFT)
 *    c. Inverse FFT to get correlation
 *    d. Accumulate energy from correlation grid
 *
 * Memory management:
 * - Supports tiled processing for large grids
 * - Reuses buffers across ligands for efficiency
 * - Manages CUDA streams for potential overlap
 */
class FFTCorrelationHandler {
public:
    FFTCorrelationHandler();
    ~FFTCorrelationHandler();

    /**
     * Initialize handler for given grid dimensions.
     *
     * @param grid_nx, grid_ny, grid_nz: Grid dimensions
     * @param max_tile_size: Maximum tile size (0 = auto-determine)
     * @param overlap: Overlap size for tiling (default: 16)
     */
    void initialize(
        int grid_nx, int grid_ny, int grid_nz,
        int max_tile_size = 0,
        int overlap = 16
    );

    /**
     * Precompute receptor FFT (all tiles).
     * This should be called once per receptor grid.
     *
     * @param receptor_grid: Receptor grid on host [nz, ny, nx]
     */
    void precomputeReceptorFFT(const float* receptor_grid);

    /**
     * Compute correlation energy between precomputed receptor FFT and ligand grid.
     *
     * @param ligand_grid: Ligand grid on host [nz, ny, nx]
     * @return Total correlation energy (sum over all grid points)
     */
    double computeCorrelationEnergy(const float* ligand_grid);

    /**
     * Get the full correlation grid (for debugging/visualization).
     * Must be called after computeCorrelationEnergy().
     *
     * @param correlation_grid: Output buffer [nz, ny, nx]
     */
    void getCorrelationGrid(float* correlation_grid);

    /**
     * Cleanup GPU resources.
     */
    void cleanup();

    /**
     * Check if handler is initialized.
     */
    bool isInitialized() const { return initialized_; }

    /**
     * Get memory usage estimate in bytes.
     */
    size_t getMemoryUsage() const;

private:
    // Configuration
    bool initialized_;
    int grid_nx_, grid_ny_, grid_nz_;
    int grid_size_;

    // Tiling configuration
    std::unique_ptr<TileManager> tile_manager_;
    TileManager::TileConfig tile_config_;
    int num_tiles_;

    // FFT plan manager
    std::unique_ptr<FFTPlanManager> fft_manager_;

    // GPU buffers
    // Per-tile buffers (reused for each tile)
    float* d_tile_real_;           // Tile in real space
    cufftComplex* d_tile_complex_; // Tile in frequency space

    // Receptor FFT storage (all tiles)
    std::vector<cufftComplex*> d_receptor_fft_tiles_;

    // Output accumulation
    float* d_correlation_grid_;    // Full correlation grid on GPU

    // Temporary buffers
    cufftComplex* d_product_;      // Product of receptor * conj(ligand)

    // Helper methods
    void allocateBuffers();
    void freeBuffers();
    void computeTileFFT(const float* h_grid, int tile_idx, cufftComplex* d_fft_out);
    void accumulateCorrelationTile(const cufftComplex* d_tile_fft, int tile_idx);
    double sumGridEnergy(const float* d_grid, int size);
    void getTileIndices3D(int tile_idx, int& tx, int& ty, int& tz) const;
};

} // namespace bpmfwfft

#endif // BPMFWFFT_FFT_CORRELATION_HANDLER_H
