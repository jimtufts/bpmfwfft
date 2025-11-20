// solvent_grid_cuda_handler.h
#ifndef SOLVENT_GRID_CUDA_HANDLER_H
#define SOLVENT_GRID_CUDA_HANDLER_H

#include <vector>

class SolventGridCudaHandler {
public:
    SolventGridCudaHandler();
    ~SolventGridCudaHandler();

    // Compute solvent (occupancy) grid on GPU
    std::vector<std::vector<std::vector<float>>> compute_solvent_grid(
        const std::vector<std::vector<float>>& atom_coords,  // [n_atoms, 3]
        const std::vector<float>& vdw_radii,                 // [n_atoms]
        const std::vector<float>& grid_x,                    // [nx]
        const std::vector<float>& grid_y,                    // [ny]
        const std::vector<float>& grid_z                     // [nz]
    );

private:
    // No state needed - stateless computation
};

#endif // SOLVENT_GRID_CUDA_HANDLER_H
