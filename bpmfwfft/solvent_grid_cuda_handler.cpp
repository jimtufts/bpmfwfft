// solvent_grid_cuda_handler.cpp
#include "solvent_grid_cuda_handler.h"
#include <stdexcept>

extern "C" {
    void compute_solvent_grid_cuda(
        const float* atom_coords,
        const float* vdw_radii,
        const float* grid_x,
        const float* grid_y,
        const float* grid_z,
        float* grid_output,
        int n_atoms,
        int nx, int ny, int nz
    );
}

SolventGridCudaHandler::SolventGridCudaHandler() {
}

SolventGridCudaHandler::~SolventGridCudaHandler() {
}

std::vector<std::vector<std::vector<float>>> SolventGridCudaHandler::compute_solvent_grid(
    const std::vector<std::vector<float>>& atom_coords,
    const std::vector<float>& vdw_radii,
    const std::vector<float>& grid_x,
    const std::vector<float>& grid_y,
    const std::vector<float>& grid_z)
{
    int n_atoms = atom_coords.size();
    int nx = grid_x.size();
    int ny = grid_y.size();
    int nz = grid_z.size();

    if (n_atoms == 0 || nx == 0 || ny == 0 || nz == 0) {
        throw std::runtime_error("Invalid grid dimensions or empty atom list");
    }

    if (vdw_radii.size() != n_atoms) {
        throw std::runtime_error("VDW radii size does not match number of atoms");
    }

    // Flatten atom coordinates to 1D array
    std::vector<float> flat_coords(n_atoms * 3);
    for (int i = 0; i < n_atoms; ++i) {
        if (atom_coords[i].size() != 3) {
            throw std::runtime_error("Each atom must have 3 coordinates");
        }
        flat_coords[i * 3 + 0] = atom_coords[i][0];
        flat_coords[i * 3 + 1] = atom_coords[i][1];
        flat_coords[i * 3 + 2] = atom_coords[i][2];
    }

    // Allocate output grid
    int total_points = nx * ny * nz;
    std::vector<float> grid_flat(total_points);

    // Call CUDA function
    compute_solvent_grid_cuda(
        flat_coords.data(),
        vdw_radii.data(),
        grid_x.data(),
        grid_y.data(),
        grid_z.data(),
        grid_flat.data(),
        n_atoms,
        nx, ny, nz
    );

    // Convert flat grid to 3D structure [nz, ny, nx]
    std::vector<std::vector<std::vector<float>>> grid(
        nz, std::vector<std::vector<float>>(
            ny, std::vector<float>(nx)
        )
    );

    for (int i = 0; i < nz; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nx; ++k) {
                grid[i][j][k] = grid_flat[i * ny * nx + j * nx + k];
            }
        }
    }

    return grid;
}
