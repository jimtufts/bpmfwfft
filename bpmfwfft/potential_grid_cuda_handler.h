#ifndef POTENTIAL_GRID_CUDA_HANDLER_H
#define POTENTIAL_GRID_CUDA_HANDLER_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>

class PotentialGridCUDAHandler {
public:
    PotentialGridCUDAHandler();
    ~PotentialGridCUDAHandler();

    void initialize(
        std::size_t natoms,
        std::size_t grid_x_size,
        std::size_t grid_y_size,
        std::size_t grid_z_size
    );

    void cleanup();

    void calculatePotentialGrid(
        const std::string& grid_name,
        const double* atom_coordinates,
        const double* grid_x,
        const double* grid_y,
        const double* grid_z,
        const double* charges,
        const double* lj_sigma,
        const double* vdw_radii,
        const double* clash_radii,
        const double* molecule_sasa,
        double* output_grid
    );

private:
    // Device memory pointers
    double *d_atom_coordinates;
    double *d_charges;
    double *d_lj_sigma;
    double *d_vdw_radii;
    double *d_clash_radii;
    double *d_molecule_sasa;
    double *d_grid_x;
    double *d_grid_y;
    double *d_grid_z;
    double *d_output_grid;

    // Grid dimensions
    std::size_t natoms;
    std::size_t grid_x_size;
    std::size_t grid_y_size;
    std::size_t grid_z_size;
    std::size_t grid_total_size;

    bool is_initialized;

    void allocateMemory();
    void freeMemory();

    void copyToDevice(
        const double* atom_coordinates,
        const double* grid_x,
        const double* grid_y,
        const double* grid_z,
        const double* charges,
        const double* lj_sigma,
        const double* vdw_radii,
        const double* clash_radii,
        const double* molecule_sasa
    );

    void copyFromDevice(double* output_grid);
};

#endif // POTENTIAL_GRID_CUDA_HANDLER_H
