#ifndef POTENTIAL_GRID_CUDA_H
#define POTENTIAL_GRID_CUDA_H

#include <cstddef>
#include <cstdint>

/**
 * CUDA kernel for computing potential grids
 *
 * @param d_atom_coordinates Device pointer to atom coordinates [natoms * 3]
 * @param d_charges Device pointer to atomic charges [natoms]
 * @param d_lj_sigma Device pointer to LJ sigma parameters [natoms]
 * @param d_vdw_radii Device pointer to VDW radii [natoms]
 * @param d_clash_radii Device pointer to clash radii [natoms]
 * @param d_molecule_sasa Device pointer to SASA values [natoms]
 * @param d_grid_x Device pointer to grid X coordinates [grid_x_size]
 * @param d_grid_y Device pointer to grid Y coordinates [grid_y_size]
 * @param d_grid_z Device pointer to grid Z coordinates [grid_z_size]
 * @param d_output_grid Device pointer to output grid [grid_x_size * grid_y_size * grid_z_size]
 * @param natoms Number of atoms
 * @param grid_x_size Grid X dimension
 * @param grid_y_size Grid Y dimension
 * @param grid_z_size Grid Z dimension
 * @param exponent Exponent for potential calculation (0.5=electrostatic, 3=LJa, 6=LJr, 1=water)
 * @param grid_type Grid type identifier (0=electrostatic/LJa/LJr, 1=water)
 */
void launch_potential_grid_kernel(
    const double* d_atom_coordinates,
    const double* d_charges,
    const double* d_lj_sigma,
    const double* d_vdw_radii,
    const double* d_clash_radii,
    const double* d_molecule_sasa,
    const double* d_grid_x,
    const double* d_grid_y,
    const double* d_grid_z,
    double* d_output_grid,
    std::size_t natoms,
    std::size_t grid_x_size,
    std::size_t grid_y_size,
    std::size_t grid_z_size,
    double exponent,
    int grid_type
);

/**
 * CUDA kernel for applying clash radius masking
 *
 * @param d_atom_coordinates Device pointer to atom coordinates [natoms * 3]
 * @param d_clash_radii Device pointer to clash radii [natoms]
 * @param d_grid_x Device pointer to grid X coordinates [grid_x_size]
 * @param d_grid_y Device pointer to grid Y coordinates [grid_y_size]
 * @param d_grid_z Device pointer to grid Z coordinates [grid_z_size]
 * @param d_output_grid Device pointer to output grid [grid_x_size * grid_y_size * grid_z_size]
 * @param natoms Number of atoms
 * @param grid_x_size Grid X dimension
 * @param grid_y_size Grid Y dimension
 * @param grid_z_size Grid Z dimension
 */
void launch_clash_mask_kernel(
    const double* d_atom_coordinates,
    const double* d_clash_radii,
    const double* d_grid_x,
    const double* d_grid_y,
    const double* d_grid_z,
    double* d_output_grid,
    std::size_t natoms,
    std::size_t grid_x_size,
    std::size_t grid_y_size,
    std::size_t grid_z_size
);

#endif // POTENTIAL_GRID_CUDA_H
