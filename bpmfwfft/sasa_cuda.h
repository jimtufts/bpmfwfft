#ifndef SASA_CUDA_H
#define SASA_CUDA_H

#include <cstddef>

/**
 * Launch SASA grid calculation kernel
 *
 * @param d_atom_coords Device pointer to atom coordinates [n_atoms * 3]
 * @param d_atom_radii Device pointer to atom radii (VDW + probe) [n_atoms]
 * @param d_sphere_points Device pointer to sphere points [n_sphere_points * 3]
 * @param d_atom_selection_mask Device pointer to atom selection mask [n_atoms]
 * @param d_output_grid Device pointer to output grid [grid_count_x * grid_count_y * grid_count_z]
 * @param n_atoms Number of atoms
 * @param n_sphere_points Number of sphere points
 * @param grid_count_x Grid size in x dimension
 * @param grid_count_y Grid size in y dimension
 * @param grid_count_z Grid size in z dimension
 * @param grid_spacing Grid spacing
 * @param use_ten_corners If true, use 10-corner interpolation; if false, use simple rounding
 */
void launch_sasa_kernel(
    const float* d_atom_coords,
    const float* d_atom_radii,
    const float* d_sphere_points,
    const int* d_atom_selection_mask,
    float* d_output_grid,
    std::size_t n_atoms,
    std::size_t n_sphere_points,
    int grid_count_x,
    int grid_count_y,
    int grid_count_z,
    float grid_spacing,
    bool use_ten_corners
);

#endif // SASA_CUDA_H
