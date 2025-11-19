#ifndef POTENTIAL_GRID_H
#define POTENTIAL_GRID_H

#include <vector>
#include <string>
#include <cstdint>

/**
 * Calculate potential grid for various interaction types.
 *
 * Supported grid types:
 * - "electrostatic": Coulombic potential (exponent = 0.5)
 * - "LJr": Lennard-Jones repulsive (exponent = 6)
 * - "LJa": Lennard-Jones attractive (exponent = 3)
 * - "water": Water accessibility (exponent = 1)
 * - "occupancy": Occupancy grid (binary within radius)
 * - "sasa": Solvent accessible surface area (binary within radius)
 *
 * @param name Grid type name
 * @param crd Atom coordinates [natoms][3]
 * @param grid_x X-axis grid coordinates
 * @param grid_y Y-axis grid coordinates
 * @param grid_z Z-axis grid coordinates
 * @param origin_crd Grid origin coordinates [3]
 * @param upper_most_corner_crd Upper corner coordinates [3]
 * @param upper_most_corner Upper corner indices [3]
 * @param spacing Grid spacing [3]
 * @param grid_counts Grid dimensions [3]
 * @param charges Atomic charges
 * @param lj_sigma Lennard-Jones sigma parameters
 * @param vdw_radii Van der Waals radii
 * @param clash_radii Clash radii for masking
 * @param molecule_sasa SASA values per atom (for water grid)
 * @return 3D grid of potential values
 */
std::vector<std::vector<std::vector<double>>> cal_potential_grid(
    const std::string& name,
    const std::vector<std::vector<double>>& crd,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<int64_t>& upper_most_corner,
    const std::vector<double>& spacing,
    const std::vector<int64_t>& grid_counts,
    const std::vector<double>& charges,
    const std::vector<double>& lj_sigma,
    const std::vector<double>& vdw_radii,
    const std::vector<double>& clash_radii,
    const std::vector<double>& molecule_sasa,
    const std::vector<int64_t>& atom_list
);

/**
 * Helper function: corners within radius (re-exported from charge_grid.h)
 */
std::vector<std::vector<int64_t>> corners_within_radius(
    const std::vector<double>& atom_coordinate,
    double radius,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<int64_t>& grid_counts,
    const std::vector<double>& spacing,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z
);

#endif // POTENTIAL_GRID_H
