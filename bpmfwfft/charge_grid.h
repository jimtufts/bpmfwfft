#ifndef CHARGE_GRID_H
#define CHARGE_GRID_H

#include <vector>
#include <string>
#include <cstdint>
#include <tuple>

// Main function to calculate the solvent grid
std::vector<std::vector<std::vector<double>>> cal_solvent_grid(
    const std::vector<std::vector<double>>& crd,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<int64_t>& grid_counts,
    const std::vector<double>& vdw_radii);

// Helper functions
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

std::vector<int64_t> lower_corner_of_containing_cube(
    const std::vector<double>& atom_coordinate,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<double>& spacing
);

std::vector<double> get_corner_crd(
    const std::vector<int64_t>& corner,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z
);

double distance(const std::vector<double>& x, const std::vector<double>& y);

bool is_in_grid(
    const std::vector<double>& point,
    const std::vector<double>& origin,
    const std::vector<double>& upper_corner
);

std::vector<std::vector<std::vector<double>>> cal_charge_grid(
    const std::vector<std::vector<double>>& crd,
    const std::vector<double>& charges,
    const std::string& name,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<int64_t>& upper_most_corner,
    const std::vector<double>& spacing,
    const std::vector<std::vector<int64_t>>& eight_corner_shifts,
    const std::vector<std::vector<int64_t>>& six_corner_shifts);

bool is_row_in_matrix(const std::vector<int64_t>& row, const std::vector<std::vector<int64_t>>& matrix);

std::vector<std::vector<int64_t>> get_ten_corners(
    const std::vector<double>& atom_coordinate,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<int64_t>& upper_most_corner,
    const std::vector<double>& spacing,
    const std::vector<std::vector<int64_t>>& eight_corner_shifts,
    const std::vector<std::vector<int64_t>>& six_corner_shifts,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z);

std::vector<double> get_corner_crd(
    const std::vector<int64_t>& corner,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z);

std::tuple<std::vector<std::vector<int64_t>>, int64_t, int64_t> containing_cube(
    const std::vector<double>& atom_coordinate,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<double>& spacing,
    const std::vector<std::vector<int64_t>>& eight_corner_shifts,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z);

#endif // CHARGE_GRID_H
