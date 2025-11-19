#include "potential_grid.h"
#include "charge_grid.h"  // For helper functions like corners_within_radius
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    const std::vector<int64_t>& atom_list) {

    // Input validation
    if (crd.empty() || grid_x.empty() || grid_y.empty() || grid_z.empty()) {
        throw std::runtime_error("Empty input arrays");
    }
    if (origin_crd.size() != 3 || upper_most_corner_crd.size() != 3 ||
        spacing.size() != 3 || grid_counts.size() != 3) {
        throw std::runtime_error("Invalid grid parameters");
    }

    int64_t i_max = static_cast<int64_t>(grid_x.size());
    int64_t j_max = static_cast<int64_t>(grid_y.size());
    int64_t k_max = static_cast<int64_t>(grid_z.size());
    int64_t natoms = static_cast<int64_t>(crd.size());

    // Initialize grid
    std::vector<std::vector<std::vector<double>>> grid(i_max,
        std::vector<std::vector<double>>(j_max,
        std::vector<double>(k_max, 0.0)));

    // Handle different grid types
    if (name == "LJa" || name == "LJr" || name == "electrostatic" || name == "water") {
        // Determine exponent based on grid type
        double exponent;
        if (name == "LJa") {
            exponent = 3.0;
        } else if (name == "LJr") {
            exponent = 6.0;
        } else if (name == "electrostatic") {
            exponent = 0.5;
        } else { // water
            exponent = 1.0;
        }

        #ifdef _OPENMP
        #pragma omp parallel
        {
            // Thread-local grid for accumulation
            std::vector<std::vector<std::vector<double>>> local_grid(i_max,
                std::vector<std::vector<double>>(j_max,
                std::vector<double>(k_max, 0.0)));

            #pragma omp for schedule(dynamic)
            for (int64_t atom_ind = 0; atom_ind < natoms; ++atom_ind) {
                const std::vector<double>& atom_coordinate = crd[atom_ind];

                // Get atom-specific parameters
                double charge, lj_diameter, clash_radius;

                if (name == "water") {
                    // For water grid, use SASA and VDW radii
                    // Note: water grid accumulates contributions from multiple atoms
                    charge = molecule_sasa[atom_ind];
                    lj_diameter = vdw_radii[atom_ind];
                    double surface_layer = lj_diameter + 1.4; // probe radius

                    // Temporary grid for this atom (reset for each atom like Cython)
                    std::vector<std::vector<std::vector<double>>> grid_tmp(i_max,
                        std::vector<std::vector<double>>(j_max,
                        std::vector<double>(k_max, 0.0)));

                    // Get corners within surface layer
                    auto corners = corners_within_radius(atom_coordinate, surface_layer,
                                                        origin_crd, upper_most_corner_crd,
                                                        grid_counts, spacing,
                                                        grid_x, grid_y, grid_z);

                    // Set grid points to 1 within surface layer
                    for (const auto& corner : corners) {
                        int64_t i = corner[0];
                        int64_t j = corner[1];
                        int64_t k = corner[2];
                        if (i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max) {
                            grid_tmp[i][j][k] = 1.0;
                        }
                    }

                    // Accumulate into local grid
                    for (int64_t i = 0; i < i_max; ++i) {
                        for (int64_t j = 0; j < j_max; ++j) {
                            for (int64_t k = 0; k < k_max; ++k) {
                                local_grid[i][j][k] += grid_tmp[i][j][k];
                            }
                        }
                    }
                } else {
                    // For LJa, LJr, electrostatic
                    charge = charges[atom_ind];
                    lj_diameter = lj_sigma[atom_ind];
                    clash_radius = clash_radii[atom_ind];

                    // Pre-compute distance squared arrays for efficiency
                    std::vector<double> dx2(i_max);
                    std::vector<double> dy2(j_max);
                    std::vector<double> dz2(k_max);

                    for (int64_t i = 0; i < i_max; ++i) {
                        double dx = atom_coordinate[0] - grid_x[i];
                        dx2[i] = dx * dx;
                    }
                    for (int64_t j = 0; j < j_max; ++j) {
                        double dy = atom_coordinate[1] - grid_y[j];
                        dy2[j] = dy * dy;
                    }
                    for (int64_t k = 0; k < k_max; ++k) {
                        double dz = atom_coordinate[2] - grid_z[k];
                        dz2[k] = dz * dz;
                    }

                    // Temporary grid for this atom
                    std::vector<std::vector<std::vector<double>>> grid_tmp(i_max,
                        std::vector<std::vector<double>>(j_max,
                        std::vector<double>(k_max, 0.0)));

                    // Calculate potential at each grid point
                    for (int64_t i = 0; i < i_max; ++i) {
                        double dx_tmp = dx2[i];
                        for (int64_t j = 0; j < j_max; ++j) {
                            double dy_tmp = dy2[j];
                            for (int64_t k = 0; k < k_max; ++k) {
                                double d_squared = dx_tmp + dy_tmp + dz2[k];
                                double d = std::pow(d_squared, exponent);
                                if (d > 1e-10) {  // Avoid division by zero
                                    grid_tmp[i][j][k] = charge / d;
                                }
                            }
                        }
                    }

                    // Apply clash radius masking
                    auto corners = corners_within_radius(atom_coordinate, clash_radius,
                                                        origin_crd, upper_most_corner_crd,
                                                        grid_counts, spacing,
                                                        grid_x, grid_y, grid_z);

                    for (const auto& corner : corners) {
                        int64_t i = corner[0];
                        int64_t j = corner[1];
                        int64_t k = corner[2];
                        if (i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max) {
                            grid_tmp[i][j][k] = 0.0;
                        }
                    }

                    // Accumulate into local grid
                    for (int64_t i = 0; i < i_max; ++i) {
                        for (int64_t j = 0; j < j_max; ++j) {
                            for (int64_t k = 0; k < k_max; ++k) {
                                local_grid[i][j][k] += grid_tmp[i][j][k];
                            }
                        }
                    }
                }
            }

            // Merge local grids into main grid (critical section)
            #pragma omp critical
            {
                for (int64_t i = 0; i < i_max; ++i) {
                    for (int64_t j = 0; j < j_max; ++j) {
                        for (int64_t k = 0; k < k_max; ++k) {
                            grid[i][j][k] += local_grid[i][j][k];
                        }
                    }
                }
            }
        }
        #else
        // Non-OpenMP version (sequential)
        for (int64_t atom_ind = 0; atom_ind < natoms; ++atom_ind) {
            const std::vector<double>& atom_coordinate = crd[atom_ind];

            double charge, lj_diameter, clash_radius;

            if (name == "water") {
                charge = molecule_sasa[atom_ind];
                lj_diameter = vdw_radii[atom_ind];
                double surface_layer = lj_diameter + 1.4;

                std::vector<std::vector<std::vector<double>>> grid_tmp(i_max,
                    std::vector<std::vector<double>>(j_max,
                    std::vector<double>(k_max, 0.0)));

                auto corners = corners_within_radius(atom_coordinate, surface_layer,
                                                    origin_crd, upper_most_corner_crd,
                                                    grid_counts, spacing,
                                                    grid_x, grid_y, grid_z);

                for (const auto& corner : corners) {
                    int64_t i = corner[0];
                    int64_t j = corner[1];
                    int64_t k = corner[2];
                    if (i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max) {
                        grid_tmp[i][j][k] = 1.0;
                    }
                }

                for (int64_t i = 0; i < i_max; ++i) {
                    for (int64_t j = 0; j < j_max; ++j) {
                        for (int64_t k = 0; k < k_max; ++k) {
                            grid[i][j][k] += grid_tmp[i][j][k];
                        }
                    }
                }
            } else {
                charge = charges[atom_ind];
                lj_diameter = lj_sigma[atom_ind];
                clash_radius = clash_radii[atom_ind];

                std::vector<double> dx2(i_max);
                std::vector<double> dy2(j_max);
                std::vector<double> dz2(k_max);

                for (int64_t i = 0; i < i_max; ++i) {
                    double dx = atom_coordinate[0] - grid_x[i];
                    dx2[i] = dx * dx;
                }
                for (int64_t j = 0; j < j_max; ++j) {
                    double dy = atom_coordinate[1] - grid_y[j];
                    dy2[j] = dy * dy;
                }
                for (int64_t k = 0; k < k_max; ++k) {
                    double dz = atom_coordinate[2] - grid_z[k];
                    dz2[k] = dz * dz;
                }

                std::vector<std::vector<std::vector<double>>> grid_tmp(i_max,
                    std::vector<std::vector<double>>(j_max,
                    std::vector<double>(k_max, 0.0)));

                for (int64_t i = 0; i < i_max; ++i) {
                    double dx_tmp = dx2[i];
                    for (int64_t j = 0; j < j_max; ++j) {
                        double dy_tmp = dy2[j];
                        for (int64_t k = 0; k < k_max; ++k) {
                            double d_squared = dx_tmp + dy_tmp + dz2[k];
                            double d = std::pow(d_squared, exponent);
                            if (d > 1e-10) {
                                grid_tmp[i][j][k] = charge / d;
                            }
                        }
                    }
                }

                auto corners = corners_within_radius(atom_coordinate, clash_radius,
                                                    origin_crd, upper_most_corner_crd,
                                                    grid_counts, spacing,
                                                    grid_x, grid_y, grid_z);

                for (const auto& corner : corners) {
                    int64_t i = corner[0];
                    int64_t j = corner[1];
                    int64_t k = corner[2];
                    if (i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max) {
                        grid_tmp[i][j][k] = 0.0;
                    }
                }

                for (int64_t i = 0; i < i_max; ++i) {
                    for (int64_t j = 0; j < j_max; ++j) {
                        for (int64_t k = 0; k < k_max; ++k) {
                            grid[i][j][k] += grid_tmp[i][j][k];
                        }
                    }
                }
            }
        }
        #endif
    }
    else if (name == "occupancy" || name == "sasa") {
        // For occupancy/sasa grids, mark grid points within atom radii
        // Use atom_list if provided, otherwise use all atoms

        #ifdef _OPENMP
        #pragma omp parallel
        {
            std::vector<std::vector<std::vector<bool>>> local_grid(i_max,
                std::vector<std::vector<bool>>(j_max,
                std::vector<bool>(k_max, false)));

            #pragma omp for schedule(dynamic)
            for (int64_t idx = 0; idx < static_cast<int64_t>(atom_list.size()); ++idx) {
                int64_t atom_ind = atom_list[idx];
                if (atom_ind < 0 || atom_ind >= natoms) continue;

                const std::vector<double>& atom_coordinate = crd[atom_ind];
                double lj_diameter = clash_radii[atom_ind];

                auto corners = corners_within_radius(atom_coordinate, lj_diameter,
                                                    origin_crd, upper_most_corner_crd,
                                                    grid_counts, spacing,
                                                    grid_x, grid_y, grid_z);

                for (const auto& corner : corners) {
                    int64_t i = corner[0];
                    int64_t j = corner[1];
                    int64_t k = corner[2];
                    if (i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max) {
                        local_grid[i][j][k] = true;
                    }
                }
            }

            #pragma omp critical
            {
                for (int64_t i = 0; i < i_max; ++i) {
                    for (int64_t j = 0; j < j_max; ++j) {
                        for (int64_t k = 0; k < k_max; ++k) {
                            if (local_grid[i][j][k]) {
                                grid[i][j][k] = 1.0;
                            }
                        }
                    }
                }
            }
        }
        #else
        for (int64_t idx = 0; idx < static_cast<int64_t>(atom_list.size()); ++idx) {
            int64_t atom_ind = atom_list[idx];
            if (atom_ind < 0 || atom_ind >= natoms) continue;

            const std::vector<double>& atom_coordinate = crd[atom_ind];
            double lj_diameter = clash_radii[atom_ind];

            auto corners = corners_within_radius(atom_coordinate, lj_diameter,
                                                origin_crd, upper_most_corner_crd,
                                                grid_counts, spacing,
                                                grid_x, grid_y, grid_z);

            for (const auto& corner : corners) {
                int64_t i = corner[0];
                int64_t j = corner[1];
                int64_t k = corner[2];
                if (i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max) {
                    grid[i][j][k] = 1.0;
                }
            }
        }
        #endif
    }
    else {
        throw std::runtime_error("Unknown grid name: " + name);
    }

    return grid;
}
