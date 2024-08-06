#include "charge_grid.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cstdint>
#include <Eigen/Dense>
#include <string>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

void print_omp_info() {
    #ifdef _OPENMP
        std::cout << "OpenMP is enabled. Max threads: " << omp_get_max_threads() << std::endl;
    #else
        std::cout << "OpenMP is not enabled." << std::endl;
    #endif
}

std::vector<std::vector<std::vector<double>>> cal_solvent_grid(
    const std::vector<std::vector<double>>& crd,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<int64_t>& grid_counts,
    const std::vector<double>& vdw_radii) {
    
    print_omp_info();
    printf("Starting cal_solvent_grid\n");
    printf("crd size: %zu\n", crd.size());
    printf("grid_x size: %zu\n", grid_x.size());

    if (crd.empty() || grid_x.empty() || grid_y.empty() || grid_z.empty() || 
        origin_crd.size() != 3 || upper_most_corner_crd.size() != 3 || 
        grid_counts.size() != 3 || vdw_radii.size() != crd.size()) {
        throw std::runtime_error("Invalid input parameters");
    }

    int i_max = grid_counts[0];
    int j_max = grid_counts[1];
    int k_max = grid_counts[2];

    printf("Grid dimensions: %d x %d x %d\n", i_max, j_max, k_max);

    std::vector<std::vector<std::vector<double>>> grid(i_max, std::vector<std::vector<double>>(j_max, std::vector<double>(k_max, 0.0)));

    std::vector<double> spacing(3);
    for (int i = 0; i < 3; ++i) {
        spacing[i] = (upper_most_corner_crd[i] - origin_crd[i]) / (grid_counts[i] - 1);
    }

    printf("Processing atoms\n");
    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;        
        std::vector<std::vector<std::vector<double>>> local_grid(i_max, std::vector<std::vector<double>>(j_max, std::vector<double>(k_max, 0.0)));

        #pragma omp for schedule(dynamic)
        for (size_t atom_ind = 0; atom_ind < crd.size(); ++atom_ind) {
            const std::vector<double>& atom_coordinate = crd[atom_ind];
            double lj_diameter = vdw_radii[atom_ind];

            double surface_layer = lj_diameter + 1.4;
            auto corners = corners_within_radius(atom_coordinate, surface_layer, origin_crd,
                                                 upper_most_corner_crd, grid_counts,
                                                 spacing, grid_x, grid_y, grid_z);

            for (const auto& corner : corners) {
                int i = corner[0];
                int j = corner[1];
                int k = corner[2];
                if (i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max) {
                    local_grid[i][j][k] += 1.0;
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < i_max; ++i) {
                for (int j = 0; j < j_max; ++j) {
                    for (int k = 0; k < k_max; ++k) {
                        grid[i][j][k] += local_grid[i][j][k];
                    }
                }
            }
        }
    }
    #else
        for (size_t atom_ind = 0; atom_ind < crd.size(); ++atom_ind) {
            const std::vector<double>& atom_coordinate = crd[atom_ind];
            double lj_diameter = vdw_radii[atom_ind];

            double surface_layer = lj_diameter + 1.4;
            auto corners = corners_within_radius(atom_coordinate, surface_layer, origin_crd,
                                                 upper_most_corner_crd, grid_counts,
                                                 spacing, grid_x, grid_y, grid_z);

            for (const auto& corner : corners) {
                int i = corner[0];
                int j = corner[1];
                int k = corner[2];
                if (i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max) {
                    grid[i][j][k] += 1.0;
                }
            }
        }
    #endif

    printf("cal_solvent_grid completed\n");
    return grid;
}

Eigen::VectorXd nnls(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, double tolerance = 1e-10, int max_iterations = 1000) {
    int m = A.rows();
    int n = A.cols();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd w = A.transpose() * (b - A * x);
    std::vector<int> P;
    std::vector<int> R(n);
    for (int i = 0; i < n; ++i) R[i] = i;

    for (int iter = 0; iter < max_iterations; ++iter) {
        if (R.empty() || w.maxCoeff() <= tolerance) break;

        int j = std::max_element(R.begin(), R.end(), [&w](int i, int j) { return w[i] < w[j]; }) - R.begin();
        int t = R[j];
        P.push_back(t);
        R.erase(R.begin() + j);

        Eigen::MatrixXd AP(m, P.size());
        for (size_t i = 0; i < P.size(); ++i) AP.col(i) = A.col(P[i]);

        Eigen::VectorXd s = AP.colPivHouseholderQr().solve(b);

        while (s.minCoeff() <= 0) {
            double alpha = INFINITY;
            int alpha_index = -1;
            for (size_t i = 0; i < P.size(); ++i) {
                if (s[i] <= 0) {
                    double ratio = x[P[i]] / (x[P[i]] - s[i]);
                    if (ratio < alpha) {
                        alpha = ratio;
                        alpha_index = i;
                    }
                }
            }

            for (size_t i = 0; i < P.size(); ++i) {
                x[P[i]] += alpha * (s[i] - x[P[i]]);
            }

            std::vector<int> to_remove;
            for (size_t i = 0; i < P.size(); ++i) {
                if (std::abs(x[P[i]]) < tolerance) {
                    R.push_back(P[i]);
                    to_remove.push_back(i);
                }
            }

            for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it) {
                P.erase(P.begin() + *it);
            }

            AP = Eigen::MatrixXd(m, P.size());
            for (size_t i = 0; i < P.size(); ++i) AP.col(i) = A.col(P[i]);

            s = AP.colPivHouseholderQr().solve(b);
        }

        for (size_t i = 0; i < P.size(); ++i) {
            x[P[i]] = s[i];
        }

        w = A.transpose() * (b - A * x);
    }

    return x;
}

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
    const std::vector<std::vector<int64_t>>& six_corner_shifts) {

    if (crd.empty() || charges.empty() || name.empty() ||
        grid_x.empty() || grid_y.empty() || grid_z.empty() ||
        origin_crd.size() != 3 || upper_most_corner_crd.size() != 3 ||
        upper_most_corner.size() != 3 || spacing.size() != 3 ||
        crd.size() != charges.size() || crd.size() != name.size()) {
        throw std::runtime_error("Invalid input parameters");
    }

    int64_t i_max = upper_most_corner[0];
    int64_t j_max = upper_most_corner[1];
    int64_t k_max = upper_most_corner[2];

    std::vector<std::vector<std::vector<double>>> grid(i_max,
        std::vector<std::vector<double>>(j_max,
        std::vector<double>(k_max, 0.0)));

    #ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<std::vector<std::vector<double>>> local_grid(i_max,
            std::vector<std::vector<double>>(j_max,
            std::vector<double>(k_max, 0.0)));

        #pragma omp for schedule(dynamic)
        for (int64_t atom_ind = 0; atom_ind < static_cast<int64_t>(crd.size()); ++atom_ind) {
            const auto& atom_coordinate = crd[atom_ind];
            double charge = charges[atom_ind];

            auto ten_corners = get_ten_corners(atom_coordinate, origin_crd, upper_most_corner_crd,
                                             upper_most_corner, spacing, eight_corner_shifts,
                                             six_corner_shifts, grid_x, grid_y, grid_z);

            Eigen::MatrixXd a_matrix = Eigen::MatrixXd::Zero(10, 10);
            Eigen::VectorXd b_vector = Eigen::VectorXd::Zero(10);
            b_vector(0) = charge;
            a_matrix.row(0).setOnes();

            std::vector<std::vector<double>> delta_vectors;
            for (const auto& corner : ten_corners) {
                auto corner_crd = get_corner_crd(corner, grid_x, grid_y, grid_z);
                std::vector<double> delta(3);
                for (int i = 0; i < 3; ++i) {
                    delta[i] = corner_crd[i] - atom_coordinate[i];
                }
                delta_vectors.push_back(delta);
            }

            for (int j = 0; j < 10; ++j) {
                a_matrix(1, j) = delta_vectors[j][0];
                a_matrix(2, j) = delta_vectors[j][1];
                a_matrix(3, j) = delta_vectors[j][2];
            }

            int row = 3;
            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    ++row;
                    for (int k = 0; k < 10; ++k) {
                        a_matrix(row, k) = delta_vectors[k][i] * delta_vectors[k][j];
                    }
                }
            }

            Eigen::VectorXd distributed_charges;
            if (name == "electrostatic") {
                distributed_charges = a_matrix.colPivHouseholderQr().solve(b_vector);
            } else {
                // Use non-negative least squares for Lennard-Jones to prevent singularity issues
                distributed_charges = nnls(a_matrix, b_vector);
            }

            for (size_t i = 0; i < ten_corners.size(); ++i) {
                int64_t l = ten_corners[i][0];
                int64_t m = ten_corners[i][1];
                int64_t n = ten_corners[i][2];
                if (l >= 0 && l < i_max && m >= 0 && m < j_max && n >= 0 && n < k_max) {
                    local_grid[l][m][n] += distributed_charges(i);
                }
            }
        }

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
    for (int64_t atom_ind = 0; atom_ind < static_cast<int64_t>(crd.size()); ++atom_ind) {
        const auto& atom_coordinate = crd[atom_ind];
        double charge = charges[atom_ind];

        auto ten_corners = get_ten_corners(atom_coordinate, origin_crd, upper_most_corner_crd,
                                         upper_most_corner, spacing, eight_corner_shifts,
                                         six_corner_shifts, grid_x, grid_y, grid_z);

        Eigen::MatrixXd a_matrix = Eigen::MatrixXd::Zero(10, 10);
        Eigen::VectorXd b_vector = Eigen::VectorXd::Zero(10);
        b_vector(0) = charge;
        a_matrix.row(0).setOnes();

        std::vector<std::vector<double>> delta_vectors;
        for (const auto& corner : ten_corners) {
            auto corner_crd = get_corner_crd(corner, grid_x, grid_y, grid_z);
            std::vector<double> delta(3);
            for (int i = 0; i < 3; ++i) {
                delta[i] = corner_crd[i] - atom_coordinate[i];
            }
            delta_vectors.push_back(delta);
        }

        for (int j = 0; j < 10; ++j) {
            a_matrix(1, j) = delta_vectors[j][0];
            a_matrix(2, j) = delta_vectors[j][1];
            a_matrix(3, j) = delta_vectors[j][2];
        }

        int row = 3;
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                ++row;
                for (int k = 0; k < 10; ++k) {
                    a_matrix(row, k) = delta_vectors[k][i] * delta_vectors[k][j];
                }
            }
        }

        Eigen::VectorXd distributed_charges;
        if (name == "electrostatic") {
            distributed_charges = a_matrix.colPivHouseholderQr().solve(b_vector);
        } else {
            // Use non-negative least squares for Lennard-Jones to prevent singularity issues
            distributed_charges = nnls(a_matrix, b_vector);
        }

        for (size_t i = 0; i < ten_corners.size(); ++i) {
            int64_t l = ten_corners[i][0];
            int64_t m = ten_corners[i][1];
            int64_t n = ten_corners[i][2];
            if (l >= 0 && l < i_max && m >= 0 && m < j_max && n >= 0 && n < k_max) {
                grid[l][m][n] += distributed_charges(i);
            }
        }
    }
    #endif

    return grid;
}

std::vector<std::vector<int64_t>> corners_within_radius(
    const std::vector<double>& atom_coordinate,
    double radius,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<int64_t>& grid_counts,
    const std::vector<double>& spacing,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z) {

    if (radius <= 0) {
        return {};
    }

    std::vector<int64_t> lower_corner = lower_corner_of_containing_cube(atom_coordinate, origin_crd, upper_most_corner_crd, spacing);

    std::vector<std::vector<int64_t>> corners;

    if (!lower_corner.empty()) {
        std::vector<double> lower_corner_crd = get_corner_crd(lower_corner, grid_x, grid_y, grid_z);
        double r = radius + distance(lower_corner_crd, atom_coordinate);

        std::vector<int64_t> count(3);
        for (int64_t i = 0; i < 3; ++i) {
            count[i] = static_cast<int64_t>(std::ceil(r / spacing[i]));
        }

        for (int64_t i = std::max(static_cast<int64_t>(0), lower_corner[0] - count[0]); 
             i <= std::min(grid_counts[0] - 1, lower_corner[0] + count[0]); ++i) {
            for (int64_t j = std::max(static_cast<int64_t>(0), lower_corner[1] - count[1]); 
                 j <= std::min(grid_counts[1] - 1, lower_corner[1] + count[1]); ++j) {
                for (int64_t k = std::max(static_cast<int64_t>(0), lower_corner[2] - count[2]); 
                     k <= std::min(grid_counts[2] - 1, lower_corner[2] + count[2]); ++k) {
                    std::vector<double> corner_crd = {grid_x[i], grid_y[j], grid_z[k]};
                    if (distance(corner_crd, atom_coordinate) <= radius) {
                        corners.push_back({i, j, k});
                    }
                }
            }
        }
    } else {
        for (int64_t i = 0; i < grid_counts[0]; ++i) {
            for (int64_t j = 0; j < grid_counts[1]; ++j) {
                for (int64_t k = 0; k < grid_counts[2]; ++k) {
                    std::vector<double> corner_crd = {grid_x[i], grid_y[j], grid_z[k]};
                    if (distance(corner_crd, atom_coordinate) <= radius) {
                        corners.push_back({i, j, k});
                    }
                }
            }
        }
    }

    return corners;
}


// Helper function to check if a point is in the grid
bool is_in_grid(const std::vector<double>& point, const std::vector<double>& origin, const std::vector<double>& upper_corner) {
    for (size_t i = 0; i < point.size(); ++i) {
        if (point[i] < origin[i] || point[i] > upper_corner[i]) {
            return false;
        }
    }
    return true;
}

std::vector<int64_t> lower_corner_of_containing_cube(const std::vector<double>& atom_coordinate,
                                                 const std::vector<double>& origin_crd,
                                                 const std::vector<double>& upper_most_corner_crd,
                                                 const std::vector<double>& spacing) {
    if (!is_in_grid(atom_coordinate, origin_crd, upper_most_corner_crd)) {
        return {};
    }
    std::vector<int64_t> lower_corner(atom_coordinate.size());
    for (size_t i = 0; i < atom_coordinate.size(); ++i) {
        lower_corner[i] = static_cast<int64_t>((atom_coordinate[i] - origin_crd[i]) / spacing[i]);
    }
    return lower_corner;
}

double distance(const std::vector<double>& x, const std::vector<double>& y) {
    double d = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double tmp = x[i] - y[i];
        d += tmp * tmp;
    }
    return std::sqrt(d);
}

std::vector<double> get_corner_crd(const std::vector<int64_t>& corner,
                                   const std::vector<double>& grid_x,
                                   const std::vector<double>& grid_y,
                                   const std::vector<double>& grid_z) {
    return {grid_x[corner[0]], grid_y[corner[1]], grid_z[corner[2]]};
}

std::tuple<std::vector<std::vector<int64_t>>, int64_t, int64_t> containing_cube(
    const std::vector<double>& atom_coordinate,
    const std::vector<double>& origin_crd,
    const std::vector<double>& upper_most_corner_crd,
    const std::vector<double>& spacing,
    const std::vector<std::vector<int64_t>>& eight_corner_shifts,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    const std::vector<double>& grid_z) 
{
    if (!is_in_grid(atom_coordinate, origin_crd, upper_most_corner_crd)) {
        return std::make_tuple(std::vector<std::vector<int64_t>>(), 0, 0);
    }

    std::vector<int64_t> lower_corner(3);
    for (size_t i = 0; i < 3; ++i) {
        lower_corner[i] = static_cast<int64_t>((atom_coordinate[i] - origin_crd[i]) / spacing[i]);
    }

    std::vector<std::vector<int64_t>> eight_corners;
    for (const auto& shift : eight_corner_shifts) {
        std::vector<int64_t> corner(3);
        for (size_t i = 0; i < 3; ++i) {
            corner[i] = lower_corner[i] + shift[i];
        }
        eight_corners.push_back(corner);
    }

    std::vector<double> distances;
    for (const auto& corner : eight_corners) {
        std::vector<double> corner_crd = get_corner_crd(corner, grid_x, grid_y, grid_z);
        distances.push_back(distance(corner_crd, atom_coordinate));
    }

    auto nearest_it = std::min_element(distances.begin(), distances.end());
    auto furthest_it = std::max_element(distances.begin(), distances.end());

    int nearest_ind = std::distance(distances.begin(), nearest_it);
    int furthest_ind = std::distance(distances.begin(), furthest_it);

    return std::make_tuple(eight_corners, nearest_ind, furthest_ind);
}

bool is_row_in_matrix(const std::vector<int64_t>& row, const std::vector<std::vector<int64_t>>& matrix) {
    return std::find(matrix.begin(), matrix.end(), row) != matrix.end();
}

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
    const std::vector<double>& grid_z) {
    
    auto [eight_corners, nearest_ind, furthest_ind] = containing_cube(
        atom_coordinate, origin_crd, upper_most_corner_crd, spacing,
        eight_corner_shifts, grid_x, grid_y, grid_z);
    
    if (eight_corners.empty()) {
        throw std::runtime_error("Atom is outside the grid");
    }
    
    const auto& nearest_corner = eight_corners[nearest_ind];
    for (size_t i = 0; i < nearest_corner.size(); ++i) {
        if (nearest_corner[i] == 0 || nearest_corner[i] == upper_most_corner[i]) {
            throw std::runtime_error("The nearest corner is on the grid boundary");
        }
    }
    
    std::vector<std::vector<int64_t>> six_corners;
    for (const auto& shift : six_corner_shifts) {
        std::vector<int64_t> corner(3);
        for (size_t i = 0; i < 3; ++i) {
            corner[i] = nearest_corner[i] + shift[i];
        }
        six_corners.push_back(corner);
    }
    
    std::vector<std::vector<int64_t>> three_corners;
    for (const auto& corner : six_corners) {
        if (!is_row_in_matrix(corner, eight_corners)) {
            three_corners.push_back(corner);
        }
    }
    
    eight_corners.erase(eight_corners.begin() + furthest_ind);
    std::vector<std::vector<int64_t>> ten_corners = eight_corners;
    ten_corners.insert(ten_corners.end(), three_corners.begin(), three_corners.end());
    
    return ten_corners;
}
