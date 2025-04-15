#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <Eigen/Dense>

#include "sasa.h"
#include "vectorize.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Point and index structure definitions
struct Point3D {
    float x, y, z;
};

struct GridIndex {
    int x, y, z;
};

// Helper function to get corner coordinates
Point3D get_corner_crd(const GridIndex& corner, const float* grid_x, const float* grid_y, const float* grid_z) {
    return {grid_x[corner.x], grid_y[corner.y], grid_z[corner.z]};
}

// Function to check if a point is in the grid
bool is_in_grid(const Point3D& point, const Point3D& origin, const Point3D& upper_corner) {
    return (point.x >= origin.x && point.x < upper_corner.x &&
            point.y >= origin.y && point.y < upper_corner.y &&
            point.z >= origin.z && point.z < upper_corner.z);
}

// Function to find the lower corner of the containing cube
GridIndex lower_corner_of_containing_cube(const Point3D& atom_coordinate, 
                                        const Point3D& origin_crd, 
                                        const Point3D& upper_most_corner_crd, 
                                        float grid_spacing) {
    GridIndex lower_corner;
    lower_corner.x = static_cast<int>((atom_coordinate.x - origin_crd.x) / grid_spacing);
    lower_corner.y = static_cast<int>((atom_coordinate.y - origin_crd.y) / grid_spacing);
    lower_corner.z = static_cast<int>((atom_coordinate.z - origin_crd.z) / grid_spacing);
    return lower_corner;
}

// Function to calculate the ten corners
std::vector<GridIndex> get_ten_corners(const Point3D& atom_coordinate,
                                    const Point3D& origin_crd,
                                    const Point3D& upper_most_corner_crd,
                                    const int* counts,
                                    float grid_spacing,
                                    const float* grid_x,
                                    const float* grid_y,
                                    const float* grid_z) {
    // Define corner shifts
    const std::array<GridIndex, 8> eight_corner_shifts = {{
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    }};
    
    const std::array<GridIndex, 6> six_corner_shifts = {{
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
    }};
    
    // Check if atom is inside grid
    if (!is_in_grid(atom_coordinate, origin_crd, upper_most_corner_crd)) {
        // Silent failure for out-of-bounds points, just return empty vector
        return {};
    }
    
    // Find the lower corner
    GridIndex lower_corner = lower_corner_of_containing_cube(
        atom_coordinate, origin_crd, upper_most_corner_crd, grid_spacing);
    
    // Generate 8 corners of the cube
    std::vector<GridIndex> eight_corners;
    std::vector<float> distances;
    
    for (const auto& shift : eight_corner_shifts) {
        GridIndex corner = {
            lower_corner.x + shift.x,
            lower_corner.y + shift.y,
            lower_corner.z + shift.z
        };
        
        eight_corners.push_back(corner);
        
        Point3D corner_crd = get_corner_crd(corner, grid_x, grid_y, grid_z);
        float dx = corner_crd.x - atom_coordinate.x;
        float dy = corner_crd.y - atom_coordinate.y;
        float dz = corner_crd.z - atom_coordinate.z;
        distances.push_back(sqrt(dx*dx + dy*dy + dz*dz));
    }
    
    // Find nearest and furthest corners
    int nearest_ind = std::min_element(distances.begin(), distances.end()) - distances.begin();
    int furthest_ind = std::max_element(distances.begin(), distances.end()) - distances.begin();
    
    // Verify the nearest corner is not on the grid boundary
    GridIndex& nearest_corner = eight_corners[nearest_ind];
    if (nearest_corner.x == 0 || nearest_corner.x >= counts[0] - 1 ||
        nearest_corner.y == 0 || nearest_corner.y >= counts[1] - 1 ||
        nearest_corner.z == 0 || nearest_corner.z >= counts[2] - 1) {
        // Silent failure for boundary points, just fall back to simple rounding
        return {};
    }
    
    // Find three additional corners
    std::vector<GridIndex> three_corners;
    for (const auto& shift : six_corner_shifts) {
        GridIndex potential_corner = {
            nearest_corner.x + shift.x,
            nearest_corner.y + shift.y,
            nearest_corner.z + shift.z
        };
        
        // Check if this corner is not in eight_corners
        bool found = false;
        for (const auto& corner : eight_corners) {
            if (corner.x == potential_corner.x && 
                corner.y == potential_corner.y && 
                corner.z == potential_corner.z) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            three_corners.push_back(potential_corner);
            if (three_corners.size() >= 3) break;
        }
    }
    
    // Make sure we found enough corners
    if (three_corners.size() < 3) {
        // Not enough corners, fall back to simple rounding
        return {};
    }
    
    // Combine 7 corners from eight_corners (excluding furthest) + 3 new corners
    std::vector<GridIndex> ten_corners;
    for (size_t i = 0; i < eight_corners.size(); i++) {
        if (i != furthest_ind) {
            ten_corners.push_back(eight_corners[i]);
        }
    }
    
    ten_corners.insert(ten_corners.end(), three_corners.begin(), three_corners.end());
    return ten_corners;
}

/**
 * Calculate the accessible surface area of each atom in a single snapshot
 *
 * Parameters
 * ----------
 * frame : 2d array, shape=[n_atoms, 3]
 *     The coordinates of the nuclei
 * n_atoms : int
 *     the major axis length of frame
 * atom_radii : 1d array, shape=[n_atoms]
 *     the van der waals radii of the atoms PLUS the probe radius
 * sphere_points : 2d array, shape=[n_sphere_points, 3]
 *     a bunch of uniformly distributed points on a sphere
 * n_sphere_points : int
 *    the number of sphere points
 * atom_selection_mask : 1d array, shape[n_atoms]
 *    one index per atom indicating whether the SASA
 *    should be computed for this atom (`atom_selection_mask[i] = 1'
 *    or or not (`atom_selection_mask[i] = 0`)
 * centered_sphere_points : WORK BUFFER 2d array, shape=[n_sphere_points, 3]
 *    empty memory that intermediate calculations can be stored in
 * neighbor_indices : WORK BUFFER 2d array, shape=[n_atoms]
 *    empty memory that intermediate calculations can be stored in
 * NOTE: the point of these work buffers is that if we want to call
 *    this function repreatedly, its more efficient not to keep re-mallocing
 *    these work buffers, but instead just reuse them.
 *
 * areas : 1d array, shape=[n_atoms]
 *     the output buffer to place the results in -- the surface area of each
 *     atom
 */
void asa_frame(const float* frame, const int n_atoms, const float* atom_radii,
               const float* sphere_points, const int n_sphere_points,
               int* neighbor_indices, float* centered_sphere_points,
               const int* atom_selection_mask, float* out_grid,
               const int* counts, const float grid_spacing) {
    // Calculate total number of grid points
    int total_grid_points = counts[0] * counts[1] * counts[2];

    // Initialize the output grid to zero
    for (int i = 0; i < total_grid_points; i++) {
        out_grid[i] = 0.0f;
    }

    float constant = 4.0f * M_PI / n_sphere_points;
    
    // Generate grid coordinate arrays
    std::vector<float> grid_x(counts[0]);
    std::vector<float> grid_y(counts[1]);
    std::vector<float> grid_z(counts[2]);
    
    for (int i = 0; i < counts[0]; i++) {
        grid_x[i] = i * grid_spacing;
    }
    for (int i = 0; i < counts[1]; i++) {
        grid_y[i] = i * grid_spacing;
    }
    for (int i = 0; i < counts[2]; i++) {
        grid_z[i] = i * grid_spacing;
    }
    
    // Define grid boundaries
    Point3D origin_crd = {0.0f, 0.0f, 0.0f};
    Point3D upper_most_corner_crd = {
        (counts[0] - 1) * grid_spacing,
        (counts[1] - 1) * grid_spacing,
        (counts[2] - 1) * grid_spacing
    };

    for (int i = 0; i < n_atoms; i++) {
        // Skip atom if not in selection
        int in_selection = atom_selection_mask[i];
        if (in_selection == 0)
            continue;

        float atom_radius_i = atom_radii[i];
        fvec4 r_i(frame[i*3], frame[i*3+1], frame[i*3+2], 0);

        // Get all the atoms close to atom `i`
        int n_neighbor_indices = 0;
        for (int j = 0; j < n_atoms; j++) {
            if (i == j)
                continue;

            fvec4 r_j(frame[j*3], frame[j*3+1], frame[j*3+2], 0);
            fvec4 r_ij = r_i-r_j;
            float atom_radius_j = atom_radii[j];

            // Look for atoms `j` that are nearby atom `i`
            float radius_cutoff = atom_radius_i+atom_radius_j;
            float radius_cutoff2 = radius_cutoff*radius_cutoff;
            float r2 = dot3(r_ij, r_ij);
            if (r2 < radius_cutoff2) {
                neighbor_indices[n_neighbor_indices]  = j;
                n_neighbor_indices++;
            }
            if (r2 < 1e-10f) {
                printf("ERROR: THIS CODE IS KNOWN TO FAIL WHEN ATOMS ARE VIRTUALLY");
                printf("ON TOP OF ONE ANOTHER. YOU SUPPLIED TWO ATOMS %f", sqrtf(r2));
                printf("APART. QUITTING NOW");
                exit(1);
            }
        }

        // Center the sphere points on atom i
        for (int j = 0; j < n_sphere_points; j++) {
            centered_sphere_points[3*j] = frame[3*i] + atom_radius_i*sphere_points[3*j];
            centered_sphere_points[3*j+1] = frame[3*i+1] + atom_radius_i*sphere_points[3*j+1];
            centered_sphere_points[3*j+2] = frame[3*i+2] + atom_radius_i*sphere_points[3*j+2];
        }

        // Check if each of these points is accessible
        int k_closest_neighbor = 0;
        for (int j = 0; j < n_sphere_points; j++) {
            bool is_accessible = true;
            fvec4 r_j(centered_sphere_points[3*j], centered_sphere_points[3*j+1], centered_sphere_points[3*j+2], 0);

            // Iterate through the sphere points by cycling through them
            // in a circle, starting with k_closest_neighbor and then wrapping
            // around
            for (int k = k_closest_neighbor; k < n_neighbor_indices + k_closest_neighbor; k++) {
                int k_prime = k % n_neighbor_indices;
                float r = atom_radii[neighbor_indices[k_prime]];

                int index = neighbor_indices[k_prime];
                fvec4 r_jk = r_j-fvec4(frame[3*index], frame[3*index+1], frame[3*index+2], 0);
                if (dot3(r_jk, r_jk) < r*r) {
                    k_closest_neighbor = k;
                    is_accessible = false;
                    break;
                }
            }

            if (is_accessible) {
                // Create a Point3D for the accessible sphere point
                Point3D atom_coordinate = {r_j[0], r_j[1], r_j[2]};
                float value = constant * atom_radius_i * atom_radius_i;
                
                // Try to get ten corners for this point
                auto ten_corners = get_ten_corners(
                    atom_coordinate, origin_crd, upper_most_corner_crd, counts,
                    grid_spacing, grid_x.data(), grid_y.data(), grid_z.data());
                
                if (!ten_corners.empty() && ten_corners.size() == 10) {
                    // Use ten corners method
                    try {
                        // Set up the system of equations using Eigen
                        Eigen::MatrixXd a_matrix = Eigen::MatrixXd::Zero(10, 10);
                        Eigen::VectorXd b_vector = Eigen::VectorXd::Zero(10);
                        b_vector(0) = value;
                        a_matrix.row(0).setOnes();
                        
                        // Calculate delta vectors
                        std::vector<std::array<float, 3>> delta_vectors;
                        for (const auto& corner : ten_corners) {
                            Point3D corner_crd = get_corner_crd(corner, grid_x.data(), grid_y.data(), grid_z.data());
                            delta_vectors.push_back({
                                corner_crd.x - atom_coordinate.x,
                                corner_crd.y - atom_coordinate.y,
                                corner_crd.z - atom_coordinate.z
                            });
                        }
                        
                        // Fill the matrix
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
                        
                        // Solve the system
                        Eigen::VectorXd sasa_values = a_matrix.colPivHouseholderQr().solve(b_vector);
                        
                        // Distribute the values to the grid
                        for (size_t i = 0; i < ten_corners.size(); ++i) {
                            const auto& corner = ten_corners[i];
                            if (corner.x >= 0 && corner.x < counts[0] && 
                                corner.y >= 0 && corner.y < counts[1] && 
                                corner.z >= 0 && corner.z < counts[2]) {
                                // Calculate 1D index from 3D coordinates
                                int grid_index = corner.x * counts[1] * counts[2] + corner.y * counts[2] + corner.z;
				out_grid[grid_index] += sasa_values(i);
                            }
                        }
                    } catch (const std::exception& e) {
                        // Fall back to simple rounding if Eigen fails
                        // Snap coordinates to grid
			// / Add warning
    			fprintf(stderr, "Warning: Eigen solver failed for ten corners method at point (%f, %f, %f): %s. "
            			"Falling back to simple rounding method.\n",
            			r_j[0], r_j[1], r_j[2], e.what());
                        float x = roundf(r_j[0] / grid_spacing) * grid_spacing;
                        float y = roundf(r_j[1] / grid_spacing) * grid_spacing;
                        float z = roundf(r_j[2] / grid_spacing) * grid_spacing;

                        // Calculate grid indices
                        int ix = (int)(x / grid_spacing);
                        int iy = (int)(y / grid_spacing);
                        int iz = (int)(z / grid_spacing);

                        // Ensure indices are within bounds
                        if (ix >= 0 && ix < counts[0] && iy >= 0 && iy < counts[1] && iz >= 0 && iz < counts[2]) {
                            // Calculate grid index
                            int grid_index = ix * counts[1] * counts[2] + iy * counts[2] + iz;
			    out_grid[grid_index] += value;
                        }
                    }
                } else {
                    // Fall back to simple rounding method
                    // Snap coordinates to grid
                    float x = roundf(r_j[0] / grid_spacing) * grid_spacing;
                    float y = roundf(r_j[1] / grid_spacing) * grid_spacing;
                    float z = roundf(r_j[2] / grid_spacing) * grid_spacing;

                    // Calculate grid indices
                    int ix = (int)(x / grid_spacing);
                    int iy = (int)(y / grid_spacing);
                    int iz = (int)(z / grid_spacing);

                    // Ensure indices are within bounds
                    if (ix >= 0 && ix < counts[0] && iy >= 0 && iy < counts[1] && iz >= 0 && iz < counts[2]) {
                        // Calculate grid index
                        int grid_index = ix * counts[1] * counts[2] + iy * counts[2] + iz;
                        out_grid[grid_index] += value;
                    }
                }
            }
        }
    }
}

static void generate_sphere_points(float* sphere_points, int n_points)
{
  /*
  // Compute the coordinates of points on a sphere using the
  // Golden Section Spiral algorithm.
  //
  // Parameters
  // ----------
  // sphere_points : array, shape=(n_points, 3)
  //     Empty array of length n_points*3 -- will be filled with the points
  //     as an array in C-order. i.e. sphere_points[3*i], sphere_points[3*i+1]
  //     and sphere_points[3*i+2] are the x,y,z coordinates of the ith point
  // n_pts : int
  //     Number of points to generate on the sphere
  //
  */
  int i;
  float y, r, phi;
  float inc = M_PI * (3.0 - sqrt(5.0));
  float offset = 2.0 / n_points;

  for (i = 0; i < n_points; i++) {
    y = i * offset - 1.0 + (offset / 2.0);
    r = sqrt(1.0 - y*y);
    phi = i * inc;

    sphere_points[3*i] = cos(phi) * r;
    sphere_points[3*i+1] = y;
    sphere_points[3*i+2] = sin(phi) * r;
  }
}

void sasa(const int n_frames, const int n_atoms, const float* xyzlist,
          const float* atom_radii, const int n_sphere_points,
          const int* atom_selection_mask, float* out,
          const int* counts, const float grid_spacing)
{
    int i;

    /* work buffers that will be thread-local */
    int* wb1;
    float* wb2;

    /* generate the sphere points */
    float* sphere_points = (float*) malloc(n_sphere_points*3*sizeof(float));
    generate_sphere_points(sphere_points, n_sphere_points);

    // Calculate total number of grid points
    int total_grid_points = counts[0] * counts[1] * counts[2];

#ifdef _OPENMP
    #pragma omp parallel private(wb1, wb2)
    {
#endif

    /* malloc the work buffers for each thread */
    wb1 = (int*) malloc(n_atoms*sizeof(int));
    wb2 = (float*) malloc(3*n_sphere_points*sizeof(float));

#ifdef _OPENMP
    #pragma omp for
#endif
    for (i = 0; i < n_frames; i++) {
        asa_frame(xyzlist + i*n_atoms*3, n_atoms, atom_radii, sphere_points,
                  n_sphere_points, wb1, wb2, atom_selection_mask,
                  out + i*total_grid_points, counts, grid_spacing);
    }

    free(wb1);
    free(wb2);

#ifdef _OPENMP
    } /* close omp parallel private */
#endif

    free(sphere_points);
}
