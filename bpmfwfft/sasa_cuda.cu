#include "sasa_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Device structure for grid indices
struct GridIndex {
    int x, y, z;
};

// Device function to solve 10x10 linear system using Gaussian elimination
// This is a simple direct solver suitable for small systems on GPU
__device__ bool solve_10x10_system(const float a[10][10], const float b[10], float x[10]) {
    float aug[10][11];  // Augmented matrix

    // Copy to augmented matrix
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            aug[i][j] = a[i][j];
        }
        aug[i][10] = b[i];
    }

    // Forward elimination with partial pivoting
    for (int k = 0; k < 10; k++) {
        // Find pivot
        int max_row = k;
        float max_val = fabsf(aug[k][k]);
        for (int i = k + 1; i < 10; i++) {
            float val = fabsf(aug[i][k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singular matrix
        if (max_val < 1e-10f) {
            return false;
        }

        // Swap rows if needed
        if (max_row != k) {
            for (int j = 0; j <= 10; j++) {
                float tmp = aug[k][j];
                aug[k][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }

        // Eliminate column
        for (int i = k + 1; i < 10; i++) {
            float factor = aug[i][k] / aug[k][k];
            for (int j = k; j <= 10; j++) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // Back substitution
    for (int i = 9; i >= 0; i--) {
        float sum = aug[i][10];
        for (int j = i + 1; j < 10; j++) {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    return true;
}

// Device function to get 10 corners around a point
__device__ int get_ten_corners(
    float px, float py, float pz,
    float grid_spacing,
    int grid_count_x, int grid_count_y, int grid_count_z,
    GridIndex corners[10]
) {
    // Find the lower corner of the containing cube
    int lx = (int)(px / grid_spacing);
    int ly = (int)(py / grid_spacing);
    int lz = (int)(pz / grid_spacing);

    // Check if point is in grid
    if (px < 0 || py < 0 || pz < 0 ||
        px >= (grid_count_x - 1) * grid_spacing ||
        py >= (grid_count_y - 1) * grid_spacing ||
        pz >= (grid_count_z - 1) * grid_spacing) {
        return 0;
    }

    // Generate 8 corners of the containing cube
    GridIndex eight_corners[8];
    float distances[8];

    const int shifts[8][3] = {
        {0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
        {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}
    };

    for (int i = 0; i < 8; i++) {
        eight_corners[i].x = lx + shifts[i][0];
        eight_corners[i].y = ly + shifts[i][1];
        eight_corners[i].z = lz + shifts[i][2];

        float cx = eight_corners[i].x * grid_spacing;
        float cy = eight_corners[i].y * grid_spacing;
        float cz = eight_corners[i].z * grid_spacing;

        float dx = cx - px;
        float dy = cy - py;
        float dz = cz - pz;
        distances[i] = sqrtf(dx*dx + dy*dy + dz*dz);
    }

    // Find nearest and furthest corners
    int nearest_idx = 0;
    int furthest_idx = 0;
    float min_dist = distances[0];
    float max_dist = distances[0];

    for (int i = 1; i < 8; i++) {
        if (distances[i] < min_dist) {
            min_dist = distances[i];
            nearest_idx = i;
        }
        if (distances[i] > max_dist) {
            max_dist = distances[i];
            furthest_idx = i;
        }
    }

    // Check if nearest corner is on boundary
    GridIndex& nearest = eight_corners[nearest_idx];
    if (nearest.x == 0 || nearest.x >= grid_count_x - 1 ||
        nearest.y == 0 || nearest.y >= grid_count_y - 1 ||
        nearest.z == 0 || nearest.z >= grid_count_z - 1) {
        return 0;
    }

    // Find 3 additional corners from the 6-neighbor offsets
    const int six_shifts[6][3] = {
        {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
    };

    GridIndex three_corners[3];
    int found_count = 0;

    for (int i = 0; i < 6 && found_count < 3; i++) {
        GridIndex candidate;
        candidate.x = nearest.x + six_shifts[i][0];
        candidate.y = nearest.y + six_shifts[i][1];
        candidate.z = nearest.z + six_shifts[i][2];

        // Check if this corner is not in eight_corners
        bool is_in_eight = false;
        for (int j = 0; j < 8; j++) {
            if (candidate.x == eight_corners[j].x &&
                candidate.y == eight_corners[j].y &&
                candidate.z == eight_corners[j].z) {
                is_in_eight = true;
                break;
            }
        }

        if (!is_in_eight) {
            three_corners[found_count++] = candidate;
        }
    }

    if (found_count < 3) {
        return 0;
    }

    // Combine: 7 from eight_corners (excluding furthest) + 3 new corners
    int corner_idx = 0;
    for (int i = 0; i < 8; i++) {
        if (i != furthest_idx) {
            corners[corner_idx++] = eight_corners[i];
        }
    }
    for (int i = 0; i < 3; i++) {
        corners[corner_idx++] = three_corners[i];
    }

    return 10;
}

/**
 * CUDA kernel to compute SASA grid
 * Each thread processes one atom
 */
__global__ void compute_sasa_kernel(
    const float* __restrict__ atom_coords,
    const float* __restrict__ atom_radii,
    const float* __restrict__ sphere_points,
    const int* __restrict__ atom_selection_mask,
    float* __restrict__ output_grid,
    std::size_t n_atoms,
    std::size_t n_sphere_points,
    int grid_count_x,
    int grid_count_y,
    int grid_count_z,
    float grid_spacing,
    bool use_ten_corners
) {
    int atom_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (atom_i >= n_atoms) return;
    if (atom_selection_mask[atom_i] == 0) return;

    float constant = 4.0f * M_PI / n_sphere_points;
    float radius_i = atom_radii[atom_i];
    float ax = atom_coords[atom_i * 3 + 0];
    float ay = atom_coords[atom_i * 3 + 1];
    float az = atom_coords[atom_i * 3 + 2];

    // Find neighbors (atoms within radius_i + radius_j)
    int neighbor_indices[512];  // Max 512 neighbors per atom
    int n_neighbors = 0;

    for (int j = 0; j < n_atoms; j++) {
        if (j == atom_i) continue;

        float bx = atom_coords[j * 3 + 0];
        float by = atom_coords[j * 3 + 1];
        float bz = atom_coords[j * 3 + 2];

        float dx = ax - bx;
        float dy = ay - by;
        float dz = az - bz;
        float dist_sq = dx*dx + dy*dy + dz*dz;

        float radius_j = atom_radii[j];
        float cutoff = radius_i + radius_j;

        if (dist_sq < cutoff * cutoff) {
            if (n_neighbors < 512) {
                neighbor_indices[n_neighbors++] = j;
            }
        }
    }

    // Check each sphere point for accessibility
    for (int sp_idx = 0; sp_idx < n_sphere_points; sp_idx++) {
        // Center sphere point on atom_i
        float spx = ax + radius_i * sphere_points[sp_idx * 3 + 0];
        float spy = ay + radius_i * sphere_points[sp_idx * 3 + 1];
        float spz = az + radius_i * sphere_points[sp_idx * 3 + 2];

        // Check if this point is accessible (not inside any neighbor)
        bool is_accessible = true;
        for (int n = 0; n < n_neighbors; n++) {
            int neighbor_idx = neighbor_indices[n];
            float nbx = atom_coords[neighbor_idx * 3 + 0];
            float nby = atom_coords[neighbor_idx * 3 + 1];
            float nbz = atom_coords[neighbor_idx * 3 + 2];

            float dx = spx - nbx;
            float dy = spy - nby;
            float dz = spz - nbz;
            float dist_sq = dx*dx + dy*dy + dz*dz;

            float nb_radius = atom_radii[neighbor_idx];
            if (dist_sq < nb_radius * nb_radius) {
                is_accessible = false;
                break;
            }
        }

        if (is_accessible) {
            float value = constant * radius_i * radius_i;

            if (use_ten_corners) {
                // Use 10-corner interpolation method
                GridIndex ten_corners[10];
                int n_corners = get_ten_corners(
                    spx, spy, spz, grid_spacing,
                    grid_count_x, grid_count_y, grid_count_z,
                    ten_corners
                );

                if (n_corners == 10) {
                    // Set up and solve the 10x10 system
                    float a_matrix[10][10];
                    float b_vector[10];

                    // Initialize
                    for (int i = 0; i < 10; i++) {
                        b_vector[i] = 0.0f;
                        for (int j = 0; j < 10; j++) {
                            a_matrix[i][j] = 0.0f;
                        }
                    }

                    b_vector[0] = value;
                    for (int j = 0; j < 10; j++) {
                        a_matrix[0][j] = 1.0f;
                    }

                    // Calculate delta vectors and fill matrix
                    float delta_x[10], delta_y[10], delta_z[10];
                    for (int j = 0; j < 10; j++) {
                        float cx = ten_corners[j].x * grid_spacing;
                        float cy = ten_corners[j].y * grid_spacing;
                        float cz = ten_corners[j].z * grid_spacing;

                        delta_x[j] = cx - spx;
                        delta_y[j] = cy - spy;
                        delta_z[j] = cz - spz;

                        a_matrix[1][j] = delta_x[j];
                        a_matrix[2][j] = delta_y[j];
                        a_matrix[3][j] = delta_z[j];
                    }

                    // Fill second-order terms
                    int row = 3;
                    for (int i = 0; i < 3; i++) {
                        for (int j = i; j < 3; j++) {
                            row++;
                            for (int k = 0; k < 10; k++) {
                                float di = (i == 0) ? delta_x[k] : ((i == 1) ? delta_y[k] : delta_z[k]);
                                float dj = (j == 0) ? delta_x[k] : ((j == 1) ? delta_y[k] : delta_z[k]);
                                a_matrix[row][k] = di * dj;
                            }
                        }
                    }

                    // Solve the system
                    float sasa_values[10];
                    if (solve_10x10_system(a_matrix, b_vector, sasa_values)) {
                        // Distribute values to grid (with atomic adds for thread safety)
                        for (int j = 0; j < 10; j++) {
                            int gx = ten_corners[j].x;
                            int gy = ten_corners[j].y;
                            int gz = ten_corners[j].z;

                            if (gx >= 0 && gx < grid_count_x &&
                                gy >= 0 && gy < grid_count_y &&
                                gz >= 0 && gz < grid_count_z) {
                                int grid_idx = gx * grid_count_y * grid_count_z +
                                             gy * grid_count_z + gz;
                                atomicAdd(&output_grid[grid_idx], sasa_values[j]);
                            }
                        }
                    } else {
                        // Fallback to simple rounding if solver fails
                        int ix = (int)roundf(spx / grid_spacing);
                        int iy = (int)roundf(spy / grid_spacing);
                        int iz = (int)roundf(spz / grid_spacing);

                        if (ix >= 0 && ix < grid_count_x &&
                            iy >= 0 && iy < grid_count_y &&
                            iz >= 0 && iz < grid_count_z) {
                            int grid_idx = ix * grid_count_y * grid_count_z +
                                         iy * grid_count_z + iz;
                            atomicAdd(&output_grid[grid_idx], value);
                        }
                    }
                } else {
                    // Fallback to simple rounding
                    int ix = (int)roundf(spx / grid_spacing);
                    int iy = (int)roundf(spy / grid_spacing);
                    int iz = (int)roundf(spz / grid_spacing);

                    if (ix >= 0 && ix < grid_count_x &&
                        iy >= 0 && iy < grid_count_y &&
                        iz >= 0 && iz < grid_count_z) {
                        int grid_idx = ix * grid_count_y * grid_count_z +
                                     iy * grid_count_z + iz;
                        atomicAdd(&output_grid[grid_idx], value);
                    }
                }
            } else {
                // Simple rounding method
                int ix = (int)roundf(spx / grid_spacing);
                int iy = (int)roundf(spy / grid_spacing);
                int iz = (int)roundf(spz / grid_spacing);

                if (ix >= 0 && ix < grid_count_x &&
                    iy >= 0 && iy < grid_count_y &&
                    iz >= 0 && iz < grid_count_z) {
                    int grid_idx = ix * grid_count_y * grid_count_z +
                                 iy * grid_count_z + iz;
                    atomicAdd(&output_grid[grid_idx], value);
                }
            }
        }
    }
}

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
) {
    // Each block processes multiple atoms
    int threads_per_block = 128;
    int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

    // Initialize output grid to zero
    int grid_size = grid_count_x * grid_count_y * grid_count_z;
    CHECK_CUDA(cudaMemset(d_output_grid, 0, grid_size * sizeof(float)));

    // Launch kernel
    compute_sasa_kernel<<<num_blocks, threads_per_block>>>(
        d_atom_coords,
        d_atom_radii,
        d_sphere_points,
        d_atom_selection_mask,
        d_output_grid,
        n_atoms,
        n_sphere_points,
        grid_count_x,
        grid_count_y,
        grid_count_z,
        grid_spacing,
        use_ten_corners
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
