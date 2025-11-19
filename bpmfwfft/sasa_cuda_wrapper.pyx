# sasa_cuda_wrapper.pyx
# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "sasa_cuda_handler.h":
    cdef cppclass SASACUDAHandler:
        SASACUDAHandler() except +
        void initialize(size_t n_atoms, size_t n_sphere_points,
                       int grid_count_x, int grid_count_y, int grid_count_z) except +
        void cleanup() except +
        void calculateSASAGrid(const float* atom_coords,
                              const float* atom_radii,
                              const float* sphere_points,
                              const int* atom_selection_mask,
                              float* output_grid,
                              float grid_spacing,
                              bint use_ten_corners) except +


def calculate_sasa_cuda(
    np.ndarray[np.float32_t, ndim=3, mode="c"] xyzlist not None,
    np.ndarray[np.float32_t, ndim=1, mode="c"] atom_radii not None,
    int n_sphere_points,
    atom_selection_mask,
    counts,
    float grid_spacing,
    bint use_ten_corners=True
):
    """
    GPU-accelerated SASA grid calculation using CUDA

    Parameters
    ----------
    xyzlist : np.ndarray[float32, ndim=3]
        Atom coordinates [n_frames, n_atoms, 3]
    atom_radii : np.ndarray[float32, ndim=1]
        Atom radii (VDW + probe) [n_atoms]
    n_sphere_points : int
        Number of sphere points for SASA calculation
    atom_selection_mask : array-like
        Atom selection mask [n_atoms]
    counts : array-like
        Grid dimensions [3]
    grid_spacing : float
        Grid spacing
    use_ten_corners : bool, optional
        If True, use 10-corner interpolation method (default)
        If False, use simple rounding method

    Returns
    -------
    np.ndarray[float32, ndim=4]
        SASA grid [n_frames, counts[0], counts[1], counts[2]]
    """
    cdef int n_frames = xyzlist.shape[0]
    cdef int n_atoms = xyzlist.shape[1]
    cdef int frame_idx
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] atom_selection_mask_int
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] counts_int
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] sphere_points
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"] out
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] coords_flat
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] sphere_points_flat
    cdef SASACUDAHandler handler

    # Ensure atom_selection_mask is the correct type
    atom_selection_mask_int = np.asarray(atom_selection_mask, dtype=np.int32)

    # Ensure counts is the correct type
    counts_int = np.asarray(counts, dtype=np.int32)

    # Generate sphere points (uniformly distributed on unit sphere)
    sphere_points = generate_sphere_points(n_sphere_points)

    # Create output array
    out = np.zeros((n_frames, counts_int[0], counts_int[1], counts_int[2]), dtype=np.float32)

    try:
        handler.initialize(
            n_atoms,
            n_sphere_points,
            counts_int[0],
            counts_int[1],
            counts_int[2]
        )

        # Process each frame
        for frame_idx in range(n_frames):
            # Flatten coordinates for this frame
            coords_flat = np.ascontiguousarray(xyzlist[frame_idx].ravel())

            # Flatten sphere points
            sphere_points_flat = np.ascontiguousarray(sphere_points.ravel())

            # Calculate SASA grid for this frame
            handler.calculateSASAGrid(
                &coords_flat[0],
                &atom_radii[0],
                &sphere_points_flat[0],
                <const int*>&atom_selection_mask_int[0],
                &out[frame_idx, 0, 0, 0],
                grid_spacing,
                use_ten_corners
            )

    finally:
        handler.cleanup()

    return out


def generate_sphere_points(int n_points):
    """
    Generate uniformly distributed points on a unit sphere using Fibonacci spiral

    Parameters
    ----------
    n_points : int
        Number of points to generate

    Returns
    -------
    np.ndarray[float32, ndim=2]
        Sphere points [n_points, 3]
    """
    cdef np.ndarray[np.float32_t, ndim=2] points = np.zeros((n_points, 3), dtype=np.float32)
    cdef float phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    cdef int i
    cdef float y, radius, theta, x, z

    for i in range(n_points):
        y = 1.0 - (i / float(n_points - 1)) * 2.0  # y goes from 1 to -1
        radius = np.sqrt(1.0 - y * y)  # radius at y

        theta = 2.0 * np.pi * i / phi  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = z

    return points
