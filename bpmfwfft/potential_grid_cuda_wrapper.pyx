# potential_grid_cuda_wrapper.pyx
# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libc.stdint cimport int64_t

np.import_array()

cdef extern from "potential_grid_cuda_handler.h":
    cdef cppclass PotentialGridCUDAHandler:
        PotentialGridCUDAHandler() except +
        void initialize(size_t natoms, size_t grid_x_size, size_t grid_y_size, size_t grid_z_size) except +
        void cleanup() except +
        void calculatePotentialGrid(
            const string& grid_name,
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
        ) except +


def py_cal_potential_grid_cuda(
        str name,
        np.ndarray[double, ndim=2] crd,
        np.ndarray[double, ndim=1] grid_x,
        np.ndarray[double, ndim=1] grid_y,
        np.ndarray[double, ndim=1] grid_z,
        np.ndarray[double, ndim=1] origin_crd,
        np.ndarray[double, ndim=1] upper_most_corner_crd,
        np.ndarray[int64_t, ndim=1] upper_most_corner,
        np.ndarray[double, ndim=1] spacing,
        np.ndarray[int64_t, ndim=1] grid_counts,
        np.ndarray[double, ndim=1] charges,
        np.ndarray[double, ndim=1] lj_sigma,
        np.ndarray[double, ndim=1] vdw_radii,
        np.ndarray[double, ndim=1] clash_radii,
        np.ndarray[double, ndim=1] molecule_sasa,
        atom_list
):
    """
    GPU-accelerated potential grid calculation using CUDA

    Parameters
    ----------
    name : str
        Grid type ("electrostatic", "LJr", "LJa", "water", "occupancy", "sasa")
    crd : np.ndarray[double, ndim=2]
        Atom coordinates [natoms, 3]
    grid_x, grid_y, grid_z : np.ndarray[double, ndim=1]
        Grid coordinate arrays
    origin_crd : np.ndarray[double, ndim=1]
        Grid origin [3]
    upper_most_corner_crd : np.ndarray[double, ndim=1]
        Upper corner coordinates [3]
    upper_most_corner : np.ndarray[int64_t, ndim=1]
        Upper corner indices [3]
    spacing : np.ndarray[double, ndim=1]
        Grid spacing [3]
    grid_counts : np.ndarray[int64_t, ndim=1]
        Grid dimensions [3]
    charges : np.ndarray[double, ndim=1]
        Atomic charges
    lj_sigma : np.ndarray[double, ndim=1]
        Lennard-Jones sigma parameters
    vdw_radii : np.ndarray[double, ndim=1]
        Van der Waals radii
    clash_radii : np.ndarray[double, ndim=1]
        Clash radii
    molecule_sasa : np.ndarray[double, ndim=1]
        SASA values
    atom_list : list
        List of atom indices (not used in GPU version but kept for API compatibility)

    Returns
    -------
    np.ndarray[double, ndim=3]
        3D potential grid
    """
    cdef size_t natoms = crd.shape[0]
    cdef size_t grid_x_size = grid_x.shape[0]
    cdef size_t grid_y_size = grid_y.shape[0]
    cdef size_t grid_z_size = grid_z.shape[0]

    # Flatten coordinate array to C-contiguous layout
    cdef np.ndarray[double, ndim=1] crd_flat = np.ascontiguousarray(crd.ravel())

    # Allocate output grid
    cdef np.ndarray[double, ndim=3] output_grid = np.zeros(
        (grid_x_size, grid_y_size, grid_z_size),
        dtype=np.float64,
        order='C'
    )

    # Create and initialize handler
    cdef PotentialGridCUDAHandler handler
    cdef string name_cpp = name.encode('utf-8')

    try:
        handler.initialize(natoms, grid_x_size, grid_y_size, grid_z_size)

        # Calculate grid
        handler.calculatePotentialGrid(
            name_cpp,
            &crd_flat[0],
            &grid_x[0],
            &grid_y[0],
            &grid_z[0],
            &charges[0],
            &lj_sigma[0],
            &vdw_radii[0],
            &clash_radii[0],
            &molecule_sasa[0],
            &output_grid[0, 0, 0]
        )

    finally:
        handler.cleanup()

    return output_grid
