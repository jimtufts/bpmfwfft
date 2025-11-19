# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t
from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "charge_grid_cuda.h":
    void cuda_cal_charge_grid(
        const double* d_atom_coordinates,
        const double* d_charges,
        const double* d_grid_x,
        const double* d_grid_y,
        const double* d_grid_z,
        const double* d_origin_crd,
        const double* d_upper_most_corner_crd,
        const int64_t* d_upper_most_corner,
        const double* d_spacing,
        const int64_t* d_eight_corner_shifts,
        const int64_t* d_six_corner_shifts,
        int num_atoms,
        int i_max, int j_max, int k_max,
        double* d_grid,
        bool use_nnls_solver
    ) except +

# Declare the C++ class
cdef extern from "charge_grid_cuda_handler.h":
    cdef cppclass ChargeGridCUDAHandler:
        ChargeGridCUDAHandler() except +
        void initialize(size_t num_atoms, size_t grid_size) except +
        void cleanup() except +
        void calculateChargeGrid(
            const double* atom_coordinates,
            const double* charges,
            const double* grid_x,
            const double* grid_y,
            const double* grid_z,
            const double* origin_crd,
            const double* upper_most_corner_crd,
            const int64_t* upper_most_corner,
            const double* spacing,
            const int64_t* eight_corner_shifts,
            const int64_t* six_corner_shifts,
            double* output_grid,
            bool use_nnls_solver) except +

def py_cal_charge_grid_cuda(
    np.ndarray[double, ndim=2, mode="c"] atom_coordinates not None,
    np.ndarray[double, ndim=1, mode="c"] charges not None,
    str charge_name,
    np.ndarray[double, ndim=1, mode="c"] grid_x not None,
    np.ndarray[double, ndim=1, mode="c"] grid_y not None,
    np.ndarray[double, ndim=1, mode="c"] grid_z not None,
    np.ndarray[double, ndim=1, mode="c"] origin_crd not None,
    np.ndarray[double, ndim=1, mode="c"] upper_most_corner_crd not None,
    np.ndarray[int64_t, ndim=1, mode="c"] upper_most_corner not None,
    np.ndarray[double, ndim=1, mode="c"] spacing not None,
    np.ndarray[int64_t, ndim=2, mode="c"] eight_corner_shifts not None,
    np.ndarray[int64_t, ndim=2, mode="c"] six_corner_shifts not None
):
    cdef int num_atoms = charges.shape[0]
    cdef int grid_size = grid_x.shape[0]

    # Determine whether to use NNLS solver based on charge type
    # LJr (Lennard-Jones repulsive) and LJa (Lennard-Jones attractive) require non-negative charges
    cdef bool use_nnls = (charge_name == "LJr" or charge_name == "LJa")

    cdef np.ndarray[double, ndim=3, mode="c"] output_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

    cdef ChargeGridCUDAHandler handler
    try:
        handler.initialize(num_atoms, grid_size)
        handler.calculateChargeGrid(
            &atom_coordinates[0, 0],
            &charges[0],
            &grid_x[0],
            &grid_y[0],
            &grid_z[0],
            &origin_crd[0],
            &upper_most_corner_crd[0],
            &upper_most_corner[0],
            &spacing[0],
            &eight_corner_shifts[0, 0],
            &six_corner_shifts[0, 0],
            &output_grid[0, 0, 0],
            use_nnls
        )
    except RuntimeError as e:
        raise RuntimeError(f"CUDA error in calculateChargeGrid: {str(e)}")
    finally:
        handler.cleanup()

    return output_grid
