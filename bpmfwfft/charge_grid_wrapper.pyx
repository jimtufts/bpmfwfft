# charge_grid_wrapper.pyx
# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport int64_t
from libc.stdio cimport printf
#from libcpp.array cimport array

np.import_array()

cdef extern from "charge_grid.h":
    vector[vector[vector[double]]] cal_solvent_grid(
        const vector[vector[double]]& crd,
        const vector[double]& grid_x,
        const vector[double]& grid_y,
        const vector[double]& grid_z,
        const vector[double]& origin_crd,
        const vector[double]& upper_most_corner_crd,
        const vector[int64_t]& grid_counts,
        const vector[double]& vdw_radii
    )

def py_cal_solvent_grid(np.ndarray[np.float64_t, ndim=3] crd,
                        np.ndarray[np.float64_t, ndim=1] grid_x,
                        np.ndarray[np.float64_t, ndim=1] grid_y,
                        np.ndarray[np.float64_t, ndim=1] grid_z,
                        np.ndarray[np.float64_t, ndim=1] origin_crd,
                        np.ndarray[np.float64_t, ndim=1] upper_most_corner_crd,
                        np.ndarray[np.int64_t, ndim=1] grid_counts,
                        np.ndarray[np.float64_t, ndim=1] vdw_radii):
    printf("Starting py_cal_solvent_grid\n")
    printf("crd shape: (%d, %d, %d)\n", crd.shape[0], crd.shape[1], crd.shape[2])
    printf("grid_x shape: (%d)\n", grid_x.shape[0])
    cdef vector[vector[double]] crd_cpp
    cdef int i, j
    for i in range(crd.shape[1]):  # Iterate over atoms
        crd_cpp.push_back(vector[double]())
        for j in range(crd.shape[2]):  # Iterate over coordinates (x, y, z)
            crd_cpp[i].push_back(crd[0, i, j])
    printf("Converted crd to C++ vector\n")
    printf("Calling cal_solvent_grid\n")
    
    # Ensure grid_counts is treated as vector[int64_t]
    cdef vector[int64_t] grid_counts_cpp = grid_counts

    result = cal_solvent_grid(crd_cpp,
                              grid_x,
                              grid_y,
                              grid_z,
                              origin_crd,
                              upper_most_corner_crd,
                              grid_counts_cpp,
                              vdw_radii)
    
    printf("cal_solvent_grid completed\n")
    return np.array(result)

cdef extern from "charge_grid.h":
    bint is_row_in_matrix(const vector[int64_t]& row, const vector[vector[int64_t]]& matrix) nogil

    vector[vector[vector[double]]] cal_charge_grid(
            const vector[vector[double]]& crd,
            const vector[double]& charges,
            const string& name,
            const vector[double]& grid_x,
            const vector[double]& grid_y,
            const vector[double]& grid_z,
            const vector[double]& origin_crd,
            const vector[double]& upper_most_corner_crd,
            const vector[int64_t]& upper_most_corner,
            const vector[double]& spacing,
            const vector[vector[int64_t]]& eight_corner_shifts,
            const vector[vector[int64_t]]& six_corner_shifts
    ) nogil

cdef extern from "charge_grid.h":
    vector[double] get_corner_crd(const vector[int64_t] &, const vector[double] &, const vector[double] &,
                                  const vector[double] &)

def py_cal_charge_grid(
        np.ndarray[double, ndim=2] crd,
        np.ndarray[double, ndim=1] charges,
        str name,
        np.ndarray[double, ndim=1] grid_x,
        np.ndarray[double, ndim=1] grid_y,
        np.ndarray[double, ndim=1] grid_z,
        np.ndarray[double, ndim=1] origin_crd,
        np.ndarray[double, ndim=1] upper_most_corner_crd,
        np.ndarray[int64_t, ndim=1] upper_most_corner,
        np.ndarray[double, ndim=1] spacing,
        np.ndarray[int64_t, ndim=2] eight_corner_shifts,
        np.ndarray[int64_t, ndim=2] six_corner_shifts
):
    cdef vector[vector[double]] crd_cpp = crd
    cdef vector[double] charges_cpp = charges
    cdef string name_cpp = name.encode('utf-8')
    cdef vector[double] grid_x_cpp = grid_x
    cdef vector[double] grid_y_cpp = grid_y
    cdef vector[double] grid_z_cpp = grid_z
    cdef vector[double] origin_crd_cpp = origin_crd
    cdef vector[double] upper_most_corner_crd_cpp = upper_most_corner_crd
    cdef vector[int64_t] upper_most_corner_cpp = upper_most_corner
    cdef vector[double] spacing_cpp = spacing

    # Convert eight_corner_shifts to vector[vector[int64_t]]
    cdef vector[vector[int64_t]] eight_corner_shifts_cpp
    cdef vector[int64_t] temp_vector
    for i in range(eight_corner_shifts.shape[0]):
        temp_vector.clear()
        temp_vector.push_back(eight_corner_shifts[i, 0])
        temp_vector.push_back(eight_corner_shifts[i, 1])
        temp_vector.push_back(eight_corner_shifts[i, 2])
        eight_corner_shifts_cpp.push_back(temp_vector)

    # Convert six_corner_shifts to vector[vector[int64_t]]
    cdef vector[vector[int64_t]] six_corner_shifts_cpp
    for i in range(six_corner_shifts.shape[0]):
        temp_vector.clear()
        temp_vector.push_back(six_corner_shifts[i, 0])
        temp_vector.push_back(six_corner_shifts[i, 1])
        temp_vector.push_back(six_corner_shifts[i, 2])
        six_corner_shifts_cpp.push_back(temp_vector)

    cdef vector[vector[vector[double]]] result
    with nogil:
        result = cal_charge_grid(
            crd_cpp, charges_cpp, name_cpp,
            grid_x_cpp, grid_y_cpp, grid_z_cpp,
            origin_crd_cpp, upper_most_corner_crd_cpp, upper_most_corner_cpp,
            spacing_cpp, eight_corner_shifts_cpp, six_corner_shifts_cpp
        )

    return np.asarray(result)
