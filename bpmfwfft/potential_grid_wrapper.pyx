# potential_grid_wrapper.pyx
# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport int64_t

np.import_array()

cdef extern from "potential_grid.h":
    vector[vector[vector[double]]] cal_potential_grid(
        const string& name,
        const vector[vector[double]]& crd,
        const vector[double]& grid_x,
        const vector[double]& grid_y,
        const vector[double]& grid_z,
        const vector[double]& origin_crd,
        const vector[double]& upper_most_corner_crd,
        const vector[int64_t]& upper_most_corner,
        const vector[double]& spacing,
        const vector[int64_t]& grid_counts,
        const vector[double]& charges,
        const vector[double]& lj_sigma,
        const vector[double]& vdw_radii,
        const vector[double]& clash_radii,
        const vector[double]& molecule_sasa,
        const vector[int64_t]& atom_list
    ) nogil


def py_cal_potential_grid(
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
    Python wrapper for cal_potential_grid C++ function.

    Parameters
    ----------
    name : str
        Grid type name ("electrostatic", "LJr", "LJa", "water", "occupancy", "sasa")
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
        Clash radii for masking
    molecule_sasa : np.ndarray[double, ndim=1]
        SASA values (for water grid)
    atom_list : list or np.ndarray
        List of atom indices to include (for occupancy/sasa grids)

    Returns
    -------
    np.ndarray[double, ndim=3]
        3D potential grid
    """
    # Convert NumPy arrays to C++ vectors
    cdef vector[vector[double]] crd_cpp = crd
    cdef string name_cpp = name.encode('utf-8')
    cdef vector[double] grid_x_cpp = grid_x
    cdef vector[double] grid_y_cpp = grid_y
    cdef vector[double] grid_z_cpp = grid_z
    cdef vector[double] origin_crd_cpp = origin_crd
    cdef vector[double] upper_most_corner_crd_cpp = upper_most_corner_crd
    cdef vector[int64_t] upper_most_corner_cpp = upper_most_corner
    cdef vector[double] spacing_cpp = spacing
    cdef vector[int64_t] grid_counts_cpp = grid_counts
    cdef vector[double] charges_cpp = charges
    cdef vector[double] lj_sigma_cpp = lj_sigma
    cdef vector[double] vdw_radii_cpp = vdw_radii
    cdef vector[double] clash_radii_cpp = clash_radii
    cdef vector[double] molecule_sasa_cpp = molecule_sasa

    # Convert atom_list to vector[int64_t]
    cdef vector[int64_t] atom_list_cpp
    if atom_list is not None:
        for atom_idx in atom_list:
            atom_list_cpp.push_back(atom_idx)

    # Call C++ function
    cdef vector[vector[vector[double]]] result
    with nogil:
        result = cal_potential_grid(
            name_cpp,
            crd_cpp,
            grid_x_cpp, grid_y_cpp, grid_z_cpp,
            origin_crd_cpp, upper_most_corner_crd_cpp,
            upper_most_corner_cpp,
            spacing_cpp, grid_counts_cpp,
            charges_cpp, lj_sigma_cpp, vdw_radii_cpp, clash_radii_cpp,
            molecule_sasa_cpp,
            atom_list_cpp
        )

    # Convert result back to NumPy array
    return np.asarray(result)
