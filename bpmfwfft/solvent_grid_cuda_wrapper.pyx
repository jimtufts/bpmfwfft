# solvent_grid_cuda_wrapper.pyx
# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

np.import_array()

cdef extern from "solvent_grid_cuda_handler.h":
    cppclass SolventGridCudaHandler:
        SolventGridCudaHandler()
        vector[vector[vector[float]]] compute_solvent_grid(
            const vector[vector[float]]& atom_coords,
            const vector[float]& vdw_radii,
            const vector[float]& grid_x,
            const vector[float]& grid_y,
            const vector[float]& grid_z
        )

def py_cal_solvent_grid_cuda(
    np.ndarray[np.float32_t, ndim=2] crd,      # [n_atoms, 3]
    np.ndarray[np.float32_t, ndim=1] vdw_radii, # [n_atoms]
    np.ndarray[np.float64_t, ndim=1] grid_x,    # [nx]
    np.ndarray[np.float64_t, ndim=1] grid_y,    # [ny]
    np.ndarray[np.float64_t, ndim=1] grid_z     # [nz]
):
    """
    Compute solvent (occupancy) grid on GPU

    Parameters
    ----------
    crd : np.ndarray, shape (n_atoms, 3)
        Atom coordinates
    vdw_radii : np.ndarray, shape (n_atoms,)
        VDW radii for each atom
    grid_x : np.ndarray, shape (nx,)
        X-axis grid coordinates
    grid_y : np.ndarray, shape (ny,)
        Y-axis grid coordinates
    grid_z : np.ndarray, shape (nz,)
        Z-axis grid coordinates

    Returns
    -------
    grid : np.ndarray, shape (nz, ny, nx)
        Solvent grid with 1.0 where atoms occupy space, 0.0 otherwise
    """
    cdef int n_atoms = crd.shape[0]
    cdef int i, j

    # Convert coordinates to C++ vector
    cdef vector[vector[float]] crd_cpp
    for i in range(n_atoms):
        crd_cpp.push_back(vector[float]())
        for j in range(3):
            crd_cpp[i].push_back(crd[i, j])

    # Convert VDW radii to C++ vector
    cdef vector[float] vdw_radii_cpp
    for i in range(n_atoms):
        vdw_radii_cpp.push_back(vdw_radii[i])

    # Convert grid coordinates to C++ vectors (cast double to float)
    cdef vector[float] grid_x_cpp
    cdef vector[float] grid_y_cpp
    cdef vector[float] grid_z_cpp

    for i in range(grid_x.shape[0]):
        grid_x_cpp.push_back(<float>grid_x[i])
    for i in range(grid_y.shape[0]):
        grid_y_cpp.push_back(<float>grid_y[i])
    for i in range(grid_z.shape[0]):
        grid_z_cpp.push_back(<float>grid_z[i])

    # Create handler and compute grid
    cdef SolventGridCudaHandler handler
    cdef vector[vector[vector[float]]] result

    result = handler.compute_solvent_grid(
        crd_cpp, vdw_radii_cpp,
        grid_x_cpp, grid_y_cpp, grid_z_cpp
    )

    # Convert result to numpy array
    return np.asarray(result, dtype=np.float32)
