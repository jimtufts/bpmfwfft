# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t

cdef extern from "sasa.h":
    void sasa(const int n_frames, const int n_atoms, const float * xyzlist,
              const float * atom_radii, const int n_sphere_points,
              const int * atom_selection_mask, float * out,
              const int * counts, const float grid_spacing)

def calculate_sasa(np.ndarray[np.float32_t, ndim=3, mode="c"] xyzlist not None,
                   np.ndarray[np.float32_t, ndim=1, mode="c"] atom_radii not None,
                   int n_sphere_points,
                   atom_selection_mask,
                   counts,
                   float grid_spacing):
    cdef int n_frames = xyzlist.shape[0]
    cdef int n_atoms = xyzlist.shape[1]

    # Ensure atom_selection_mask is the correct type
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] atom_selection_mask_int = np.asarray(atom_selection_mask,
                                                                                       dtype=np.int32)

    # Ensure counts is the correct type
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] counts_int = np.asarray(counts, dtype=np.int32)

    # Calculate total number of grid points
    cdef int total_grid_points = counts_int[0] * counts_int[1] * counts_int[2]

    # Create output array
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"] out = np.zeros(
        (n_frames, counts_int[0], counts_int[1], counts_int[2]), dtype=np.float32)

    sasa(n_frames, n_atoms, &xyzlist[0, 0, 0],
         &atom_radii[0], n_sphere_points,
         <const int *> &atom_selection_mask_int[0], &out[0, 0, 0, 0],
         <const int *> &counts_int[0], grid_spacing)

    return out

#def calculate_sasa(np.ndarray[float, ndim=3, mode="c"] xyzlist not None,
#                   np.ndarray[float, ndim=1, mode="c"] atom_radii not None,
#                   int n_sphere_points,
#                   np.ndarray[int, ndim=1, mode="c"] atom_mapping not None,
#                   np.ndarray[int, ndim=1, mode="c"] atom_selection_mask not None,
#                   int n_groups):
#    cdef int n_frames = xyzlist.shape[0]
#    cdef int n_atoms = xyzlist.shape[1]
#    
#    # Create output array
#    cdef np.ndarray[float, ndim=3, mode="c"] out = np.zeros((n_frames, n_sphere_points, n_groups), dtype=np.float32)
#    
#    sasa.sasa(n_frames, n_atoms, &xyzlist[0, 0, 0],
#              &atom_radii[0], n_sphere_points,
#              &atom_mapping[0], &atom_selection_mask[0],
#              n_groups, &out[0, 0, 0])
#    
#    return out
