import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "sasa.h":
    void sasa(int n_frames, int n_atoms, const float* xyzlist,
              const float* atom_radii, int n_sphere_points,
              const int* atom_selection_mask, float* out)

def calculate_sasa(np.ndarray[np.float32_t, ndim=3, mode="c"] xyzlist not None,
                   np.ndarray[np.float32_t, ndim=1, mode="c"] atom_radii not None,
                   int n_sphere_points,
                   np.ndarray[np.int32_t, ndim=1, mode="c"] atom_selection_mask not None):
    cdef int n_frames = xyzlist.shape[0]
    cdef int n_atoms = xyzlist.shape[1]
    
    # Create output array
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"] out = np.zeros((n_frames, n_atoms, n_sphere_points, 4), dtype=np.float32)

    cdef np.ndarray[int, ndim=1, mode="c"] atom_selection_mask_int = atom_selection_mask.astype(np.int32)
    
    sasa(n_frames, n_atoms, &xyzlist[0, 0, 0],
         &atom_radii[0], n_sphere_points,
         <const int*>&atom_selection_mask_int[0], &out[0, 0, 0, 0])
    
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
