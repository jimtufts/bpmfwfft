
cimport cython
import numpy as np
cimport numpy as np
import math
from cython cimport view
from scipy.optimize import nnls


cdef extern from "math.h":
    double sqrt(double)
    double cos(double)
    double sin(double)
    double fmod(double x, double y)
    double M_PI

@cython.boundscheck(False)
def c_greater_or_equal(np.ndarray[np.int64_t, ndim=1] corner, long x):
    cdef int lmax = corner.shape[0]
    for i in range(lmax):
        if corner[i] < x:
            return False
    return True

@cython.boundscheck(False)
def c_less_or_equal(np.ndarray[np.int64_t, ndim=1] corner1, np.ndarray[np.int64_t, ndim=1] corner2):
    cdef int lmax = corner1.shape[0]
    cdef int i
    for i in range(lmax):
        if corner1[i] > corner2[i]:
            return False
    return True

@cython.boundscheck(False)
def cdot(double[:] x, double[:] y):
    cdef int i, lmax
    cdef double dot, tmp
    lmax = x.shape[0]
    dot = 0.
    for i in range(lmax):
        tmp = x[i]*y[i]
        dot += tmp
    return dot

@cython.boundscheck(False)
def cdistance(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    cdef int i, lmax
    cdef double d, tmp
    lmax = x.shape[0]
    d = 0.
    for i in range(lmax):
        tmp = x[i] - y[i]
        d += tmp*tmp
    return sqrt(d)

@cython.boundscheck(False)
def cmround(np.ndarray[np.float64_t, ndim=1] crd, np.ndarray[np.float64_t, ndim=1] m):
    cdef:
        int lmax = crd.shape[0]
        int i
        np.ndarray[np.float64_t, ndim = 1] aligned_crd = np.zeros([lmax], dtype=np.float64)

    for i in range(lmax):
        aligned_crd[i] = m[i] * round(crd[i]/m[i])
    return aligned_crd

@cython.boundscheck(False)
def c_list_to_array(list value_list):
    cdef:
        int lmax = len(value_list)
        int i,j
        value_array = view.array(shape=(lmax,3), itemsize=sizeof(int), format="i")
    for i in range(lmax):
        for j in range(3):
            value_array[i,j] = value_list[i][j]
    return value_array

@cython.boundscheck(False)
def c_list_to_array_long(list value_list):
    cdef:
        int lmax = len(value_list)
        int i,j
        value_array = view.array(shape=(lmax,3), itemsize=sizeof(long), format="l")
    for i in range(lmax):
        for j in range(3):
            value_array[i,j] = value_list[i][j]
    return value_array

@cython.boundscheck(False)
def c_crd_to_grid(np.ndarray[np.float64_t, ndim=1] crd, np.ndarray[np.float64_t, ndim=1] spacing):
    cdef:
        int lmax = crd.shape[0]
        int i
        np.ndarray[np.float64_t, ndim = 1] grid_crd = crd/spacing
    return grid_crd.astype(int)

@cython.boundscheck(False)
def c_points_to_grid(np.ndarray[np.float64_t, ndim=3] points,
                     np.ndarray[np.float64_t, ndim=1] spacing,
                     np.ndarray[np.int64_t, ndim=1] counts):
    cdef:
        int i, j
        Py_ssize_t x, y, z
        int lmax = points[0, 0, :3].shape[0]
        int natoms = points.shape[0]
        int npoints = points.shape[1]
        np.ndarray[np.float64_t, ndim = 1] aligned_crd = np.empty([lmax], dtype=np.float64)
        np.ndarray[np.float64_t, ndim = 3] grid = np.zeros(counts, dtype=np.float64)
    for i in range(natoms):
        for j in range(npoints):
            aligned_crd = cmround(points[i, j, :3], spacing)
            x, y, z = c_crd_to_grid(aligned_crd, spacing)
            grid[x, y, z] += 4 * np.pi * (points[i,j,3] ** 2) / npoints
    return grid

@cython.boundscheck(False)
def c_get_corner_crd(np.ndarray[np.int64_t, ndim=1] corner,
                     np.ndarray[np.float64_t, ndim=1] grid_x, 
                     np.ndarray[np.float64_t, ndim=1] grid_y,
                     np.ndarray[np.float64_t, ndim=1] grid_z):
    cdef:
        int i, j, k
        np.ndarray[np.float64_t, ndim=1] crd

    i, j, k = corner
    crd = np.array([grid_x[i], grid_y[j], grid_z[k]] , dtype=float)
    return crd


@cython.boundscheck(False)
def c_is_in_grid(np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                 np.ndarray[np.float64_t, ndim=1] origin_crd,
                 np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd):
    
    cdef int i, lmax
    lmax = atom_coordinate.shape[0]
    for i in range(lmax):
        if (atom_coordinate[i] < origin_crd[i]) or (atom_coordinate[i] >= uper_most_corner_crd[i]):
            return False
    return True


@cython.boundscheck(False)
def c_containing_cube(  np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                        np.ndarray[np.float64_t, ndim=1] origin_crd,
                        np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                        np.ndarray[np.float64_t, ndim=1] spacing,
                        np.ndarray[np.int64_t, ndim=2]   eight_corner_shifts,
                        np.ndarray[np.float64_t, ndim=1] grid_x,
                        np.ndarray[np.float64_t, ndim=1] grid_y,
                        np.ndarray[np.float64_t, ndim=1] grid_z ):

    cdef:
        np.ndarray[np.float64_t, ndim=1] tmp
        np.ndarray[np.float64_t, ndim=1] corner_crd

        np.ndarray[np.int64_t, ndim=1]   lower_corner
        np.ndarray[np.int64_t, ndim=1]   corner

        list eight_corners, distances
        int nearest_ind, furthest_ind

    if not c_is_in_grid(atom_coordinate, origin_crd, uper_most_corner_crd):
        return [], 0, 0
    
    tmp = atom_coordinate - origin_crd
    lower_corner = np.array(tmp / spacing, dtype=int)
    eight_corners = [lower_corner + shift for shift in eight_corner_shifts]

    distances = []
    for corner in eight_corners:
        corner_crd = c_get_corner_crd(corner, grid_x, grid_y, grid_z)
        distances.append(cdistance(corner_crd, atom_coordinate))

    nearest_ind   = distances.index(min(distances))
    furthest_ind  = distances.index(max(distances))
    return eight_corners, nearest_ind, furthest_ind


@cython.boundscheck(False)
def c_lower_corner_of_containing_cube(  np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                                        np.ndarray[np.float64_t, ndim=1] origin_crd,
                                        np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                                        np.ndarray[np.float64_t, ndim=1] spacing):
    cdef:
        np.ndarray[np.float64_t, ndim=1] tmp
        np.ndarray[np.int64_t, ndim=1]   lower_corner

    if not c_is_in_grid(atom_coordinate, origin_crd, uper_most_corner_crd):
        return np.array([], dtype=int)

    tmp = atom_coordinate - origin_crd
    lower_corner = np.array(tmp / spacing, dtype=int)
    return lower_corner


@cython.boundscheck(False)
def c_corners_within_radius(np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                            double radius,
                            np.ndarray[np.float64_t, ndim=1] origin_crd,
                            np.ndarray[np.float64_t, ndim=1] upper_most_corner_crd,
                            np.ndarray[np.int64_t, ndim=1]   upper_most_corner,
                            np.ndarray[np.float64_t, ndim=1] spacing,
                            np.ndarray[np.float64_t, ndim=1] grid_x,
                            np.ndarray[np.float64_t, ndim=1] grid_y,
                            np.ndarray[np.float64_t, ndim=1] grid_z,
                            np.ndarray[np.int64_t, ndim=1]   grid_counts):
    cdef:
        list corners
        Py_ssize_t count_i, count_j, count_k
        Py_ssize_t i, j, k, l
        int lmax = atom_coordinate.shape[0]
        float r, R

        np.ndarray[np.int64_t, ndim=1] lower_corner = np.zeros([lmax], dtype=np.int64)
        long[:] lower_corner_view = lower_corner
        np.ndarray[np.int64_t, ndim=1] corner = np.zeros([lmax], dtype=np.int64)
        long[:] corner_view = corner
        np.ndarray[np.int64_t, ndim=1] tmp_corner = np.zeros([lmax], dtype=np.int64)
        long[:] tmp_corner_view = tmp_corner

        np.ndarray[np.float64_t, ndim=1] lower_corner_crd
        np.ndarray[np.float64_t, ndim=1] corner_crd
        np.ndarray[np.float64_t, ndim=1] tmp
        np.ndarray[np.float64_t, ndim=1] lower_bound
        np.ndarray[np.float64_t, ndim=1] upper_bound
        np.ndarray[np.float64_t, ndim=1] dx2, dy2, dz2

    assert radius >= 0, "radius must be non-negative"
    if radius == 0:
        return []

    lower_corner = c_lower_corner_of_containing_cube(atom_coordinate, origin_crd, upper_most_corner_crd, spacing)
    if lower_corner.shape[0] > 0:
        lower_corner_crd = c_get_corner_crd(lower_corner, grid_x, grid_y, grid_z)
        r = radius + cdistance(lower_corner_crd, atom_coordinate)

        tmp = np.ceil(r / spacing)
        count_i, count_j, count_k = np.array(tmp, dtype=int)

        corners = []
        for i in range(-count_i, count_i + 1):
            tmp_corner_view[0] = i
            for j in range(-count_j, count_j + 1):
                tmp_corner_view[1] = j
                for k in range(-count_k, count_k + 1):
                    tmp_corner_view[2] = k
                    for l in range(lmax):
                        corner_view[l] = lower_corner[l] + tmp_corner_view[l]
                    if c_greater_or_equal(corner,0) and c_less_or_equal(corner,upper_most_corner):
                        corner_crd = c_get_corner_crd(corner, grid_x, grid_y, grid_z)

                        if cdistance(corner_crd, atom_coordinate) <= radius:
                            corners.append(np.array([corner_view[0], corner_view[1], corner_view[2]], dtype=int))
        return corners
    else:
        lower_bound = origin_crd - radius
        upper_bound = upper_most_corner_crd + radius
        if np.any(atom_coordinate < lower_bound) or np.any(atom_coordinate > upper_bound):
            return []
        else:
            dx2 = (grid_x - atom_coordinate[0]) ** 2
            dy2 = (grid_y - atom_coordinate[1]) ** 2
            dz2 = (grid_z - atom_coordinate[2]) ** 2

            corners = []
            count_i, count_j, count_k = grid_counts

            for i in range(count_i):
                for j in range(count_j):
                    for k in range(count_k):
                        R = dx2[i] + dy2[j] + dz2[k]
                        R = sqrt(R)
                        if R <= radius:
                            corners.append(np.array([i, j, k], dtype=int))
            return corners


@cython.boundscheck(False)
def c_is_row_in_matrix( np.ndarray[np.int64_t, ndim=1] row, 
                        list matrix):
    cdef:
        np.ndarray[np.int64_t, ndim=1] r

    for r in matrix:
        if (row == r).all():
            return True
    return False


@cython.boundscheck(False)
def c_ten_corners(  np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                    np.ndarray[np.float64_t, ndim=1] origin_crd,
                    np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                    np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
                    np.ndarray[np.float64_t, ndim=1] spacing,
                    np.ndarray[np.int64_t, ndim=2]   eight_corner_shifts,
                    np.ndarray[np.int64_t, ndim=2]   six_corner_shifts,
                    np.ndarray[np.float64_t, ndim=1] grid_x,
                    np.ndarray[np.float64_t, ndim=1] grid_y,
                    np.ndarray[np.float64_t, ndim=1] grid_z ):
    """
    to find the ten corners as described in the Qin et al J Chem Theory Comput 2014, 10, 2824
    """
    cdef:
        list eight_corners, six_corners, three_corners, ten_corners
        int nearest_ind, furthest_ind
        int i
        np.ndarray[np.int64_t, ndim=1] nearest_corner
        np.ndarray[np.int64_t, ndim=1] corner

    eight_corners, nearest_ind, furthest_ind = c_containing_cube(atom_coordinate, origin_crd,
                                                                uper_most_corner_crd, spacing,
                                                                eight_corner_shifts,
                                                                grid_x, grid_y, grid_z)
    if not eight_corners:
        raise RuntimeError("Atom is outside the grid")

    nearest_corner  = eight_corners[nearest_ind]
    for i in range(len(nearest_corner)):
        if nearest_corner[i] == 0 or nearest_corner[i] == uper_most_corner[i]:
            raise RuntimeError("The nearest corner is on the grid boundary")

    six_corners = [nearest_corner + corner for corner in six_corner_shifts]

    three_corners = []
    for corner in six_corners:
        if not c_is_row_in_matrix(corner, eight_corners):
            three_corners.append(corner)

    eight_corners.pop(furthest_ind)
    ten_corners = eight_corners + three_corners
    return ten_corners
    

@cython.boundscheck(False)
def c_distr_charge_one_atom( str name,
                             np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                             double charge,
                             np.ndarray[np.float64_t, ndim=1] origin_crd,
                             np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                             np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
                             np.ndarray[np.float64_t, ndim=1] spacing,
                             np.ndarray[np.int64_t, ndim=2]   eight_corner_shifts,
                             np.ndarray[np.int64_t, ndim=2]   six_corner_shifts,
                             np.ndarray[np.float64_t, ndim=1] grid_x,
                             np.ndarray[np.float64_t, ndim=1] grid_y,
                             np.ndarray[np.float64_t, ndim=1] grid_z ):
    cdef:
        int i, j, k, row 
        list delta_vectors, ten_corners
        np.ndarray[np.int64_t, ndim=1] corner
        np.ndarray[np.float64_t, ndim=1] b_vector = np.zeros([10], dtype=float)
        np.ndarray[np.float64_t, ndim=2] a_matrix = np.zeros([10,10], dtype=float)
        np.ndarray[np.float64_t, ndim=1] corner_crd
        np.ndarray[np.float64_t, ndim=1] distributed_charges = np.zeros([10], dtype=float)

    ten_corners = c_ten_corners(atom_coordinate, origin_crd, uper_most_corner_crd, uper_most_corner,
                                spacing, eight_corner_shifts, six_corner_shifts, grid_x, grid_y, grid_z)
    b_vector[0] = charge
    a_matrix[0,:] = 1.0

    delta_vectors = []
    for corner in ten_corners:
        corner_crd = c_get_corner_crd(corner, grid_x, grid_y, grid_z)
        delta_vectors.append(corner_crd - atom_coordinate)

    for j in range(10):
        a_matrix[1][j] = delta_vectors[j][0]
        a_matrix[2][j] = delta_vectors[j][1]
        a_matrix[3][j] = delta_vectors[j][2]

    row = 3
    for i in range(3):
        for j in range(i, 3):
            row += 1
            for k in range(10):
                a_matrix[row][k] = delta_vectors[k][i] * delta_vectors[k][j]
    if name == "electrostatic":
        distributed_charges = np.linalg.solve(a_matrix, b_vector)
    else:
        distributed_charges = nnls(a_matrix, b_vector)[0]
    # distributed_charges = np.linalg.solve(a_matrix, b_vector)

    return ten_corners, distributed_charges


@cython.boundscheck(False)
def get_min_dists(
                        np.ndarray[np.float64_t, ndim=2] rec_crd,
                        np.ndarray[np.float64_t, ndim=2] lig_crd,
                        np.ndarray[np.float64_t, ndim=1] rec_lj_sigma,
                        np.ndarray[np.float64_t, ndim=1] lig_lj_sigma,
                        np.ndarray[np.float64_t, ndim=1] rec_vdw_radii,
                        np.ndarray[np.float64_t, ndim=1] lig_vdw_radii,
                        list rec_atom_names,
                        list lig_atom_names,
                        list rec_res_names,
                        list lig_res_names,
                        np.ndarray[float, ndim=2] rec_sasa,
                        np.ndarray[float, ndim=2] lig_sasa,
                        float surface_core_cutoff,
                        bint use_vdw,
                        bint exclude_H
                           ):
    cdef:
        int rec_ind, lig_ind, i
        # int ssr_ind, ssl_ind, scr_ind, scl_ind, csr_ind, csl_ind, ccr_ind, ccl_ind
        # int msr_ind, msl_ind, smr_ind, sml_ind, mcr_ind, mcl_ind, cmr_ind, cml_ind
        int rec_natoms = rec_crd.shape[0]
        int lig_natoms = lig_crd.shape[0]
        list rec_inds = []
        list lig_inds = []
        # dict rec, lig = {}
        dict sigmaR = {}
        dict sigmaL = {}
        dict indR = {}
        dict indL = {}
        dict dist = {}
        dict nameR = {}
        dict nameL = {}
        str rad_type, diameter
        list metal_ions = ["ZN", "CA", "MG", "SR"]
        list key_types = ["ss", "sc", "cs", "cc", "ms", "sm", "mc", "cm"]

    # make a dictionary from components for cleaner code
    rec = dict()
    lig = dict()
    rec["lj_sigma"] = rec_lj_sigma
    lig["lj_sigma"] = lig_lj_sigma
    rec["vdw_dia"] = rec_vdw_radii*2.
    lig["vdw_dia"] = lig_vdw_radii*2.
    rec["atom_names"] = rec_atom_names
    lig["atom_names"] = lig_atom_names
    rec["res_names"] = rec_res_names
    lig["res_names"] = lig_res_names
    rec["sasa"] = rec_sasa
    lig["sasa"] = lig_sasa
    # Set initial values for comparison
    for key in key_types:
        sigmaR[key] = 0.
        sigmaL[key] = 0.
        indR[key] = 0
        indL[key] = 0
        dist[key] = math.inf
        nameR[key] = "None"
        nameL[key] = "None"
    # Set either LJ_sigma or VDW diameter for atom size
    if use_vdw:
        diameter = "vdw_dia"
    else:
        diameter = "lj_sigma"
    # Build index list without hydrogen if desired
    for i in range(rec_natoms):
        if exclude_H:
            if "H" not in rec["atom_names"][i]:
                rec_inds.append(i)
        else:
            rec_inds.append(i)
    for i in range(lig_natoms):
        if exclude_H:
            if "H" not in lig["atom_names"][i]:
                lig_inds.append(i)
        else:
            lig_inds.append(i)
    # Find the minimum distances between rec and lig for various atom types
    # s = surface, c = core, m = metal ion
    for rec_ind in rec_inds:
        if rec["res_names"][rec_ind] not in metal_ions:
            if rec["sasa"][0][rec_ind] <= surface_core_cutoff:
                for lig_ind in lig_inds:
                    distance = cdistance(rec_crd[rec_ind], lig_crd[lig_ind])
                    if lig["res_names"][lig_ind] not in metal_ions:
                        if lig["sasa"][0][lig_ind] <= surface_core_cutoff:
                            if dist["cc"] > distance:
                                dist["cc"] = distance
                                indR["cc"] = rec_ind
                                indL["cc"] = lig_ind
                                sigmaR["cc"] = rec[diameter][rec_ind]
                                sigmaL["cc"] = lig[diameter][lig_ind]
                                nameR["cc"] = rec_atom_names[rec_ind]
                                nameL["cc"] = lig_atom_names[lig_ind]
                        else:
                            if dist["cs"] > distance:
                                dist["cs"] = distance
                                indR["cs"] = rec_ind
                                indL["cs"] = lig_ind
                                sigmaR["cs"] = rec[diameter][rec_ind]
                                sigmaL["cs"] = lig[diameter][lig_ind]
                                nameR["cs"] = rec_atom_names[rec_ind]
                                nameL["cs"] = lig_atom_names[lig_ind]
                    else:
                        if dist["cm"] > distance:
                                dist["cm"] = distance
                                indR["cm"] = rec_ind
                                indL["cm"] = lig_ind
                                sigmaR["cm"] = rec[diameter][rec_ind]
                                sigmaL["cm"] = lig[diameter][lig_ind]
                                nameR["cm"] = rec_atom_names[rec_ind]
                                nameL["cm"] = lig_atom_names[lig_ind]
            else:
                for lig_ind in lig_inds:
                    distance = cdistance(rec_crd[rec_ind], lig_crd[lig_ind])
                    if lig["res_names"][lig_ind] not in metal_ions:
                        if lig["sasa"][0][lig_ind] <= surface_core_cutoff:
                            if dist["sc"] > distance:
                                dist["sc"] = distance
                                indR["sc"] = rec_ind
                                indL["sc"] = lig_ind
                                sigmaR["sc"] = rec[diameter][rec_ind]
                                sigmaL["sc"] = lig[diameter][lig_ind]
                                nameR["sc"] = rec_atom_names[rec_ind]
                                nameL["sc"] = lig_atom_names[lig_ind]
                        else:
                            if dist["ss"] > distance:
                                dist["ss"] = distance
                                indR["ss"] = rec_ind
                                indL["ss"] = lig_ind
                                sigmaR["ss"] = rec[diameter][rec_ind]
                                sigmaL["ss"] = lig[diameter][lig_ind]
                                nameR["ss"] = rec_atom_names[rec_ind]
                                nameL["ss"] = lig_atom_names[lig_ind]
                    else:
                        if dist["sm"] > distance:
                                dist["sm"] = distance
                                indR["sm"] = rec_ind
                                indL["sm"] = lig_ind
                                sigmaR["sm"] = rec[diameter][rec_ind]
                                sigmaL["sm"] = lig[diameter][lig_ind]
                                nameR["sm"] = rec_atom_names[rec_ind]
                                nameL["sm"] = lig_atom_names[lig_ind]
        else:
            for lig_ind in lig_inds:
                distance = cdistance(rec_crd[rec_ind], lig_crd[lig_ind])
                if lig["res_names"][lig_ind] not in metal_ions:
                    if lig["sasa"][0][lig_ind] <= surface_core_cutoff:
                        if dist["mc"] > distance:
                            dist["mc"] = distance
                            indR["mc"] = rec_ind
                            indL["mc"] = lig_ind
                            sigmaR["mc"] = rec[diameter][rec_ind]
                            sigmaL["mc"] = lig[diameter][lig_ind]
                            nameR["mc"] = rec_atom_names[rec_ind]
                            nameL["mc"] = lig_atom_names[lig_ind]
                    else:
                        if dist["ms"] > distance:
                            dist["ms"] = distance
                            indR["ms"] = rec_ind
                            indL["ms"] = lig_ind
                            sigmaR["ms"] = rec[diameter][rec_ind]
                            sigmaL["ms"] = lig[diameter][lig_ind]
                            nameR["ms"] = rec_atom_names[rec_ind]
                            nameL["ms"] = lig_atom_names[lig_ind]

    # result = (sigmaR_ss, sigmaL_ss, sigmaR_sc, sigmaL_sc,
    #           sigmaR_cs, sigmaL_cs, sigmaR_cc, sigmaL_cc,
    #           sigmaR_ms, sigmaL_ms, sigmaR_sm, sigmaL,sm,
    #           sigmaR_mc, sigmaL_mc, sigmaR_cm, sigmaL_cm,
    #           dmin_ss, dmin_sc, dmin_cs, dmin_cc, ind_list)
    result = {"sigmaR": sigmaR, "sigmaL": sigmaL, "dist": dist, "indR": indR, "indL": indL, "nameR": nameR, "nameL": nameL}
    return result

@cython.boundscheck(False)
# Custom function to create a unique key for label pairs
def get_pair_key(label1, label2):
    return label1 < label2 and f"{label1}:{label2}" or f"{label2}:{label1}"

# Function to calculate distances
def calculate_distances(np.ndarray[double, ndim=2] array1,
                        np.ndarray[double, ndim=2] array2,
                        list array1_labels,
                        list array2_labels,
                        np.ndarray[double, ndim=1] r_sigmas,
                        np.ndarray[double, ndim=1] l_sigmas,
                        np.ndarray[double, ndim=1] r_radii,
                        np.ndarray[double, ndim=1] l_radii,
                        np.ndarray[double, ndim=1] r_mass,
                        np.ndarray[double, ndim=1] l_mass,
                        np.ndarray[float, ndim=2] rec_sasa,
                        np.ndarray[float, ndim=2] lig_sasa,
                        float surface_core_cutoff,
                        double cutoff_distance,
                        str system_name,
                        bint use_vdw,
                        bint exclude_H
                        ):
    cdef:
        dict distances = {}
        dict indR = {}
        dict indL = {}
        dict sigmaR = {}
        dict sigmaL = {}
    cdef double previous_distance
    cdef double current_distance
    cdef int i, j
    cdef list rec_inds = []
    cdef list lig_inds = []
    cdef list row
    cdef list rows = []
    cdef rec_natoms = array1.shape[0]
    cdef lig_natoms = array2.shape[0]

    for i in range(rec_natoms):
        if exclude_H:
            if "H" not in array1_labels[i]:
                rec_inds.append(i)
        else:
            rec_inds.append(i)
    for i in range(lig_natoms):
        if exclude_H:
            if "H" not in array2_labels[i]:
                    lig_inds.append(i)
        else:
            lig_inds.append(i)

    for i in rec_inds:
        for j in lig_inds:
            # Select which atom size source
            if use_vdw:
                rs = r_radii[i]*2.
                ls = l_radii[j]*2.
            else:
                rs = r_sigmas[i]
                ls = l_sigmas[j]
            rm = r_mass[i]
            lm = l_mass[j]
            current_distance = cdistance(array1[i], array2[j])
            array1_label = array1_labels[i]
            array2_label = array2_labels[j]
            pair_key = get_pair_key(array1_label, array2_label)
            if current_distance <= cutoff_distance:
                row = [rs,ls,current_distance,i,j,array1_label,array2_label,rm,lm,system_name]
                rows.append(row)
    return rows

@cython.boundscheck(False)
def c_cal_potential_grid_pp(   str name,
                            np.ndarray[np.float64_t, ndim=2] crd,
                            np.ndarray[np.float64_t, ndim=1] grid_x,
                            np.ndarray[np.float64_t, ndim=1] grid_y,
                            np.ndarray[np.float64_t, ndim=1] grid_z,
                            np.ndarray[np.float64_t, ndim=1] origin_crd,
                            np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                            np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
                            np.ndarray[np.float64_t, ndim=1] spacing,
                            np.ndarray[np.int64_t, ndim=1]   grid_counts,
                            np.ndarray[np.float64_t, ndim=1] charges,
                            np.ndarray[np.float64_t, ndim=1] lj_sigma,
                            np.ndarray[np.float64_t, ndim=1] vdw_radii,
                            np.ndarray[np.float64_t, ndim=1] clash_radii,
                            list atom_list,
                            np.ndarray[float, ndim=2] molecule_sasa,
                            np.ndarray[float, ndim=2] sasa_cutoffs,
                            list rec_res_names,
                            float rec_core_scaling,
                            float rec_surface_scaling,
                            float rec_metal_scaling):

    cdef:
        list corners
        list metal_ions = ["ZN", "CA", "MG", "SR"]
        int natoms = crd.shape[0]
        int i_max = grid_x.shape[0]
        int j_max = grid_y.shape[0]
        int k_max = grid_z.shape[0]
        int i, j, k
        int atom_ind
        double charge, lj_diameter
        double d, exponent
        double dx_tmp, dy_tmp
        np.ndarray[np.float64_t, ndim=3] grid = np.zeros([i_max, j_max, k_max], dtype=float)
        np.ndarray[np.float64_t, ndim=3] grid_tmp
        double[:,:,:] grid_tmp_view
        np.ndarray[np.float64_t, ndim=1] atom_coordinate
        np.ndarray[np.float64_t, ndim=1] dx2, dy2, dz2

    if name in ["LJa","LJr","electrostatic","water"]:
        if name == "LJa":
            exponent = 3.
        elif name == "LJr":
            exponent = 6.
        elif name == "electrostatic":
            exponent = 0.5
        elif name == "water":
            exponent = 1.
        else:
            raise RuntimeError("Wrong grid name %s"%name)
        grid_tmp = np.empty([i_max, j_max, k_max], dtype=float)
        grid_tmp_view = grid_tmp
        for atom_ind in range(natoms):
            atom_coordinate = crd[atom_ind]
            dx2 = (atom_coordinate[0] - grid_x) ** 2
            dy2 = (atom_coordinate[1] - grid_y) ** 2
            dz2 = (atom_coordinate[2] - grid_z) ** 2
            if name == "water":
                grid_tmp[grid_tmp > 0.] = 0.
                charge = molecule_sasa[0][atom_ind]
                lj_diameter = vdw_radii[atom_ind]
                surface_layer = lj_diameter + 1.4
                corners = c_corners_within_radius(atom_coordinate,
                                                  surface_layer,
                                                  origin_crd,
                                                  uper_most_corner_crd,
                                                  uper_most_corner,
                                                  spacing,
                                                  grid_x,
                                                  grid_y,
                                                  grid_z,
                                                  grid_counts)

                for i, j, k in corners:
                    grid_tmp_view[i, j, k] = 1.
            else:
                charge = charges[atom_ind]
                lj_diameter = lj_sigma[atom_ind]
                for i in range(i_max):
                    dx_tmp = dx2[i]
                    for j in range(j_max):
                        dy_tmp = dy2[j]
                        for k in range(k_max):
                            d = dx_tmp + dy_tmp + dz2[k]
                            d = d**exponent
                            grid_tmp_view[i,j,k] = charge / d
                corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
                                                    uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)

                for i, j, k in corners:
                    grid_tmp[i,j,k] = 0

            # corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
            #                                     uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
            #
            # for i, j, k in corners:
            #     grid_tmp[i,j,k] = 0

            grid += grid_tmp
    else:
        for atom_ind in atom_list:
            atom_coordinate = crd[atom_ind]
            lj_diameter = clash_radii[atom_ind]
            corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
                                              uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
            for i, j, k in corners:
                grid[i, j, k] = 1.
    return grid

@cython.boundscheck(False)
def c_asa_frame(      np.ndarray[np.float64_t, ndim=2] crd,
                            np.ndarray[np.float64_t, ndim=1] atom_radii,
                            np.ndarray[np.float64_t, ndim=1] spacing,
                            np.ndarray[np.float64_t, ndim=2] sphere_points,
                            int n_sphere_points,
                            int natoms_i,
                            int atomind
                            ):

    cdef:
        Py_ssize_t i, j, k, index
        int natoms = crd.shape[0]
        Py_ssize_t n_neighbor_indices, k_closest_neighbor, k_prime
        float atom_radius_i, atom_radius_j
        float radius_cutoff, radius_cutoff2, r2, r
        float constant = 4.0 * np.pi / n_sphere_points
        bint is_accessible
        double[:,:] crd_view = crd
        np.ndarray[np.int64_t, ndim=1] grid_crd = np.empty(3, dtype=np.int64)
        np.ndarray[np.float64_t, ndim=1] sphere_point_coordinate
        np.ndarray[np.float64_t, ndim=1] r_i = np.zeros([3], dtype=np.float64)
        np.ndarray[np.float64_t, ndim=1] r_j = np.zeros([3], dtype=np.float64)
        np.ndarray[np.float64_t, ndim=1] r_ij = np.zeros([3], dtype=np.float64)
        np.ndarray[np.float64_t, ndim=1] r_jk = np.zeros([3], dtype=np.float64)
        double[:] r_iv = r_i
        double[:] r_jv = r_j
        double[:] r_ijv = r_ij
        double[:] r_jkv = r_jk
        np.ndarray[np.int64_t, ndim = 1] neighbor_indices = np.empty(natoms, dtype=np.int64)
        np.ndarray[np.float64_t, ndim = 3] centered_sphere_points = np.zeros([natoms,n_sphere_points, 3], dtype=np.float64)
        np.ndarray[np.float64_t, ndim = 3] accessible_sphere_points = np.zeros([natoms_i, n_sphere_points, 4], dtype=np.float64)

    for i in range(natoms)[atomind:atomind+natoms_i]:
        atom_radius_i = atom_radii[i]
        r_iv = crd_view[i]

        # Get all the atoms close to atom i
        n_neighbor_indices = 0
        for j in range(natoms):
            if i == j:
                continue

            r_jv[0] = crd_view[j][0]
            r_jv[1] = crd_view[j][1]
            r_jv[2] = crd_view[j][2]
            r_ijv[0] = r_iv[0]-r_jv[0]
            r_ijv[1] = r_iv[1]-r_jv[1]
            r_ijv[2] = r_iv[2]-r_jv[2]
            atom_radius_j = atom_radii[j]

            # look for atoms j around atom i
            radius_cutoff = atom_radius_i+atom_radius_j
            radius_cutoff2 = radius_cutoff*radius_cutoff
            r2 = cdot(r_ijv, r_ijv)

            if r2 < radius_cutoff2:
                neighbor_indices[n_neighbor_indices] = j
                n_neighbor_indices += 1
            if r2 < 1e-10:
                print("This code is known to fail when atoms are too close")
                break
        # Center the sphere points on atom i
        for j in range(n_sphere_points):
            centered_sphere_points[i, j, 0] = crd_view[i, 0] + atom_radius_i*sphere_points[j, 0]
            centered_sphere_points[i, j, 1] = crd_view[i, 1] + atom_radius_i*sphere_points[j, 1]
            centered_sphere_points[i, j, 2] = crd_view[i, 2] + atom_radius_i*sphere_points[j, 2]

        # Check if these points are accessible
        k_closest_neighbor = 0
        for j in range(n_sphere_points):
            is_accessible = True
            r_j[0] = centered_sphere_points[i, j, 0]
            r_j[1] = centered_sphere_points[i, j, 1]
            r_j[2] = centered_sphere_points[i, j, 2]
            for k in range(n_neighbor_indices + k_closest_neighbor)[k_closest_neighbor:]:
                k_prime = k % n_neighbor_indices
                r = atom_radii[neighbor_indices[k_prime]]
                index = neighbor_indices[k_prime]
                r_jk[0] = r_j[0] - crd_view[index][0]
                r_jk[1] = r_j[1] - crd_view[index][1]
                r_jk[2] = r_j[2] - crd_view[index][2]
                if cdot(r_jkv,r_jkv) < r*r:
                    k_closest_neighbor = k
                    is_accessible = False
                    break;
            if(is_accessible):
                accessible_sphere_points[i - atomind, j, 0] = centered_sphere_points[i, j, 0]
                accessible_sphere_points[i - atomind, j, 1] = centered_sphere_points[i, j, 1]
                accessible_sphere_points[i - atomind, j, 2] = centered_sphere_points[i, j, 2]
                accessible_sphere_points[i - atomind, j, 3] = atom_radius_i

    return accessible_sphere_points

@cython.boundscheck(True)
def c_generate_sphere_points(int n_points):
    cdef:
        int i
        float y, r, phi
        float inc = M_PI * (3.0 - sqrt(5.0))
        float offset = 2.0 / n_points
        np.ndarray[np.float64_t, ndim = 2] sphere_points = np.empty([n_points, 3], dtype=np.float64)

    for i in range(n_points):
        y = i * offset - 1.0 + (offset / 2.0)
        r = sqrt(1.0 - y*y)
        phi = i * inc

        sphere_points[i, 0] = cos(phi) * r
        sphere_points[i, 1] = y
        sphere_points[i, 2] = sin(phi) * r
    return sphere_points


@cython.boundscheck(False)
def c_sasa(         np.ndarray[np.float64_t, ndim=2] crd,
                    np.ndarray[np.float64_t, ndim=1] atom_radii,
                    np.ndarray[np.float64_t, ndim=1] spacing,
                    float probe_size,
                    int n_sphere_points,
                    int natoms_i,
                    int atomind
                    ):
    cdef:
        int natoms = crd.shape[0]
        int i, j
        np.ndarray[np.float64_t, ndim = 2] sphere_points = np.empty([n_sphere_points, 3], dtype=np.float64)
        np.ndarray[np.float64_t, ndim = 3] accessible_sphere_points = np.zeros([natoms_i, n_sphere_points, 4], dtype=np.float64)
    sphere_points = c_generate_sphere_points(n_sphere_points)
    accessible_sphere_points = c_asa_frame(crd, atom_radii+probe_size, spacing, sphere_points, n_sphere_points, natoms_i, atomind)

    return accessible_sphere_points

@cython.boundscheck(False)
def c_cal_charge_grid_pp_mp(  str name,
                        np.ndarray[np.float64_t, ndim=2] crd,
                        np.ndarray[np.float64_t, ndim=1] grid_x,
                        np.ndarray[np.float64_t, ndim=1] grid_y,
                        np.ndarray[np.float64_t, ndim=1] grid_z,
                        np.ndarray[np.float64_t, ndim=1] origin_crd,
                        np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                        np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
                        np.ndarray[np.float64_t, ndim=1] spacing,
                        np.ndarray[np.int64_t, ndim=2]   eight_corner_shifts,
                        np.ndarray[np.int64_t, ndim=2]   six_corner_shifts,
                        np.ndarray[np.int64_t, ndim=1]   grid_counts,
                        np.ndarray[np.float64_t, ndim=1] charges,
                        np.ndarray[np.float64_t, ndim=1] lj_sigma,
                        np.ndarray[np.float64_t, ndim=1] vdw_radii,
                        np.ndarray[np.float64_t, ndim=1] clash_radii,
                        list atom_list,
                        int natoms_i,
                        int atomind,
                        np.ndarray[float, ndim=2] molecule_sasa,
                        np.ndarray[float, ndim=2] sasa_cutoffs,
                        list lig_res_names,
                        float lig_core_scaling,
                        float lig_surface_scaling,
                        float lig_metal_scaling):

    cdef:
        list corners
        list metal_ions = ["ZN", "CA", "MG", "SR"]
        int atom_ind, i, j, k, l, m, n
        int ind, num_corners
        int natoms = crd.shape[0]
        int i_max = grid_x.shape[0]
        int j_max = grid_y.shape[0]
        int k_max = grid_z.shape[0]
        double charge, lj_diameter
        list ten_corners, six_corners
        np.ndarray[np.float64_t, ndim=1] distributed_charges
        np.ndarray[np.float64_t, ndim=1] atom_coordinate
        np.ndarray[np.float64_t, ndim=3] grid = np.zeros([i_max, j_max, k_max], dtype=float)
        double[:,:,:] grid_view = grid
        np.ndarray[np.float64_t, ndim = 3] grid_tmp
        double[:,:,:] grid_tmp_view
        long[:,:] ten_corners_view
        double[:,:] crd_view = crd
        float[:,:] molecule_sasa_view = molecule_sasa

    assert name in ["occupancy", "sasa", "water", "LJa", "LJr", "electrostatic"], "Name %s not allowed"%name

    if name in ["LJa", "LJr", "electrostatic"]:
        for atom_ind in range(atomind,atomind+natoms_i):
            atom_coordinate = crd[atom_ind]
            charge = charges[atom_ind]
            ten_corners, distributed_charges = c_distr_charge_one_atom(name, atom_coordinate, charge,
                                                                    origin_crd, uper_most_corner_crd,
                                                                    uper_most_corner, spacing,
                                                                    eight_corner_shifts, six_corner_shifts,
                                                                    grid_x, grid_y, grid_z)

            ten_corners_view = c_list_to_array_long(ten_corners)
            for i in range(10):
                l = ten_corners_view[i][0]
                m = ten_corners_view[i][1]
                n = ten_corners_view[i][2]
                grid[l, m, n] += distributed_charges[i]
    elif name == "water":
        for atom_ind in range(atomind,atomind+natoms_i):
            atom_coordinate = crd[atom_ind]
            charge = molecule_sasa_view[0][atom_ind]
            lj_diameter = vdw_radii[atom_ind]

            surface_layer = lj_diameter + 1.4
            corners = c_corners_within_radius(atom_coordinate, surface_layer, origin_crd,
                                              uper_most_corner_crd,
                                              uper_most_corner, spacing, grid_x, grid_y, grid_z,
                                              grid_counts)
            grid_tmp = np.zeros([i_max, j_max, k_max], dtype=float)
            grid_tmp_view = grid_tmp
            # num_corners = len(corners)
            # for ind in range(num_corners):
            for i,j,k in corners:
                # corner = corners[ind]
                # i = corner[0]
                # j = corner[1]
                # k = corner[2]
                grid_tmp_view[i, j, k] = 1.

            # corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
            #                               uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)


            # for i, j, k in corners:
            #     grid_tmp[i, j, k] = 0

            # grid += grid_tmp

    else:
        for atom_ind in atom_list:
            atom_coordinate = crd[atom_ind]
            lj_diameter = clash_radii[atom_ind]
            corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
                                              uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
            for i, j, k in corners:
                grid[i, j, k] = 1.

    return grid

# TODO: Delete these old functions as they don't serve any purpose currently
# @cython.boundscheck(False)
# def c_cal_charge_grid_pp(  str name,
#                         np.ndarray[np.float64_t, ndim=2] crd,
#                         np.ndarray[np.float64_t, ndim=1] grid_x,
#                         np.ndarray[np.float64_t, ndim=1] grid_y,
#                         np.ndarray[np.float64_t, ndim=1] grid_z,
#                         np.ndarray[np.float64_t, ndim=1] origin_crd,
#                         np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
#                         np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
#                         np.ndarray[np.float64_t, ndim=1] spacing,
#                         np.ndarray[np.int64_t, ndim=2]   eight_corner_shifts,
#                         np.ndarray[np.int64_t, ndim=2]   six_corner_shifts,
#                         np.ndarray[np.int64_t, ndim=1]   grid_counts,
#                         np.ndarray[np.float64_t, ndim=1] charges,
#                         np.ndarray[np.float64_t, ndim=1] lj_sigma,
#                         np.ndarray[np.float64_t, ndim=1] clash_radii,
#                         list atom_list,
#                         np.ndarray[float, ndim=2] molecule_sasa,
#                         np.ndarray[float, ndim=2] sasa_cutoffs,
#                         list lig_res_names,
#                         float lig_core_scaling,
#                         float lig_surface_scaling,
#                         float lig_metal_scaling):
#
#     cdef:
#         list corners
#         list metal_ions = ["ZN", "CA", "MG", "SR"]
#         int atom_ind, i, l, m, n
#         int natoms = crd.shape[0]
#         int i_max = grid_x.shape[0]
#         int j_max = grid_y.shape[0]
#         int k_max = grid_z.shape[0]
#         double charge, lj_diameter
#         list ten_corners, six_corners
#         np.ndarray[np.float64_t, ndim=1] distributed_charges
#         np.ndarray[np.float64_t, ndim=1] atom_coordinate
#         np.ndarray[np.float64_t, ndim=3] grid = np.zeros([i_max, j_max, k_max], dtype=float)
#
#     assert name in ["occupancy", "sasa", "water", "LJa", "LJr", "electrostatic"], "Name %s not allowed"%name
#
#     if name in ["LJa", "LJr", "electrostatic"]:
#         for atom_ind in range(natoms):
#             atom_coordinate = crd[atom_ind]
#             charge = charges[atom_ind]
#             ten_corners, distributed_charges = c_distr_charge_one_atom(name, atom_coordinate, charge,
#                                                                     origin_crd, uper_most_corner_crd,
#                                                                     uper_most_corner, spacing,
#                                                                     eight_corner_shifts, six_corner_shifts,
#                                                                     grid_x, grid_y, grid_z)
#             for i in range(len(ten_corners)):
#                 l, m, n = ten_corners[i]
#                 grid[l, m, n] += distributed_charges[i]
#     elif name == "water":
#         for atom_ind in range(natoms):
#             atom_coordinate = crd[atom_ind]
#             charge = molecule_sasa[0][atom_ind]
#             lj_diameter = clash_radii[atom_ind]
#
#             surface_layer = lj_diameter + 1.4
#             corners = c_corners_within_radius(atom_coordinate, surface_layer, origin_crd,
#                                               uper_most_corner_crd,
#                                               uper_most_corner, spacing, grid_x, grid_y, grid_z,
#                                               grid_counts)
#             grid_tmp = np.empty([i_max, j_max, k_max], dtype=float)
#             dx2 = (atom_coordinate[0] - grid_x) ** 2
#             dy2 = (atom_coordinate[1] - grid_y) ** 2
#             dz2 = (atom_coordinate[2] - grid_z) ** 2
#             for i, j, k in corners:
#                 dx_tmp = dx2[i]
#                 dy_tmp = dy2[j]
#                 d = dx_tmp + dy_tmp + dz2[k]
#                 grid_tmp[i, j, k] = 1.
#
#             # corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#             #                               uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#
#
#             # for i, j, k in corners:
#             #     grid_tmp[i, j, k] = 0
#
#             grid += grid_tmp
#
#     else:
#         for atom_ind in atom_list:
#             atom_coordinate = crd[atom_ind]
#             if lig_res_names[atom_ind] in metal_ions:
#                 scale = lig_metal_scaling
#             else:
#                 if sasa_cutoffs[0][atom_ind] > 0.:
#                     scale = lig_surface_scaling
#                 else:
#                     scale = lig_core_scaling
#             lj_diameter = clash_radii[atom_ind] * scale
#             corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#                                               uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#             for i, j, k in corners:
#                 grid[i, j, k] = 1.
#
#     return grid

# @cython.boundscheck(False)
# def c_cal_potential_grid(   str name,
#                             np.ndarray[np.float64_t, ndim=2] crd,
#                             np.ndarray[np.float64_t, ndim=1] grid_x,
#                             np.ndarray[np.float64_t, ndim=1] grid_y,
#                             np.ndarray[np.float64_t, ndim=1] grid_z,
#                             np.ndarray[np.float64_t, ndim=1] origin_crd,
#                             np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
#                             np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
#                             np.ndarray[np.float64_t, ndim=1] spacing,
#                             np.ndarray[np.int64_t, ndim=1]   grid_counts,
#                             np.ndarray[np.float64_t, ndim=1] charges,
#                             np.ndarray[np.float64_t, ndim=1] lj_sigma,
#                             list atom_list,
#                             np.ndarray[float, ndim=2] molecule_sasa,
#                             list rec_res_names,
#                             float rec_core_scaling,
#                             float rec_surface_scaling,
#                             float rec_metal_scaling,
#                             float rho,
#                             np.ndarray[np.float64_t, ndim=3] sasai_grid):
#
#     cdef:
#         list corners, rho_i_corners
#         list metal_ions = ["ZN", "CA", "MG", "SR"]
#         int natoms = crd.shape[0]
#         int i_max = grid_x.shape[0]
#         int j_max = grid_y.shape[0]
#         int k_max = grid_z.shape[0]
#         int i, j, k
#         int atom_ind
#         double charge, lj_diameter
#         double d, exponent
#         double dx_tmp, dy_tmp
#         np.ndarray[np.float64_t, ndim=3] grid = np.zeros([i_max, j_max, k_max], dtype=float)
#         np.ndarray[np.float64_t, ndim=3] grid_tmp
#         np.ndarray[np.float64_t, ndim=1] atom_coordinate
#         np.ndarray[np.float64_t, ndim=1] dx2, dy2, dz2
#
#     if name[:4] != "SASA":
#
#         if name == "LJa":
#             exponent = 3.
#         elif name == "LJr":
#             exponent = 6.
#         elif name == "electrostatic":
#             exponent = 0.5
#         else:
#             raise RuntimeError("Wrong grid name %s"%name)
#
#         grid_tmp = np.empty([i_max, j_max, k_max], dtype=float)
#         for atom_ind in range(natoms):
#             atom_coordinate = crd[atom_ind]
#             charge = charges[atom_ind]
#             lj_diameter = lj_sigma[atom_ind]
#
#             dx2 = (atom_coordinate[0] - grid_x)**2
#             dy2 = (atom_coordinate[1] - grid_y)**2
#             dz2 = (atom_coordinate[2] - grid_z)**2
#
#             for i in range(i_max):
#                 dx_tmp = dx2[i]
#                 for j in range(j_max):
#                     dy_tmp = dy2[j]
#                     for k in range(k_max):
#
#                         d = dx_tmp + dy_tmp + dz2[k]
#                         d = d**exponent
#                         grid_tmp[i,j,k] = charge / d
#
#             corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#                                                 uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#
#             for i, j, k in corners:
#                 grid_tmp[i,j,k] = 0
#
#             grid += grid_tmp
#     # TODO: Add SASA grid as replacement for occupancy grid
#     else:
#         rho_i_corners = []
#         if name == "SASAi":
#             for atom_ind in atom_list: # for "SASAi" grid
#                 atom_coordinate = crd[atom_ind]
#                 if rec_res_names[atom_ind] not in metal_ions:
#                     if molecule_sasa[0][atom_ind] > 0.:  # surface atom
#                         lj_diameter = lj_sigma[atom_ind] * rec_surface_scaling
#                         corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#                                                           uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#                         for i, j, k in corners:
#                             grid[i, j, k] = rho
#                             rho_i_corners.append(np.array([i, j, k], dtype=int))
#
#                     elif molecule_sasa[0][atom_ind] <= 0.: # core atom
#                         lj_diameter = lj_sigma[atom_ind] * rec_core_scaling
#                         corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#                                                           uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#                         for i, j, k in corners:
#                             grid[i, j, k] = rho
#                             rho_i_corners.append(np.array([i, j, k], dtype=int))
#                 else:
#                     lj_diameter = lj_sigma[atom_ind] * rec_metal_scaling
#                     corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#                                                       uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#                     for i, j, k in corners:
#                         grid[i, j, k] = rho
#                         rho_i_corners.append(np.array([i, j, k], dtype=int))
#         else:
#             for atom_ind in atom_list: # for "SASAr" grid
#                 atom_coordinate = crd[atom_ind]
#                 if molecule_sasa[0][atom_ind] > 0.: # surface atom
#                     lj_diameter = (lj_sigma[atom_ind] * rec_surface_scaling) + 2.8 # 2.8A corresponds to H2O diameter
#                     corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#                                                       uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#                     for i, j, k in corners:
#                         offset = origin_crd/spacing
#                         if sasai_grid[i+int(offset[0]),j,k] != rho:
#                             grid[i,j,k] = 1.
#
#     return grid
#
#
# @cython.boundscheck(False)
# def c_cal_charge_grid(  str name,
#                         np.ndarray[np.float64_t, ndim=2] crd,
#                         np.ndarray[np.float64_t, ndim=1] charges,
#                         np.ndarray[np.float64_t, ndim=1] origin_crd,
#                         np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
#                         np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
#                         np.ndarray[np.float64_t, ndim=1] spacing,
#                         np.ndarray[np.int64_t, ndim=2]   eight_corner_shifts,
#                         np.ndarray[np.int64_t, ndim=2]   six_corner_shifts,
#                         np.ndarray[np.float64_t, ndim=1] grid_x,
#                         np.ndarray[np.float64_t, ndim=1] grid_y,
#                         np.ndarray[np.float64_t, ndim=1] grid_z ):
#
#     cdef:
#         int atom_ind, i, l, m, n
#         int natoms = crd.shape[0]
#         int i_max = grid_x.shape[0]
#         int j_max = grid_y.shape[0]
#         int k_max = grid_z.shape[0]
#         double charge
#         list ten_corners
#         np.ndarray[np.float64_t, ndim=1] distributed_charges
#         np.ndarray[np.float64_t, ndim=1] atom_coordinate
#         np.ndarray[np.float64_t, ndim=3] grid = np.zeros([i_max, j_max, k_max], dtype=float)
#
#     assert name in ["SASAr", "SASAi", "LJa", "LJr", "electrostatic"], "Name %s not allowed"%name
#
#     if name[:4] != "SASA":
#         for atom_ind in range(natoms):
#             atom_coordinate = crd[atom_ind]
#             charge = charges[atom_ind]
#             ten_corners, distributed_charges = c_distr_charge_one_atom( atom_coordinate, charge,
#                                                                     origin_crd, uper_most_corner_crd,
#                                                                     uper_most_corner, spacing,
#                                                                     eight_corner_shifts, six_corner_shifts,
#                                                                     grid_x, grid_y, grid_z)
#             for i in range(len(ten_corners)):
#                 l, m, n = ten_corners[i]
#                 grid[l, m, n] += distributed_charges[i]
#     else:
#         for atom_ind in range(natoms):
#             atom_coordinate = crd[atom_ind]
#             ten_corners = c_ten_corners(atom_coordinate, origin_crd, uper_most_corner_crd,
#                                     uper_most_corner, spacing, eight_corner_shifts, six_corner_shifts,
#                                     grid_x, grid_y, grid_z )
#             for i in range(len(ten_corners)):
#                 l, m, n = ten_corners[i]
#                 grid[l, m, n] = 1.0
#     return grid
#
# @cython.boundscheck(False)
# def c_cal_charge_grid_new(  str name,
#                         np.ndarray[np.float64_t, ndim=2] crd,
#                         np.ndarray[np.float64_t, ndim=1] grid_x,
#                         np.ndarray[np.float64_t, ndim=1] grid_y,
#                         np.ndarray[np.float64_t, ndim=1] grid_z,
#                         np.ndarray[np.float64_t, ndim=1] origin_crd,
#                         np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
#                         np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
#                         np.ndarray[np.float64_t, ndim=1] spacing,
#                         np.ndarray[np.int64_t, ndim=2]   eight_corner_shifts,
#                         np.ndarray[np.int64_t, ndim=2]   six_corner_shifts,
#                         np.ndarray[np.int64_t, ndim=1]   grid_counts,
#                         np.ndarray[np.float64_t, ndim=1] charges,
#                         np.ndarray[np.float64_t, ndim=1] lj_sigma,
#                         np.ndarray[float, ndim=2] molecule_sasa):
#
#     cdef:
#         int atom_ind, i, l, m, n
#         int natoms = crd.shape[0]
#         int i_max = grid_x.shape[0]
#         int j_max = grid_y.shape[0]
#         int k_max = grid_z.shape[0]
#         double charge
#         list ten_corners, six_corners
#         np.ndarray[np.float64_t, ndim=1] distributed_charges
#         np.ndarray[np.float64_t, ndim=1] atom_coordinate
#         np.ndarray[np.float64_t, ndim=3] grid = np.zeros([i_max, j_max, k_max], dtype=float)
#
#     assert name in ["SASAi", "SASAr", "LJa", "LJr", "electrostatic"], "Name %s not allowed"%name
#
#     for atom_ind in range(natoms):
#         atom_coordinate = crd[atom_ind]
#         charge = charges[atom_ind]
#         ten_corners, distributed_charges = c_distr_charge_one_atom( atom_coordinate, charge,
#                                                                 origin_crd, uper_most_corner_crd,
#                                                                 uper_most_corner, spacing,
#                                                                 eight_corner_shifts, six_corner_shifts,
#                                                                 grid_x, grid_y, grid_z)
#         for i in range(len(ten_corners)):
#             l, m, n = ten_corners[i]
#             # below is effectively grid[l, m, n] += distributed_charges[i] for complex nums
#             grid[l, m, n] += distributed_charges[i]
#
#     return grid
#
# @cython.boundscheck(False)
# def c_cal_lig_sasa_grid(  str name,
#                         np.ndarray[np.float64_t, ndim=2] crd,
#                         np.ndarray[np.float64_t, ndim=1] grid_x,
#                         np.ndarray[np.float64_t, ndim=1] grid_y,
#                         np.ndarray[np.float64_t, ndim=1] grid_z,
#                         np.ndarray[np.float64_t, ndim=1] origin_crd,
#                         np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
#                         np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
#                         np.ndarray[np.float64_t, ndim=1] spacing,
#                         np.ndarray[np.int64_t, ndim=2]   eight_corner_shifts,
#                         np.ndarray[np.int64_t, ndim=2]   six_corner_shifts,
#                         np.ndarray[np.int64_t, ndim=1]   grid_counts,
#                         np.ndarray[np.float64_t, ndim=1] charges,
#                         np.ndarray[np.float64_t, ndim=1] lj_sigma,
#                         np.ndarray[float, ndim=2] molecule_sasa,
#                         np.ndarray[np.float64_t, ndim=3] sasai_grid,
#                           list rho_i_corners):
#
#     cdef:
#         int atom_ind, i, l, m, n
#         int natoms = crd.shape[0]
#         int i_max = grid_x.shape[0]
#         int j_max = grid_y.shape[0]
#         int k_max = grid_z.shape[0]
#         double charge
#         list ten_corners, six_corners, rho_i_zeros
#         np.ndarray[np.float64_t, ndim=1] distributed_charges
#         np.ndarray[np.float64_t, ndim=1] atom_coordinate
#         np.ndarray[np.float64_t, ndim=3] grid = np.zeros([i_max, j_max, k_max], dtype=float)
#
#     assert name in ["SASAi", "SASAr", "LJa", "LJr", "electrostatic"], "Name %s not allowed"%name
#
#     roh_i = 9.
#     for atom_ind in range(natoms):
#         atom_coordinate = crd[atom_ind]
#         if name == "SASAi":
#             if molecule_sasa[0][atom_ind] < 0.01:  # core atom
#                 lj_diameter = lj_sigma[atom_ind] * np.sqrt(1.5)
#                 corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#                                                   uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#                 for i, j, k in corners:
#                     grid[i, j, k] = roh_i
#                     rho_i_corners.append([i,j,k])
#         elif name == "SASAr":
#             if molecule_sasa[0][atom_ind] > 0.01:  # surface atom
#                 lj_diameter = lj_sigma[atom_ind]
#                 corners = c_corners_within_radius(atom_coordinate, lj_diameter, origin_crd, uper_most_corner_crd,
#                                                   uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#                 for i, j, k in corners:
#                     if sasai_grid[i,j,k] == 0:
#                         grid[i, j, k] = 1
#             # rho_i_hits = np.array(np.where(sasai_grid == roh_i)).transpose()
#             rho_i_hits = rho_i_corners
#             # print(len(rho_i_hits))
#             # print(len(rho_i_corners))
#             if len(np.where(sasai_grid == roh_i)[0]) > 0:
#                 for i, j, k in rho_i_hits: # if 2 or more grid points are 0 next to a roh*i point, set to 1
#                     six_corners = [[i,j,k] + corner for corner in six_corner_shifts]
#                     rho_i_zeros = [[i,j,k] for corner in six_corners if sasai_grid[i,j,k] == 0]
#                     if len(rho_i_zeros) > 1:
#                         # print(rho_i_zeros)
#                         sasai_grid[i,j,k] = 0
#                         grid[i,j,k] = 1
#
#     return sasai_grid, grid, rho_i_corners
#
# @cython.boundscheck(False)
# def c_cal_lig_sasa_grids(np.ndarray[np.float64_t, ndim=2] crd,
#                         np.ndarray[np.float64_t, ndim=1] grid_x,
#                         np.ndarray[np.float64_t, ndim=1] grid_y,
#                         np.ndarray[np.float64_t, ndim=1] grid_z,
#                         np.ndarray[np.float64_t, ndim=1] origin_crd,
#                         np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
#                         np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
#                         np.ndarray[np.float64_t, ndim=1] spacing,
#                         np.ndarray[np.int64_t, ndim=2]   nearest_neighbor_shifts,
#                         np.ndarray[np.int64_t, ndim=1]   grid_counts,
#                         np.ndarray[np.float64_t, ndim=1] radii,
#                         list core_atoms,
#                         list surface_atoms,
#                         list core_ions,
#                         list surface_ions,
#                         float lig_core_scaling,
#                         float lig_surface_scaling,
#                         float lig_metal_scaling,
#                         float rho,
#                            ):
#
#     cdef:
#         int atom_ind, i, j, k
#         int natoms = crd.shape[0]
#         int i_max = grid_x.shape[0]
#         int j_max = grid_y.shape[0]
#         int k_max = grid_z.shape[0]
#         list nearest_neighbors, rho_i_zeros, changes_list
#         np.ndarray[np.float64_t, ndim=1] atom_coordinate
#         np.ndarray[np.float64_t, ndim=3] sasai_grid = np.zeros([i_max, j_max, k_max], dtype=float)
#         np.ndarray[np.float64_t, ndim=3] sasar_grid = np.zeros([i_max, j_max, k_max], dtype=float)
#
#
#     for atom_ind in core_atoms:
#         atom_coordinate = crd[atom_ind]
#         radius = radii[atom_ind] * lig_core_scaling
#         corners = c_corners_within_radius(atom_coordinate, radius, origin_crd, uper_most_corner_crd,
#                                           uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#         for i, j, k in corners:
#             sasai_grid[i, j, k] = rho
#
#     for atom_ind in core_ions:
#         atom_coordinate = crd[atom_ind]
#         radius = radii[atom_ind] * lig_metal_scaling
#         corners = c_corners_within_radius(atom_coordinate, radius, origin_crd, uper_most_corner_crd,
#                                                   uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#         for i, j, k in corners:
#             sasai_grid[i, j, k] = rho
#
#     for atom_ind in surface_atoms:
#         atom_coordinate = crd[atom_ind]
#         radius = radii[atom_ind] * lig_surface_scaling
#         corners = c_corners_within_radius(atom_coordinate, radius, origin_crd, uper_most_corner_crd,
#                                                   uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#         for i, j, k in corners:
#             if sasai_grid[i][j][k] == 0:
#                 sasar_grid[i][j][k] = 1.
#
#     for atom_ind in surface_ions:
#         atom_coordinate = crd[atom_ind]
#         radius = radii[atom_ind] * lig_metal_scaling
#         corners = c_corners_within_radius(atom_coordinate, radius, origin_crd, uper_most_corner_crd,
#                                                   uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
#         for i, j, k in corners:
#             if sasai_grid[i][j][k] == 0:
#                 sasar_grid[i][j][k] = 1.
#
#     rho_i_hits = np.array(np.where(sasai_grid == rho)).transpose()
#     # print(rho_i_hits[0])
#     changes_list = []
#     if len(rho_i_hits) > 0:
#         for i, j, k in rho_i_hits: # if 2 or more grid points are 0 next to a roh*i point, set to 1
#             nearest_neighbors = [[i,j,k] + corner for corner in nearest_neighbor_shifts]
#             rho_i_zeros = [[l,m,n] for [l,m,n] in nearest_neighbors if sasai_grid[l][m][n] == 0]
#             if len(rho_i_zeros) > 1:
#                 changes_list.append([i,j,k])
#     if len(changes_list) > 0:
#         for [i, j, k] in changes_list:
#             sasai_grid[i][j][k] = 0.
#             sasar_grid[i][j][k] = 1.
#
#     return sasai_grid, sasar_grid

# @cython.boundscheck(False)
# def c_asa_frame(      np.ndarray[np.float64_t, ndim=2] crd,
#                             np.ndarray[np.float64_t, ndim=1] atom_radii,
#                             np.ndarray[np.float64_t, ndim=2] sphere_points,
#                             np.ndarray[np.float64_t, ndim=1] spacing,
#                             int n_sphere_points,
#                             np.ndarray[np.float64_t, ndim=3] grid
#                             ):
#     print("c_asa_frame")
#     cdef:
#         int i, j, k, index
#         int l, m, n
#         int natoms = crd.shape[0]
#         int n_neighbor_indices, k_closest_neighbor, k_prime
#         float atom_radius_i, atom_radius_j
#         float radius_cutoff, radius_cutoff2, r2, r
#         float constant = 4.0 * np.pi / n_sphere_points
#         bint is_accessible
#         np.ndarray[np.int64_t, ndim=1] grid_crd = np.empty(3, dtype=np.int64)
#         np.ndarray[np.float64_t, ndim=1] sphere_point_coordinate
#         np.ndarray[np.float64_t, ndim=1] r_i, r_j, r_ij, r_jk
#         np.ndarray[np.int64_t, ndim = 1] neighbor_indices = np.empty(natoms, dtype=np.int64)
#         np.ndarray[np.float64_t, ndim = 3] centered_sphere_points = np.empty([natoms,n_sphere_points,3], dtype=np.float64)
#         np.ndarray[np.float64_t, ndim = 1] areas = np.zeros(natoms, dtype=np.float64)
#     print("debug test")
#     for i in range(natoms):
#         atom_radius_i = atom_radii[i]
#         r_i = crd[i]
#
#         # Get all the atoms close to atom i
#         n_neighbor_indices = 0
#         for j in range(natoms):
#             if i == j:
#                 continue
#
#             r_j = crd[j]
#             r_ij = r_i-r_j
#             atom_radius_j = atom_radii[j]
#
#             # look for atoms j around atom i
#             radius_cutoff = atom_radius_i+atom_radius_j
#             radius_cutoff2 = radius_cutoff*radius_cutoff
#             r2 = np.dot(r_ij, r_ij)
#
#             if r2 < radius_cutoff2:
#                 neighbor_indices[n_neighbor_indices] = j
#                 n_neighbor_indices += 1
#             if r2 < 1e-10:
#                 print("This code is known to fail when atoms are too close")
#                 break
#         # Center the sphere points on atom i
#         for j in range(n_sphere_points):
#             centered_sphere_points[i, j, 0] = crd[i, 0] + atom_radius_i*sphere_points[j, 0]
#             centered_sphere_points[i, j, 1] = crd[i, 1] + atom_radius_i*sphere_points[j, 1]
#             centered_sphere_points[i, j, 2] = crd[i, 2] + atom_radius_i*sphere_points[j, 2]
#
#         # Check if these points are accessible
#         k_closest_neighbor = 0
#         for j in range(n_sphere_points):
#             is_accessible = True
#             r_j = centered_sphere_points[i, j, :]
#             for k in range(n_neighbor_indices + k_closest_neighbor)[k_closest_neighbor:]:
#                 k_prime = k % n_neighbor_indices
#                 r = atom_radii[neighbor_indices[k_prime]]
#                 index = neighbor_indices[k_prime]
#                 r_jk = r_j-crd[index]
#                 if np.dot(r_jk,r_jk) < r*r:
#                     k_closest_neighbor = k
#                     is_accessible = False
#                     break;
#             if(is_accessible):
#                 grid_crd = c_crd_to_grid(cmround(r_j, spacing), spacing)
#                 l,m,n = grid_crd.astype(int)
#                 grid[l,m,n] += constant * (atom_radii[i]) * (atom_radii[i])
#                 areas[i] += 1
#         areas[i] *= constant * (atom_radii[i]) * (atom_radii[i])
#     return grid, areas
#
# @cython.boundscheck(False)
# def c_generate_sphere_points(int n_points):
#     cdef:
#         int i
#         float y, r, phi
#         float inc = np.pi * (3.0 - np.sqrt(5.0))
#         float offset = 2.0 / n_points
#         np.ndarray[np.float64_t, ndim = 2] sphere_points = np.empty([n_points, 3], dtype=np.float64)
#
#     for i in range(n_points):
#         y = i * offset - 1.0 + (offset / 2.0)
#         r = np.sqrt(1.0 - y*y)
#         phi = i * inc
#
#         sphere_points[i, 0] = np.cos(phi) * r
#         sphere_points[i, 1] = y
#         sphere_points[i, 2] = np.sin(phi) * r
#     return sphere_points
#
#
# @cython.boundscheck(False)
# def c_sasa(         np.ndarray[np.float64_t, ndim=2] crd,
#                     np.ndarray[np.float64_t, ndim=1] atom_radii,
#                     np.ndarray[np.float64_t, ndim=1] spacing,
#                     float probe_size,
#                     int n_sphere_points,
#                     np.ndarray[np.float64_t, ndim=3] grid):
#     print(crd.shape[0])
#     cdef:
#         int natoms = crd.shape[0]
#         int i, j
#         # np.ndarray[np.int64_t, ndim = 1] wb1 = np.empty(natoms, dtype=np.int64)
#         # np.ndarray[np.float64_t, ndim = 2] wb2 = np.empty([n_sphere_points, 3], dtype=np.float64)
#         # np.ndarray[np.float64_t, ndim = 1] outframe
#         # np.ndarray[np.float64_t, ndim = 1] outframebuffer = np.empty(natoms, dtype=np.float64)
#         np.ndarray[np.float64_t, ndim = 2] sphere_points = np.empty([n_sphere_points, 3], dtype=np.float64)
#     sphere_points = c_generate_sphere_points(n_sphere_points)
#     grid, areas = c_asa_frame(crd, atom_radii+probe_size, sphere_points, spacing, n_sphere_points, grid)
#
#     return grid, areas


