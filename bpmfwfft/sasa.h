#ifndef SASA_H_
#define SASA_H_
#ifdef __cplusplus
extern "C" {
#endif

void sasa(const int n_frames, const int n_atoms, const float* xyzlist,
          const float* atom_radii, const int n_sphere_points,
          const int* atom_selection_mask, float* out,
          const int* counts, const float grid_spacing);


#ifdef __cplusplus
}
#endif
#endif
