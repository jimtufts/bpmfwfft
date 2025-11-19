#!/usr/bin/env python
"""
Comprehensive tests comparing C++ potential_grid implementation
against original Cython implementation.
"""

import pytest
import numpy as np
import time
from bpmfwfft.util import c_cal_potential_grid_pp  # Original Cython
from bpmfwfft.potential_grid_wrapper import py_cal_potential_grid  # New C++


class TestPotentialGridAccuracy:
    """Test numerical accuracy of C++ implementation vs Cython"""

    @pytest.fixture
    def simple_system(self):
        """Simple test system with a few atoms"""
        # Use positions slightly offset from grid points to avoid division by zero in Cython
        crd = np.array([
            [0.1, 0.1, 0.1],
            [3.1, 0.1, 0.1],
            [0.1, 3.1, 0.1],
        ], dtype=np.float64)

        charges = np.array([1.0, -0.5, 0.5], dtype=np.float64)
        lj_sigma = np.array([2.0, 1.8, 1.6], dtype=np.float64)
        vdw_radii = np.array([1.5, 1.4, 1.3], dtype=np.float64)
        clash_radii = np.array([1.2, 1.1, 1.0], dtype=np.float64)
        molecule_sasa = np.array([[0.5, 0.3, 0.4]], dtype=np.float32)  # 2D array, float32 for Cython
        molecule_sasa_1d = np.array([0.5, 0.3, 0.4], dtype=np.float64)  # 1D array, float64 for C++
        atom_list = [0, 1, 2]
        bond_list = []

        return {
            'crd': crd,
            'charges': charges,
            'lj_sigma': lj_sigma,
            'vdw_radii': vdw_radii,
            'clash_radii': clash_radii,
            'molecule_sasa': molecule_sasa,  # For Cython (2D, float32)
            'molecule_sasa_1d': molecule_sasa_1d,  # For C++ (1D, float64)
            'atom_list': atom_list,
            'bond_list': bond_list,
        }

    @pytest.fixture
    def grid_params(self):
        """Grid parameters"""
        origin_crd = np.array([-5.0, -5.0, -5.0], dtype=np.float64)
        spacing = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        grid_counts = np.array([21, 21, 21], dtype=np.int64)

        grid_x = np.linspace(
            origin_crd[0],
            origin_crd[0] + ((grid_counts[0] - 1) * spacing[0]),
            num=grid_counts[0]
        )
        grid_y = np.linspace(
            origin_crd[1],
            origin_crd[1] + ((grid_counts[1] - 1) * spacing[1]),
            num=grid_counts[1]
        )
        grid_z = np.linspace(
            origin_crd[2],
            origin_crd[2] + ((grid_counts[2] - 1) * spacing[2]),
            num=grid_counts[2]
        )

        upper_most_corner_crd = origin_crd + (grid_counts - 1.) * spacing
        upper_most_corner = (grid_counts - 1).astype(np.int64)

        return {
            'origin_crd': origin_crd,
            'spacing': spacing,
            'grid_counts': grid_counts,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'upper_most_corner_crd': upper_most_corner_crd,
            'upper_most_corner': upper_most_corner,
        }

    def compare_grids(self, grid_cython, grid_cpp, name, tolerance=1e-10):
        """Compare two grids and print detailed statistics"""
        print(f"\n{'='*60}")
        print(f"Comparing {name} grids")
        print(f"{'='*60}")

        # Check shapes
        assert grid_cython.shape == grid_cpp.shape, \
            f"Shape mismatch: Cython {grid_cython.shape} vs C++ {grid_cpp.shape}"
        print(f"✓ Shapes match: {grid_cython.shape}")

        # Statistics
        print(f"\nCython implementation:")
        print(f"  Min:  {np.min(grid_cython):12.6e}")
        print(f"  Max:  {np.max(grid_cython):12.6e}")
        print(f"  Mean: {np.mean(grid_cython):12.6e}")
        print(f"  Std:  {np.std(grid_cython):12.6e}")

        print(f"\nC++ implementation:")
        print(f"  Min:  {np.min(grid_cpp):12.6e}")
        print(f"  Max:  {np.max(grid_cpp):12.6e}")
        print(f"  Mean: {np.mean(grid_cpp):12.6e}")
        print(f"  Std:  {np.std(grid_cpp):12.6e}")

        # Difference statistics
        diff = np.abs(grid_cython - grid_cpp)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Relative error (where values are non-zero)
        nonzero_mask = np.abs(grid_cython) > 1e-15
        if np.any(nonzero_mask):
            rel_error = diff[nonzero_mask] / (np.abs(grid_cython[nonzero_mask]) + 1e-15)
            max_rel_error = np.max(rel_error)
            mean_rel_error = np.mean(rel_error)
        else:
            max_rel_error = 0.0
            mean_rel_error = 0.0

        print(f"\nDifferences:")
        print(f"  Max absolute diff: {max_diff:12.6e}")
        print(f"  Mean absolute diff: {mean_diff:12.6e}")
        print(f"  Max relative error: {max_rel_error:12.6e}")
        print(f"  Mean relative error: {mean_rel_error:12.6e}")

        # Find locations of largest differences
        if max_diff > 0:
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"\nLargest difference at index {max_diff_idx}:")
            print(f"  Cython: {grid_cython[max_diff_idx]:12.6e}")
            print(f"  C++:    {grid_cpp[max_diff_idx]:12.6e}")

        # Check tolerance
        assert max_diff < tolerance, \
            f"Max difference {max_diff} exceeds tolerance {tolerance}"
        print(f"\n✓ All differences within tolerance {tolerance}")

        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'max_rel_error': max_rel_error,
            'mean_rel_error': mean_rel_error,
        }

    def test_electrostatic_grid(self, simple_system, grid_params):
        """Test electrostatic grid calculation"""
        # Cython version
        grid_cython = c_cal_potential_grid_pp(
            "electrostatic",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['bond_list'],
            simple_system['atom_list'],
            simple_system['molecule_sasa'],  # 2D, float32 for Cython
        )

        # C++ version
        grid_cpp = py_cal_potential_grid(
            "electrostatic",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['molecule_sasa_1d'],  # 1D, float64 for C++
            simple_system['atom_list'],
        )

        self.compare_grids(grid_cython, grid_cpp, "electrostatic", tolerance=1e-10)

    def test_ljr_grid(self, simple_system, grid_params):
        """Test LJr (Lennard-Jones repulsive) grid calculation"""
        grid_cython = c_cal_potential_grid_pp(
            "LJr",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['bond_list'],
            simple_system['atom_list'],
            simple_system['molecule_sasa'],  # 2D, float32 for Cython
        )

        grid_cpp = py_cal_potential_grid(
            "LJr",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['molecule_sasa_1d'],  # 1D, float64 for C++
            simple_system['atom_list'],
        )

        self.compare_grids(grid_cython, grid_cpp, "LJr", tolerance=1e-10)

    def test_lja_grid(self, simple_system, grid_params):
        """Test LJa (Lennard-Jones attractive) grid calculation"""
        grid_cython = c_cal_potential_grid_pp(
            "LJa",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['bond_list'],
            simple_system['atom_list'],
            simple_system['molecule_sasa'],  # 2D, float32 for Cython
        )

        grid_cpp = py_cal_potential_grid(
            "LJa",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['molecule_sasa_1d'],  # 1D, float64 for C++
            simple_system['atom_list'],
        )

        self.compare_grids(grid_cython, grid_cpp, "LJa", tolerance=1e-10)

    def test_water_grid(self, simple_system, grid_params):
        """Test water grid calculation"""
        grid_cython = c_cal_potential_grid_pp(
            "water",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['bond_list'],
            simple_system['atom_list'],
            simple_system['molecule_sasa'],  # 2D, float32 for Cython
        )

        grid_cpp = py_cal_potential_grid(
            "water",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['molecule_sasa_1d'],  # 1D, float64 for C++
            simple_system['atom_list'],
        )

        self.compare_grids(grid_cython, grid_cpp, "water", tolerance=1e-10)

    def test_occupancy_grid(self, simple_system, grid_params):
        """Test occupancy grid calculation"""
        grid_cython = c_cal_potential_grid_pp(
            "occupancy",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['bond_list'],
            simple_system['atom_list'],
            simple_system['molecule_sasa'],  # 2D, float32 for Cython
        )

        grid_cpp = py_cal_potential_grid(
            "occupancy",
            simple_system['crd'],
            grid_params['grid_x'],
            grid_params['grid_y'],
            grid_params['grid_z'],
            grid_params['origin_crd'],
            grid_params['upper_most_corner_crd'],
            grid_params['upper_most_corner'],
            grid_params['spacing'],
            grid_params['grid_counts'],
            simple_system['charges'],
            simple_system['lj_sigma'],
            simple_system['vdw_radii'],
            simple_system['clash_radii'],
            simple_system['molecule_sasa_1d'],  # 1D, float64 for C++
            simple_system['atom_list'],
        )

        self.compare_grids(grid_cython, grid_cpp, "occupancy", tolerance=1e-10)


class TestPotentialGridPerformance:
    """Performance benchmarks comparing C++ vs Cython"""

    @pytest.fixture
    def large_system(self):
        """Larger test system for performance testing"""
        np.random.seed(42)
        n_atoms = 100

        crd = np.random.randn(n_atoms, 3) * 10.0
        charges = np.random.randn(n_atoms) * 0.5
        lj_sigma = np.random.uniform(1.5, 2.5, n_atoms)
        vdw_radii = np.random.uniform(1.0, 2.0, n_atoms)
        clash_radii = np.random.uniform(0.8, 1.5, n_atoms)
        molecule_sasa_1d = np.random.uniform(0.0, 1.0, n_atoms)
        molecule_sasa = np.array([molecule_sasa_1d], dtype=np.float32)  # 2D, float32 for Cython
        atom_list = list(range(n_atoms))
        bond_list = []

        return {
            'crd': crd,
            'charges': charges,
            'lj_sigma': lj_sigma,
            'vdw_radii': vdw_radii,
            'clash_radii': clash_radii,
            'molecule_sasa': molecule_sasa,  # For Cython (2D, float32)
            'molecule_sasa_1d': molecule_sasa_1d,  # For C++ (1D, float64)
            'atom_list': atom_list,
            'bond_list': bond_list,
        }

    @pytest.fixture
    def large_grid_params(self):
        """Larger grid for performance testing"""
        origin_crd = np.array([-15.0, -15.0, -15.0], dtype=np.float64)
        spacing = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        grid_counts = np.array([61, 61, 61], dtype=np.int64)

        grid_x = np.linspace(
            origin_crd[0],
            origin_crd[0] + ((grid_counts[0] - 1) * spacing[0]),
            num=grid_counts[0]
        )
        grid_y = np.linspace(
            origin_crd[1],
            origin_crd[1] + ((grid_counts[1] - 1) * spacing[1]),
            num=grid_counts[1]
        )
        grid_z = np.linspace(
            origin_crd[2],
            origin_crd[2] + ((grid_counts[2] - 1) * spacing[2]),
            num=grid_counts[2]
        )

        upper_most_corner_crd = origin_crd + (grid_counts - 1.) * spacing
        upper_most_corner = (grid_counts - 1).astype(np.int64)

        return {
            'origin_crd': origin_crd,
            'spacing': spacing,
            'grid_counts': grid_counts,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'upper_most_corner_crd': upper_most_corner_crd,
            'upper_most_corner': upper_most_corner,
        }

    def benchmark_implementation(self, name, impl_name, calc_func, n_runs=3):
        """Run benchmark and return average time"""
        times = []
        for _ in range(n_runs):
            start = time.time()
            calc_func()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\n{impl_name:15s}: {avg_time:.4f} ± {std_time:.4f} s")
        return avg_time

    @pytest.mark.parametrize("grid_type", ["electrostatic", "LJr", "LJa"])
    def test_performance_comparison(self, grid_type, large_system, large_grid_params):
        """Compare performance of C++ vs Cython for different grid types"""
        print(f"\n{'='*60}")
        print(f"Performance benchmark: {grid_type} grid")
        print(f"System: {len(large_system['crd'])} atoms, "
              f"Grid: {tuple(large_grid_params['grid_counts'])}")
        print(f"{'='*60}")

        # Cython version
        def calc_cython():
            return c_cal_potential_grid_pp(
                grid_type,
                large_system['crd'],
                large_grid_params['grid_x'],
                large_grid_params['grid_y'],
                large_grid_params['grid_z'],
                large_grid_params['origin_crd'],
                large_grid_params['upper_most_corner_crd'],
                large_grid_params['upper_most_corner'],
                large_grid_params['spacing'],
                large_grid_params['grid_counts'],
                large_system['charges'],
                large_system['lj_sigma'],
                large_system['vdw_radii'],
                large_system['clash_radii'],
                large_system['bond_list'],
                large_system['atom_list'],
                large_system['molecule_sasa'],  # 2D, float32 for Cython
            )

        # C++ version
        def calc_cpp():
            return py_cal_potential_grid(
                grid_type,
                large_system['crd'],
                large_grid_params['grid_x'],
                large_grid_params['grid_y'],
                large_grid_params['grid_z'],
                large_grid_params['origin_crd'],
                large_grid_params['upper_most_corner_crd'],
                large_grid_params['upper_most_corner'],
                large_grid_params['spacing'],
                large_grid_params['grid_counts'],
                large_system['charges'],
                large_system['lj_sigma'],
                large_system['vdw_radii'],
                large_system['clash_radii'],
                large_system['molecule_sasa_1d'],  # 1D, float64 for C++
                large_system['atom_list'],
            )

        time_cython = self.benchmark_implementation(grid_type, "Cython", calc_cython)
        time_cpp = self.benchmark_implementation(grid_type, "C++ (OpenMP)", calc_cpp)

        speedup = time_cython / time_cpp
        print(f"\nSpeedup: {speedup:.2f}x")

        if speedup > 1.0:
            print(f"✓ C++ is {speedup:.2f}x faster than Cython")
        else:
            print(f"⚠ C++ is {1/speedup:.2f}x slower than Cython")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
