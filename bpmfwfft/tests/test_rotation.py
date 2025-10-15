import pytest
import numpy as np
import netCDF4
import bpmfwfft.rotation
from bpmfwfft.IO import InpcrdLoad
from pathlib import Path
import tempfile
import os

mod_path = Path(__file__).parent

# Use the same test system as test_grids.py
lig_inpcrd = (mod_path / "../../examples/amber/ubql_ubiquitin/ligand.inpcrd").resolve()


def test_rotation_matrix_identity():
    """Test that u=[0,0,0] gives identity-like rotation"""
    u = [0.0, 0.0, 0.0]
    R = bpmfwfft.rotation._rotation_matrix(u)

    # Check shape
    assert R.shape == (3, 3)

    # Check it's a valid rotation matrix (determinant = 1)
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 1e-10, f"Determinant should be 1, got {det}"

    # Check orthogonality: R @ R.T should be identity
    identity = np.dot(R, R.T)
    assert np.allclose(identity, np.eye(3), atol=1e-10)


def test_rotation_matrix_valid_range():
    """Test that u values outside [0,1] raise assertion"""
    # Test values outside valid range
    with pytest.raises(AssertionError):
        bpmfwfft.rotation._rotation_matrix([1.5, 0.5, 0.5])

    with pytest.raises(AssertionError):
        bpmfwfft.rotation._rotation_matrix([-0.1, 0.5, 0.5])


def test_rotation_matrix_properties():
    """Test mathematical properties of rotation matrices"""
    # Test several u values
    test_cases = [
        [0.5, 0.5, 0.5],
        [0.1, 0.2, 0.3],
        [0.9, 0.8, 0.7],
        [0.25, 0.75, 0.33]
    ]

    for u in test_cases:
        R = bpmfwfft.rotation._rotation_matrix(u)

        # Check determinant is 1 (proper rotation)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10, f"Determinant should be 1, got {det} for u={u}"

        # Check orthogonality
        identity = np.dot(R, R.T)
        assert np.allclose(identity, np.eye(3), atol=1e-10), f"Not orthogonal for u={u}"


def test_rotation_matrix_regression():
    """Test specific rotation matrix values for regression testing"""
    # Fixed u values should give consistent rotation matrix
    u = [0.3, 0.4, 0.5]
    R = bpmfwfft.rotation._rotation_matrix(u)

    # Reference values computed with the current implementation
    expected_R = np.array([
        [ 0.4       ,  0.53871408,  0.74147632],
        [-0.53871408, -0.5163119 ,  0.66573956],
        [ 0.74147632, -0.66573956,  0.0836881 ]
    ])

    # Check all elements match within tolerance
    assert np.allclose(R, expected_R, atol=1e-6), \
        f"Rotation matrix changed:\nExpected:\n{expected_R}\nGot:\n{R}"


def test_random_rotation_matrix_with_seed():
    """Test that random rotation matrix is reproducible with seed"""
    # Create seeded RNG
    rng1 = np.random.RandomState(42)
    R1 = bpmfwfft.rotation._random_rotation_matrix(rng1)

    # Reset seed
    rng2 = np.random.RandomState(42)
    R2 = bpmfwfft.rotation._random_rotation_matrix(rng2)

    # Should be identical
    assert np.allclose(R1, R2, atol=1e-10)

    # Check it's still a valid rotation matrix
    det = np.linalg.det(R1)
    assert abs(det - 1.0) < 1e-10


def test_random_rotation_matrix_properties():
    """Test that random rotation matrices have valid properties"""
    # Generate several random rotation matrices with fixed seed for reproducibility
    rng = np.random.RandomState(123)

    for _ in range(10):
        R = bpmfwfft.rotation._random_rotation_matrix(rng)

        # Check shape
        assert R.shape == (3, 3)

        # Check determinant is 1
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10

        # Check orthogonality
        identity = np.dot(R, R.T)
        assert np.allclose(identity, np.eye(3), atol=1e-10)


def test_rotate_molecule_identity_real_ligand():
    """Test that identity rotation doesn't change real ligand coordinates"""
    # Load real ligand coordinates
    crd = InpcrdLoad(lig_inpcrd).get_coordinates()

    # Identity rotation
    R = np.eye(3)

    # Rotate
    rotated = bpmfwfft.rotation._rotate_molecule(R, crd)

    # Should be unchanged
    assert np.allclose(rotated, crd, atol=1e-10)


def test_rotate_molecule_preserves_distances_real_ligand():
    """Test that rotation preserves inter-atomic distances on real ligand"""
    # Load real ligand coordinates
    crd = InpcrdLoad(lig_inpcrd).get_coordinates()

    # Calculate original distances for several atom pairs
    original_dist_01 = np.linalg.norm(crd[0] - crd[1])
    original_dist_05 = np.linalg.norm(crd[0] - crd[5])
    original_dist_10_20 = np.linalg.norm(crd[10] - crd[20])
    original_dist_50_100 = np.linalg.norm(crd[50] - crd[100])

    # Apply deterministic rotation (using specific u values)
    u = [0.5, 0.5, 0.5]
    R = bpmfwfft.rotation._rotation_matrix(u)
    rotated = bpmfwfft.rotation._rotate_molecule(R, crd)

    # Calculate rotated distances
    rotated_dist_01 = np.linalg.norm(rotated[0] - rotated[1])
    rotated_dist_05 = np.linalg.norm(rotated[0] - rotated[5])
    rotated_dist_10_20 = np.linalg.norm(rotated[10] - rotated[20])
    rotated_dist_50_100 = np.linalg.norm(rotated[50] - rotated[100])

    # Distances should be preserved
    assert abs(original_dist_01 - rotated_dist_01) < 1e-10
    assert abs(original_dist_05 - rotated_dist_05) < 1e-10
    assert abs(original_dist_10_20 - rotated_dist_10_20) < 1e-10
    assert abs(original_dist_50_100 - rotated_dist_50_100) < 1e-10


def test_move_molecule_to_origin_real_ligand():
    """Test moving real ligand to origin"""
    # Load real ligand coordinates
    crd = InpcrdLoad(lig_inpcrd).get_coordinates()

    # Calculate expected center
    expected_center = np.mean(crd, axis=0)

    # Move to origin
    moved_crd, displacement = bpmfwfft.rotation._move_molecule_to_origin(crd.copy())

    # Check displacement is negative of center
    assert np.allclose(displacement, -expected_center, atol=1e-10)

    # Check new center is at origin
    new_center = np.mean(moved_crd, axis=0)
    assert np.allclose(new_center, [0.0, 0.0, 0.0], atol=1e-10)


def test_move_molecule_to_original_position_real_ligand():
    """Test moving real ligand back to original position"""
    # Load real ligand coordinates
    original_crd = InpcrdLoad(lig_inpcrd).get_coordinates()

    # Move to origin
    moved_crd, displacement = bpmfwfft.rotation._move_molecule_to_origin(original_crd.copy())

    # Move back
    restored_crd = bpmfwfft.rotation._move_molecule_to_original_position(moved_crd.copy(), displacement)

    # Should match original
    assert np.allclose(restored_crd, original_crd, atol=1e-10)


def test_random_rotation_function_real_ligand():
    """Test the random_rotation function with real ligand"""
    # Load real ligand coordinates
    inpcrd = InpcrdLoad(lig_inpcrd).get_coordinates()

    # Apply random rotation with seed for reproducibility
    rotated = bpmfwfft.rotation.random_rotation(inpcrd.copy(), seed=42)

    # Check shape preserved
    assert rotated.shape == inpcrd.shape

    # Check distances preserved (rotation is rigid)
    original_dist = np.linalg.norm(inpcrd[0] - inpcrd[10])
    rotated_dist = np.linalg.norm(rotated[0] - rotated[10])
    assert abs(original_dist - rotated_dist) < 1e-10


def test_random_rotation_centered_at_origin_real_ligand():
    """Test that random_rotation centers real ligand at origin"""
    # Load real ligand coordinates
    inpcrd = InpcrdLoad(lig_inpcrd).get_coordinates()

    # Apply random rotation with seed for reproducibility
    rotated = bpmfwfft.rotation.random_rotation(inpcrd.copy(), seed=42)

    # Check that result is centered at origin
    center = np.mean(rotated, axis=0)
    assert np.allclose(center, [0.0, 0.0, 0.0], atol=1e-10)


def test_random_gen_rotation_with_seed():
    """Test that random_gen_rotation produces consistent results with seed"""
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        output_nc = tmp.name

    try:
        # Generate rotations with seed
        bpmfwfft.rotation.random_gen_rotation(str(lig_inpcrd), 10, output_nc, seed=42)

        # Read the generated file
        nc = netCDF4.Dataset(output_nc, 'r')
        positions1 = nc.variables['positions'][:]
        nc.close()

        # Delete and regenerate with same seed
        os.remove(output_nc)
        bpmfwfft.rotation.random_gen_rotation(str(lig_inpcrd), 10, output_nc, seed=42)

        # Read again
        nc = netCDF4.Dataset(output_nc, 'r')
        positions2 = nc.variables['positions'][:]
        nc.close()

        # Should be identical
        assert np.allclose(positions1, positions2, atol=1e-6)

    finally:
        # Cleanup
        if os.path.exists(output_nc):
            os.remove(output_nc)


def test_random_gen_rotation_preserves_distances():
    """Test that random_gen_rotation preserves inter-atomic distances"""
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        output_nc = tmp.name

    try:
        # Generate a few rotations
        bpmfwfft.rotation.random_gen_rotation(str(lig_inpcrd), 5, output_nc, seed=42)

        # Read the generated file
        nc = netCDF4.Dataset(output_nc, 'r')
        positions = nc.variables['positions'][:]
        nc.close()

        # Calculate distance between atoms 0 and 10 in original (first frame)
        original_dist = np.linalg.norm(positions[0, 0, :] - positions[0, 10, :])

        # Check distance is preserved in all rotations
        for i in range(1, positions.shape[0]):
            rotated_dist = np.linalg.norm(positions[i, 0, :] - positions[i, 10, :])
            assert abs(original_dist - rotated_dist) < 1e-5, \
                f"Distance not preserved in rotation {i}"

    finally:
        # Cleanup
        if os.path.exists(output_nc):
            os.remove(output_nc)


def test_systematic_gen_rotation():
    """Test that systematic_gen_rotation produces deterministic results"""
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        output_nc = tmp.name

    try:
        # Generate systematic rotations (8 total = 2^3)
        bpmfwfft.rotation.systematic_gen_rotation(str(lig_inpcrd), 8, output_nc)

        # Read the generated file
        nc = netCDF4.Dataset(output_nc, 'r')
        positions = nc.variables['positions'][:]
        n_rotations = positions.shape[0]
        nc.close()

        # Should generate 8 rotations (2^3)
        assert n_rotations == 8

        # All rotations should preserve distances
        original_dist = np.linalg.norm(positions[0, 0, :] - positions[0, 10, :])
        for i in range(1, n_rotations):
            rotated_dist = np.linalg.norm(positions[i, 0, :] - positions[i, 10, :])
            assert abs(original_dist - rotated_dist) < 1e-5

    finally:
        # Cleanup
        if os.path.exists(output_nc):
            os.remove(output_nc)


def test_rotation_matrix_invalid_input():
    """Test that invalid inputs raise appropriate errors"""
    # Test value > 1
    with pytest.raises(AssertionError):
        bpmfwfft.rotation._rotation_matrix([2.0, 0.5, 0.5])

    # Test negative values
    with pytest.raises(AssertionError):
        bpmfwfft.rotation._rotation_matrix([-0.1, 0.5, 0.5])


def test_rotate_molecule_invalid_shapes():
    """Test that invalid shapes raise appropriate errors"""
    # Load real ligand
    crd = InpcrdLoad(lig_inpcrd).get_coordinates()

    # Wrong rotation matrix shape
    R = np.eye(2)  # 2x2 instead of 3x3

    with pytest.raises(AssertionError):
        bpmfwfft.rotation._rotate_molecule(R, crd)

    # Wrong coordinate shape
    R = np.eye(3)
    crd_wrong = crd[:, :2]  # Only 2 columns

    with pytest.raises(AssertionError):
        bpmfwfft.rotation._rotate_molecule(R, crd_wrong)
