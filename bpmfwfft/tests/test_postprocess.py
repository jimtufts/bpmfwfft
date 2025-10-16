import pytest
import numpy as np
import tempfile
import netCDF4
from pathlib import Path

import bpmfwfft.postprocess as postprocess

mod_path = Path(__file__).parent


class TestBootstrapping:
    """Tests for bootstrapping() function"""

    def test_basic_bootstrapping(self):
        """Test basic bootstrapping functionality"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nrepetitions = 10

        result = postprocess.bootstrapping(data, nrepetitions)

        assert result.shape == (nrepetitions,)
        assert isinstance(result, np.ndarray)
        # Each sum should be close to the sum of original data on average
        assert result.mean() > 0

    def test_bootstrapping_reproducibility(self):
        """Test that bootstrapping with fixed seed is reproducible"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nrepetitions = 10

        np.random.seed(42)
        result1 = postprocess.bootstrapping(data, nrepetitions)

        np.random.seed(42)
        result2 = postprocess.bootstrapping(data, nrepetitions)

        assert np.array_equal(result1, result2)

    def test_bootstrapping_invalid_input(self):
        """Test that bootstrapping raises error for non-1D array"""
        data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(AssertionError):
            postprocess.bootstrapping(data_2d, 10)

    def test_bootstrapping_sum_range(self):
        """Test that bootstrap sums are in reasonable range"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nrepetitions = 100

        result = postprocess.bootstrapping(data, nrepetitions)

        # Min and max possible sums
        min_sum = data.min() * len(data)
        max_sum = data.max() * len(data)

        assert result.min() >= min_sum
        assert result.max() <= max_sum


class TestSelectRotationInd:
    """Tests for select_rotation_ind() function"""

    def test_input_validation(self):
        """Test that function validates 1D input"""
        data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(AssertionError):
            postprocess.select_rotation_ind(data_2d, 10)

    def test_basic_functionality(self):
        """Test basic function behavior"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nrepetitions = 10

        # Function currently doesn't return anything, just validates input
        result = postprocess.select_rotation_ind(data, nrepetitions)
        assert result is None


class TestPostProcessHelperMethods:
    """Tests for PostProcess helper methods that don't require full initialization"""

    def test_corresponding_gas_phase(self):
        """Test gas phase name generation"""
        # Create a minimal PostProcess instance just to test the method
        pp = postprocess.PostProcess.__new__(postprocess.PostProcess)

        assert pp._corresponding_gas_phase("OpenMM_OBC2") == "OpenMM_Gas"
        assert pp._corresponding_gas_phase("sander_PBSA") == "sander_Gas"
        assert pp._corresponding_gas_phase("OpenMM_GBSA") == "OpenMM_Gas"

    def test_get_gas_phases(self):
        """Test extraction of unique gas phases"""
        pp = postprocess.PostProcess.__new__(postprocess.PostProcess)

        # Test with multiple solvent phases from same gas phase
        pp._solvent_phases = ["OpenMM_OBC2", "OpenMM_GBSA"]
        gas_phases = pp._get_gas_phases()
        assert gas_phases == ["OpenMM_Gas"]

        # Test with multiple different gas phases
        pp._solvent_phases = ["OpenMM_OBC2", "sander_PBSA"]
        gas_phases = pp._get_gas_phases()
        assert set(gas_phases) == {"OpenMM_Gas", "sander_Gas"}


class TestPostProcessPL:
    """Tests for PostProcess_PL subclass"""

    def test_check_number_resampled_energy_override(self):
        """Test that PostProcess_PL overrides check method"""
        pp_pl = postprocess.PostProcess_PL.__new__(postprocess.PostProcess_PL)

        # Should return None without doing any checks
        result = pp_pl._check_number_resampled_energy(100)
        assert result is None


class TestOpenMMPlatformParameter:
    """Tests for OpenMM platform parameter"""

    def test_openmm_energy_with_cpu_platform(self):
        """Test that openmm_energy works with CPU platform"""
        try:
            from bpmfwfft.md_openmm import openmm_energy
        except ImportError:
            pytest.skip("OpenMM not available")

        prmtop_file = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.prmtop").resolve()
        inpcrd_file = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.inpcrd").resolve()

        # Load coordinates
        import bpmfwfft.IO as IO
        crd = IO.InpcrdLoad(str(inpcrd_file)).get_coordinates()

        # Test with CPU platform (should work without CUDA)
        try:
            energies = openmm_energy(str(prmtop_file), crd, "OpenMM_Gas", platform_name='CPU')
            assert isinstance(energies, np.ndarray)
            assert len(energies) == 1
            assert isinstance(energies[0], (float, np.floating))
        except Exception as e:
            pytest.fail(f"openmm_energy with CPU platform failed: {e}")

    def test_cal_pot_energy_passes_platform(self):
        """Test that cal_pot_energy correctly passes platform parameter"""
        try:
            from bpmfwfft.postprocess import cal_pot_energy
        except ImportError:
            pytest.skip("postprocess module not available")

        prmtop_file = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.prmtop").resolve()
        inpcrd_file = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.inpcrd").resolve()

        # Load coordinates
        import bpmfwfft.IO as IO
        crd = IO.InpcrdLoad(str(inpcrd_file)).get_coordinates()

        # Test with CPU platform
        try:
            energies = cal_pot_energy(str(prmtop_file), crd, "OpenMM_Gas", "/tmp", openmm_platform='CPU')
            assert isinstance(energies, np.ndarray)
            assert len(energies) == 1
        except Exception as e:
            pytest.fail(f"cal_pot_energy with CPU platform failed: {e}")


class TestPostProcessIntegration:
    """Integration tests for full PostProcess initialization and calculation"""

    @pytest.fixture
    def test_files(self):
        """Provide paths to test files"""
        return {
            'rec_prmtop': (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.prmtop").resolve(),
            'lig_prmtop': (mod_path / "../../examples/amber/ubql_ubiquitin/ligand.prmtop").resolve(),
            'complex_prmtop': (mod_path / "../../examples/amber/ubql_ubiquitin/complex.prmtop").resolve(),
            'sampling_nc': (mod_path / "../../examples/fft_sampling/ubql_ubiquitin/fft_sampling_test.nc").resolve()
        }

    def test_postprocess_initialization_openmm(self, test_files):
        """Test full PostProcess initialization with OpenMM phase"""
        # Check if test file exists
        if not test_files['sampling_nc'].exists():
            pytest.skip("Test sampling NC file not available")

        # Use a small subset for fast testing
        solvent_phases = ["OpenMM_OBC2"]
        nr_resampled_complexes = 5
        start = 0
        end = 2  # Only process first 2 rotations for speed
        randomly_translate_complex = False
        temperature = 300.0
        sander_tmp_dir = tempfile.mkdtemp()

        try:
            pp = postprocess.PostProcess(
                str(test_files['rec_prmtop']),
                str(test_files['lig_prmtop']),
                str(test_files['complex_prmtop']),
                str(test_files['sampling_nc']),
                solvent_phases,
                nr_resampled_complexes,
                start,
                end,
                randomly_translate_complex,
                temperature,
                sander_tmp_dir,
                openmm_platform='CPU'
            )

            # Verify PostProcess object was created
            assert hasattr(pp, '_bpmf')
            assert hasattr(pp, '_rec_desol_fe')
            assert hasattr(pp, '_lig_desol_fe')
            assert hasattr(pp, '_complex_sol_fe')

            # Verify phases are present
            assert 'OpenMM_Gas' in pp._bpmf
            assert 'OpenMM_OBC2' in pp._bpmf
            assert 'OpenMM_OBC2' in pp._rec_desol_fe
            assert 'OpenMM_OBC2' in pp._lig_desol_fe
            assert 'OpenMM_OBC2' in pp._complex_sol_fe

        finally:
            import shutil
            shutil.rmtree(sander_tmp_dir, ignore_errors=True)

    def test_postprocess_get_bpmf(self, test_files):
        """Test get_bpmf() method returns correct structure"""
        if not test_files['sampling_nc'].exists():
            pytest.skip("Test sampling NC file not available")

        solvent_phases = ["OpenMM_OBC2"]
        nr_resampled_complexes = 3
        start = 0
        end = 2
        sander_tmp_dir = tempfile.mkdtemp()

        try:
            pp = postprocess.PostProcess(
                str(test_files['rec_prmtop']),
                str(test_files['lig_prmtop']),
                str(test_files['complex_prmtop']),
                str(test_files['sampling_nc']),
                solvent_phases,
                nr_resampled_complexes,
                start,
                end,
                False,
                300.0,
                sander_tmp_dir,
                openmm_platform='CPU'
            )

            bpmf = pp.get_bpmf()

            # Should return a dictionary
            assert isinstance(bpmf, dict)
            assert 'OpenMM_Gas' in bpmf
            assert 'OpenMM_OBC2' in bpmf

            # Values should be floats (could be inf or nan for test data)
            assert isinstance(bpmf['OpenMM_Gas'], (float, np.floating))
            assert isinstance(bpmf['OpenMM_OBC2'], (float, np.floating))

        finally:
            import shutil
            shutil.rmtree(sander_tmp_dir, ignore_errors=True)

    def test_postprocess_pickle_bpmf(self, test_files):
        """Test pickle_bpmf() saves data correctly"""
        if not test_files['sampling_nc'].exists():
            pytest.skip("Test sampling NC file not available")

        solvent_phases = ["OpenMM_OBC2"]
        nr_resampled_complexes = 3
        start = 0
        end = 2
        sander_tmp_dir = tempfile.mkdtemp()

        try:
            pp = postprocess.PostProcess(
                str(test_files['rec_prmtop']),
                str(test_files['lig_prmtop']),
                str(test_files['complex_prmtop']),
                str(test_files['sampling_nc']),
                solvent_phases,
                nr_resampled_complexes,
                start,
                end,
                False,
                300.0,
                sander_tmp_dir,
                openmm_platform='CPU'
            )

            # Save to temporary pickle file
            import pickle
            pickle_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            pickle_file.close()

            pp.pickle_bpmf(pickle_file.name)

            # Load and verify structure
            with open(pickle_file.name, 'rb') as f:
                data = pickle.load(f)

            assert 'bpmf' in data
            assert 'mean_Psi' in data
            assert 'min_Psi' in data
            assert 'std' in data
            assert 'energies' in data

            # Verify sub-dictionaries
            assert isinstance(data['bpmf'], dict)
            assert isinstance(data['std'], dict)
            assert isinstance(data['energies'], dict)

            # Cleanup
            import os
            os.remove(pickle_file.name)

        finally:
            import shutil
            shutil.rmtree(sander_tmp_dir, ignore_errors=True)
