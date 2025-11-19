"""
GPU-accelerated grid correlation methods for molecular docking

This module provides GPU FFT correlation support for the grids module.
It extends the existing LigGrid class with GPU acceleration capabilities.
"""

import numpy as np

try:
    from bpmfwfft.fft_correlation_wrapper import PyFFTCorrelationHandler
    GPU_FFT_AVAILABLE = True
except ImportError:
    GPU_FFT_AVAILABLE = False
    print("Warning: GPU FFT correlation not available. Install CUDA support to enable.")


class GPUFFTCorrelationMixin:
    """
    Mixin class to add GPU FFT correlation support to LigGrid

    This provides drop-in replacement for CPU FFT correlation using GPU acceleration.
    """

    def _init_gpu_fft_correlation(self, rec_FFTs_dict):
        """
        Initialize GPU FFT correlation handlers for each grid type

        Parameters
        ----------
        rec_FFTs_dict : dict
            Dictionary of precomputed receptor FFTs for each grid type
        """
        if not GPU_FFT_AVAILABLE:
            raise RuntimeError("GPU FFT correlation not available. Check CUDA installation.")

        # Get grid dimensions
        counts = self._grid["counts"]
        nx, ny, nz = int(counts[2]), int(counts[1]), int(counts[0])

        print(f"Initializing GPU FFT correlation for {nx}×{ny}×{nz} grid...")

        # Create handlers for each grid type that will use GPU
        self._gpu_fft_handlers = {}
        self._gpu_grid_types = ["electrostatic", "LJr", "LJa"]  # Grid types to accelerate

        for grid_name in self._gpu_grid_types:
            if grid_name not in rec_FFTs_dict:
                continue

            try:
                # Create handler for this grid type
                handler = PyFFTCorrelationHandler()
                handler.initialize(nx, ny, nz)

                # Convert receptor FFT from complex128 to complex64 if needed
                rec_fft = rec_FFTs_dict[grid_name]

                # The receptor FFT is in frequency domain - we need the spatial grid
                # We'll precompute it from the spatial receptor grid instead
                # For now, store the handler and we'll precompute in the actual workflow
                self._gpu_fft_handlers[grid_name] = handler

                print(f"  ✓ GPU FFT handler created for {grid_name}")

            except Exception as e:
                print(f"  ✗ Failed to create GPU FFT handler for {grid_name}: {e}")
                # Fall back to CPU for this grid type
                if grid_name in self._gpu_fft_handlers:
                    del self._gpu_fft_handlers[grid_name]

        if not self._gpu_fft_handlers:
            print("Warning: No GPU FFT handlers initialized, falling back to CPU")
            self._use_gpu_fft = False
        else:
            print(f"GPU FFT correlation initialized for {len(self._gpu_fft_handlers)} grid types")
            self._use_gpu_fft = True

    def _precompute_receptor_ffts_gpu(self, receptor_grids_dict):
        """
        Precompute receptor FFTs on GPU

        Parameters
        ----------
        receptor_grids_dict : dict
            Dictionary of receptor grids (spatial domain) for each grid type
        """
        if not hasattr(self, '_gpu_fft_handlers') or not self._gpu_fft_handlers:
            return

        print("Precomputing receptor FFTs on GPU...")

        for grid_name, handler in self._gpu_fft_handlers.items():
            if grid_name not in receptor_grids_dict:
                print(f"  ✗ Receptor grid for {grid_name} not found")
                continue

            try:
                rec_grid = receptor_grids_dict[grid_name]

                # Ensure float32 format
                if rec_grid.dtype != np.float32:
                    rec_grid = rec_grid.astype(np.float32)

                # Ensure C-contiguous layout
                if not rec_grid.flags['C_CONTIGUOUS']:
                    rec_grid = np.ascontiguousarray(rec_grid)

                handler.precompute_receptor_fft(rec_grid)
                print(f"  ✓ Precomputed receptor FFT for {grid_name}")

            except Exception as e:
                print(f"  ✗ Failed to precompute receptor FFT for {grid_name}: {e}")
                # Remove this handler to fall back to CPU
                if grid_name in self._gpu_fft_handlers:
                    del self._gpu_fft_handlers[grid_name]

    def _cal_corr_func_gpu(self, grid_name):
        """
        GPU-accelerated version of _cal_corr_func

        Parameters
        ----------
        grid_name : str
            Name of grid type (electrostatic, LJr, LJa)

        Returns
        -------
        np.ndarray
            Correlation function grid
        """
        # Check if GPU is available for this grid type
        if not hasattr(self, '_use_gpu_fft') or not self._use_gpu_fft:
            return self._cal_corr_func_cpu(grid_name)

        if grid_name not in self._gpu_fft_handlers:
            return self._cal_corr_func_cpu(grid_name)

        handler = self._gpu_fft_handlers[grid_name]

        try:
            # Generate ligand charge grid
            lig_grid = self._cal_charge_grid(grid_name)

            # Ensure float32 format
            if lig_grid.dtype != np.float32:
                lig_grid = lig_grid.astype(np.float32)

            # Ensure C-contiguous layout
            if not lig_grid.flags['C_CONTIGUOUS']:
                lig_grid = np.ascontiguousarray(lig_grid)

            # Compute correlation on GPU and get full grid
            handler.compute_correlation_energy(lig_grid)
            corr_func = handler.get_correlation_grid()

            return corr_func

        except Exception as e:
            print(f"GPU correlation failed for {grid_name}: {e}")
            print(f"Falling back to CPU for this pose")
            return self._cal_corr_func_cpu(grid_name)

    def _cal_corr_func_cpu(self, grid_name):
        """
        CPU version of correlation (original implementation)

        This is the original _cal_corr_func method preserved for fallback.
        """
        import pyfftw.interfaces.numpy_fft as fftw

        assert grid_name in self._grid_func_names, "%s is not an allowed grid name" % grid_name
        corr_func = self._cal_charge_grid(grid_name)
        self._set_grid_key_value(grid_name, corr_func)
        corr_func = fftw.fftn(self._grid[grid_name])
        self._set_grid_key_value(grid_name, None)  # to save memory

        corr_func = corr_func.conjugate()
        corr_func = fftw.ifftn(self._rec_FFTs[grid_name] * corr_func)
        corr_func = np.real(corr_func)
        return corr_func

    def cleanup_gpu_fft(self):
        """Clean up GPU resources"""
        if hasattr(self, '_gpu_fft_handlers'):
            for handler in self._gpu_fft_handlers.values():
                handler.cleanup()
            self._gpu_fft_handlers = {}
        self._use_gpu_fft = False


def enable_gpu_fft_for_liggrid(lig_grid_instance, receptor_grids_dict):
    """
    Enable GPU FFT correlation for an existing LigGrid instance

    This function patches an existing LigGrid instance to use GPU acceleration.

    Parameters
    ----------
    lig_grid_instance : LigGrid
        The LigGrid instance to patch
    receptor_grids_dict : dict
        Dictionary of spatial receptor grids for each grid type
        Keys: grid type names (electrostatic, LJr, LJa, etc.)
        Values: np.ndarray with shape (nz, ny, nx)

    Examples
    --------
    >>> from bpmfwfft.grids import LigGrid, RecGrid
    >>> from bpmfwfft.grids_gpu import enable_gpu_fft_for_liggrid
    >>>
    >>> # Create grids normally
    >>> rec_grid = RecGrid(...)
    >>> lig_grid = LigGrid(..., receptor_grid=rec_grid)
    >>>
    >>> # Enable GPU acceleration
    >>> receptor_grids = rec_grid.get_grids()  # Get spatial grids
    >>> enable_gpu_fft_for_liggrid(lig_grid, receptor_grids)
    >>>
    >>> # Now use lig_grid normally - GPU acceleration is automatic
    >>> energy = lig_grid._cal_corr_func("electrostatic")
    """
    if not GPU_FFT_AVAILABLE:
        print("GPU FFT not available, continuing with CPU")
        return False

    # Add mixin methods to instance
    import types
    for attr_name in dir(GPUFFTCorrelationMixin):
        if not attr_name.startswith('_'):
            continue
        attr = getattr(GPUFFTCorrelationMixin, attr_name)
        if callable(attr):
            setattr(lig_grid_instance, attr_name, types.MethodType(attr, lig_grid_instance))

    # Initialize GPU FFT
    try:
        rec_FFTs = lig_grid_instance._rec_FFTs
        lig_grid_instance._init_gpu_fft_correlation(rec_FFTs)
        lig_grid_instance._precompute_receptor_ffts_gpu(receptor_grids_dict)

        # Monkey-patch the _cal_corr_func method
        original_cal_corr_func = lig_grid_instance._cal_corr_func
        lig_grid_instance._cal_corr_func_cpu = types.MethodType(
            GPUFFTCorrelationMixin._cal_corr_func_cpu,
            lig_grid_instance
        )
        lig_grid_instance._cal_corr_func = types.MethodType(
            GPUFFTCorrelationMixin._cal_corr_func_gpu,
            lig_grid_instance
        )

        print("✓ GPU FFT correlation enabled for LigGrid")
        return True

    except Exception as e:
        print(f"Failed to enable GPU FFT: {e}")
        import traceback
        traceback.print_exc()
        return False
