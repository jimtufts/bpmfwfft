# fft_correlation_wrapper.pyx
# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "fft_correlation_handler.h" namespace "bpmfwfft":
    cdef cppclass FFTCorrelationHandler:
        FFTCorrelationHandler() except +
        void initialize(int nx, int ny, int nz) except +
        void cleanup() except +
        void precomputeReceptorFFT(const float* receptor_grid) except +
        double computeCorrelationEnergy(const float* ligand_grid) except +
        void getCorrelationGrid(float* output_grid) except +


cdef class PyFFTCorrelationHandler:
    """
    Python wrapper for GPU FFT correlation handler

    This class provides a Python interface to the C++ FFTCorrelationHandler,
    which performs FFT-based molecular grid correlation on the GPU with
    automatic adaptive tiling for large grids.

    Example usage:
    >>> handler = PyFFTCorrelationHandler()
    >>> handler.initialize(nx, ny, nz)
    >>> handler.precompute_receptor_fft(receptor_grid)
    >>>
    >>> # For each ligand pose:
    >>> energy = handler.compute_correlation_energy(ligand_grid)
    >>>
    >>> handler.cleanup()
    """
    cdef FFTCorrelationHandler* handler
    cdef int nx, ny, nz
    cdef bint initialized

    def __cinit__(self):
        """Constructor - allocates the C++ handler"""
        self.handler = new FFTCorrelationHandler()
        self.initialized = False
        self.nx = 0
        self.ny = 0
        self.nz = 0

    def __dealloc__(self):
        """Destructor - frees the C++ handler"""
        if self.initialized:
            self.handler.cleanup()
        del self.handler

    def initialize(self, int nx, int ny, int nz):
        """
        Initialize the FFT correlation handler with grid dimensions

        Parameters
        ----------
        nx : int
            Grid size in x dimension
        ny : int
            Grid size in y dimension
        nz : int
            Grid size in z dimension

        Notes
        -----
        - Automatically configures adaptive tiling for large grids
        - Allocates GPU memory for FFT operations
        - Must be called before precompute_receptor_fft()
        """
        if self.initialized:
            self.handler.cleanup()

        self.handler.initialize(nx, ny, nz)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.initialized = True

    def precompute_receptor_fft(self, np.ndarray[np.float32_t, ndim=3, mode="c"] receptor_grid not None):
        """
        Precompute and store the receptor grid FFT

        Parameters
        ----------
        receptor_grid : np.ndarray[float32, ndim=3]
            Receptor grid with shape (nz, ny, nx) in C-order

        Notes
        -----
        - The receptor FFT is computed once and reused for all ligand poses
        - Grid must match dimensions specified in initialize()
        - Grid should be in (z, y, x) order (C-order)

        Raises
        ------
        ValueError
            If grid dimensions don't match initialized dimensions
        RuntimeError
            If CUDA/cuFFT errors occur
        """
        if not self.initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        # Verify dimensions (grid is in z, y, x order)
        if receptor_grid.shape[0] != self.nz or \
           receptor_grid.shape[1] != self.ny or \
           receptor_grid.shape[2] != self.nx:
            raise ValueError(
                "Grid shape ({}, {}, {}) doesn't match initialized "
                "dimensions ({}, {}, {})".format(
                    receptor_grid.shape[0], receptor_grid.shape[1], receptor_grid.shape[2],
                    self.nz, self.ny, self.nx
                )
            )

        self.handler.precomputeReceptorFFT(&receptor_grid[0, 0, 0])

    def compute_correlation_energy(self, np.ndarray[np.float32_t, ndim=3, mode="c"] ligand_grid not None):
        """
        Compute correlation energy between receptor and ligand grids

        Parameters
        ----------
        ligand_grid : np.ndarray[float32, ndim=3]
            Ligand grid with shape (nz, ny, nx) in C-order

        Returns
        -------
        float
            Correlation energy (sum of correlation grid values)

        Notes
        -----
        - Must call precompute_receptor_fft() before this method
        - Performs: IFFT(FFT(ligand) * conj(FFT_receptor))
        - Returns sum of all correlation grid values

        Raises
        ------
        ValueError
            If grid dimensions don't match
        RuntimeError
            If receptor FFT not precomputed or CUDA errors occur
        """
        if not self.initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        # Verify dimensions
        if ligand_grid.shape[0] != self.nz or \
           ligand_grid.shape[1] != self.ny or \
           ligand_grid.shape[2] != self.nx:
            raise ValueError(
                "Grid shape ({}, {}, {}) doesn't match initialized "
                "dimensions ({}, {}, {})".format(
                    ligand_grid.shape[0], ligand_grid.shape[1], ligand_grid.shape[2],
                    self.nz, self.ny, self.nx
                )
            )

        cdef double energy = self.handler.computeCorrelationEnergy(&ligand_grid[0, 0, 0])
        return energy

    def get_correlation_grid(self):
        """
        Get the full correlation grid from the last computation

        Returns
        -------
        np.ndarray[float32, ndim=3]
            Correlation grid with shape (nz, ny, nx)

        Notes
        -----
        - Must call compute_correlation_energy() before this method
        - Returns the full spatial correlation grid
        - Useful for finding optimal translation/rotation

        Raises
        ------
        RuntimeError
            If no correlation has been computed yet
        """
        if not self.initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        cdef np.ndarray[np.float32_t, ndim=3, mode="c"] output = np.zeros(
            (self.nz, self.ny, self.nx), dtype=np.float32
        )

        self.handler.getCorrelationGrid(&output[0, 0, 0])
        return output

    def cleanup(self):
        """
        Clean up GPU resources

        Notes
        -----
        - Automatically called by destructor
        - Safe to call multiple times
        - After cleanup, must call initialize() again to reuse
        """
        if self.initialized:
            self.handler.cleanup()
            self.initialized = False


def compute_fft_correlation_gpu(
    np.ndarray[np.float32_t, ndim=3, mode="c"] receptor_grid not None,
    np.ndarray[np.float32_t, ndim=3, mode="c"] ligand_grid not None
):
    """
    Compute FFT correlation energy between receptor and ligand grids (convenience function)

    This is a simple one-shot interface that handles initialization and cleanup
    automatically. For processing multiple ligand poses, use PyFFTCorrelationHandler
    directly to avoid re-initializing and re-computing the receptor FFT.

    Parameters
    ----------
    receptor_grid : np.ndarray[float32, ndim=3]
        Receptor grid with shape (nz, ny, nx) in C-order
    ligand_grid : np.ndarray[float32, ndim=3]
        Ligand grid with shape (nz, ny, nx) in C-order

    Returns
    -------
    float
        Correlation energy (sum of correlation grid values)

    Examples
    --------
    >>> energy = compute_fft_correlation_gpu(receptor_grid, ligand_grid)

    For multiple ligand poses (more efficient):
    >>> handler = PyFFTCorrelationHandler()
    >>> handler.initialize(nx, ny, nz)
    >>> handler.precompute_receptor_fft(receptor_grid)
    >>> for ligand_grid in ligand_grids:
    ...     energy = handler.compute_correlation_energy(ligand_grid)
    >>> handler.cleanup()
    """
    if receptor_grid.shape[0] != ligand_grid.shape[0] or \
       receptor_grid.shape[1] != ligand_grid.shape[1] or \
       receptor_grid.shape[2] != ligand_grid.shape[2]:
        raise ValueError(
            "Receptor and ligand grids must have same shape. "
            "Got receptor=({}, {}, {}), ligand=({}, {}, {})".format(
                receptor_grid.shape[0], receptor_grid.shape[1], receptor_grid.shape[2],
                ligand_grid.shape[0], ligand_grid.shape[1], ligand_grid.shape[2]
            )
        )

    cdef int nz = receptor_grid.shape[0]
    cdef int ny = receptor_grid.shape[1]
    cdef int nx = receptor_grid.shape[2]

    cdef PyFFTCorrelationHandler handler = PyFFTCorrelationHandler()
    handler.initialize(nx, ny, nz)
    handler.precompute_receptor_fft(receptor_grid)
    cdef double energy = handler.compute_correlation_energy(ligand_grid)
    handler.cleanup()

    return energy
