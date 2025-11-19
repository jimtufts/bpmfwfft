#ifndef FFT_PLAN_MANAGER_H
#define FFT_PLAN_MANAGER_H

#include <cufft.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace bpmfwfft {

/**
 * Manages cuFFT plans for 3D FFT operations
 * Handles plan creation, execution, and cleanup
 * Supports both in-place and out-of-place transforms
 */
class FFTPlanManager {
public:
    FFTPlanManager();
    ~FFTPlanManager();

    /**
     * Create plans for 3D R2C (real to complex) and C2R (complex to real) transforms
     *
     * @param nx, ny, nz: Grid dimensions
     * @param batch: Number of transforms to batch (default 1)
     * @param use_float64: Whether to use double precision (default false)
     */
    void createPlansR2C(
        int nx, int ny, int nz,
        int batch = 1,
        bool use_float64 = false
    );

    /**
     * Create plans for 3D C2C (complex to complex) transforms
     * More flexible than R2C/C2R, used for general FFT operations
     *
     * @param nx, ny, nz: Grid dimensions
     * @param batch: Number of transforms to batch (default 1)
     * @param use_float64: Whether to use double precision (default false)
     */
    void createPlansC2C(
        int nx, int ny, int nz,
        int batch = 1,
        bool use_float64 = false
    );

    /**
     * Destroy current plans
     */
    void destroyPlans();

    /**
     * Execute forward FFT (real to complex or complex to complex)
     *
     * For R2C: input is real array, output is complex (Hermitian symmetric)
     * For C2C: both input and output are complex
     *
     * @param input: Input data (real or complex depending on plan type)
     * @param output: Output data (complex)
     * @param stream: CUDA stream for async execution (default 0)
     */
    void executeForward(
        void* input,
        cufftComplex* output,
        cudaStream_t stream = 0
    );

    /**
     * Execute inverse FFT (complex to real or complex to complex)
     *
     * For C2R: input is complex (Hermitian), output is real
     * For C2C: both input and output are complex
     *
     * @param input: Input data (complex)
     * @param output: Output data (real or complex depending on plan type)
     * @param stream: CUDA stream for async execution (default 0)
     */
    void executeInverse(
        cufftComplex* input,
        void* output,
        cudaStream_t stream = 0
    );

    /**
     * Execute forward FFT in-place (complex to complex only)
     *
     * @param data: Input/output data (complex)
     * @param stream: CUDA stream for async execution (default 0)
     */
    void executeForwardInPlace(
        cufftComplex* data,
        cudaStream_t stream = 0
    );

    /**
     * Execute inverse FFT in-place (complex to complex only)
     *
     * @param data: Input/output data (complex)
     * @param stream: CUDA stream for async execution (default 0)
     */
    void executeInverseInPlace(
        cufftComplex* data,
        cudaStream_t stream = 0
    );

    /**
     * Get current grid dimensions
     */
    void getGridDimensions(int& nx, int& ny, int& nz) const {
        nx = nx_; ny = ny_; nz = nz_;
    }

    /**
     * Check if plans are created and valid
     */
    bool isInitialized() const {
        return plans_created_;
    }

    /**
     * Get normalization factor for inverse FFT
     * cuFFT doesn't normalize, so we need to divide by N after inverse FFT
     */
    float getNormalizationFactor() const {
        return 1.0f / (nx_ * ny_ * nz_);
    }

    /**
     * Estimate memory required for cuFFT workspace
     */
    static size_t estimateWorkspaceSize(
        int nx, int ny, int nz,
        bool use_float64 = false
    );

private:
    cufftHandle forward_plan_;
    cufftHandle inverse_plan_;

    int nx_, ny_, nz_;
    int batch_;
    bool use_float64_;
    bool use_r2c_;  // R2C/C2R vs C2C
    bool plans_created_;

    // Helper to check cuFFT errors
    void checkCufftError(cufftResult result, const char* operation);
};

} // namespace bpmfwfft

#endif // FFT_PLAN_MANAGER_H
