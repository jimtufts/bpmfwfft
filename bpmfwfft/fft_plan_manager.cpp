#include "fft_plan_manager.h"
#include <stdexcept>
#include <iostream>
#include <sstream>

namespace bpmfwfft {

FFTPlanManager::FFTPlanManager()
    : forward_plan_(0)
    , inverse_plan_(0)
    , nx_(0)
    , ny_(0)
    , nz_(0)
    , batch_(1)
    , use_float64_(false)
    , use_r2c_(false)
    , plans_created_(false)
{
}

FFTPlanManager::~FFTPlanManager() {
    destroyPlans();
}

void FFTPlanManager::checkCufftError(cufftResult result, const char* operation) {
    if (result != CUFFT_SUCCESS) {
        std::stringstream ss;
        ss << "cuFFT error during " << operation << ": ";

        switch (result) {
            case CUFFT_INVALID_PLAN:
                ss << "CUFFT_INVALID_PLAN";
                break;
            case CUFFT_ALLOC_FAILED:
                ss << "CUFFT_ALLOC_FAILED";
                break;
            case CUFFT_INVALID_TYPE:
                ss << "CUFFT_INVALID_TYPE";
                break;
            case CUFFT_INVALID_VALUE:
                ss << "CUFFT_INVALID_VALUE";
                break;
            case CUFFT_INTERNAL_ERROR:
                ss << "CUFFT_INTERNAL_ERROR";
                break;
            case CUFFT_EXEC_FAILED:
                ss << "CUFFT_EXEC_FAILED";
                break;
            case CUFFT_SETUP_FAILED:
                ss << "CUFFT_SETUP_FAILED";
                break;
            case CUFFT_INVALID_SIZE:
                ss << "CUFFT_INVALID_SIZE";
                break;
            case CUFFT_UNALIGNED_DATA:
                ss << "CUFFT_UNALIGNED_DATA";
                break;
            case CUFFT_INCOMPLETE_PARAMETER_LIST:
                ss << "CUFFT_INCOMPLETE_PARAMETER_LIST";
                break;
            case CUFFT_INVALID_DEVICE:
                ss << "CUFFT_INVALID_DEVICE";
                break;
            case CUFFT_PARSE_ERROR:
                ss << "CUFFT_PARSE_ERROR";
                break;
            case CUFFT_NO_WORKSPACE:
                ss << "CUFFT_NO_WORKSPACE";
                break;
            default:
                ss << "Unknown error code: " << result;
                break;
        }

        throw std::runtime_error(ss.str());
    }
}

void FFTPlanManager::createPlansR2C(
    int nx, int ny, int nz,
    int batch,
    bool use_float64
) {
    // Destroy existing plans if any
    destroyPlans();

    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    batch_ = batch;
    use_float64_ = use_float64;
    use_r2c_ = true;

    std::cout << "Creating cuFFT R2C/C2R plans for "
              << nx << "x" << ny << "x" << nz
              << " grid (batch=" << batch << ")" << std::endl;

    cufftResult result;

    if (batch == 1) {
        // Simple 3D plan
        if (use_float64) {
            result = cufftPlan3d(&forward_plan_, nz, ny, nx, CUFFT_D2Z);
            checkCufftError(result, "cufftPlan3d (D2Z forward)");

            result = cufftPlan3d(&inverse_plan_, nz, ny, nx, CUFFT_Z2D);
            checkCufftError(result, "cufftPlan3d (Z2D inverse)");
        } else {
            result = cufftPlan3d(&forward_plan_, nz, ny, nx, CUFFT_R2C);
            checkCufftError(result, "cufftPlan3d (R2C forward)");

            result = cufftPlan3d(&inverse_plan_, nz, ny, nx, CUFFT_C2R);
            checkCufftError(result, "cufftPlan3d (C2R inverse)");
        }
    } else {
        // Batched plan
        int n[3] = {nz, ny, nx};
        int inembed[3] = {nz, ny, nx};
        int onembed[3] = {nz, ny, nx};

        if (use_float64) {
            result = cufftPlanMany(&forward_plan_, 3, n,
                                  inembed, 1, nx * ny * nz,
                                  onembed, 1, nx * ny * nz,
                                  CUFFT_D2Z, batch);
            checkCufftError(result, "cufftPlanMany (D2Z forward)");

            result = cufftPlanMany(&inverse_plan_, 3, n,
                                  onembed, 1, nx * ny * nz,
                                  inembed, 1, nx * ny * nz,
                                  CUFFT_Z2D, batch);
            checkCufftError(result, "cufftPlanMany (Z2D inverse)");
        } else {
            result = cufftPlanMany(&forward_plan_, 3, n,
                                  inembed, 1, nx * ny * nz,
                                  onembed, 1, nx * ny * nz,
                                  CUFFT_R2C, batch);
            checkCufftError(result, "cufftPlanMany (R2C forward)");

            result = cufftPlanMany(&inverse_plan_, 3, n,
                                  onembed, 1, nx * ny * nz,
                                  inembed, 1, nx * ny * nz,
                                  CUFFT_C2R, batch);
            checkCufftError(result, "cufftPlanMany (C2R inverse)");
        }
    }

    // Query actual workspace sizes
    size_t forward_workspace = 0;
    size_t inverse_workspace = 0;

    result = cufftGetSize(forward_plan_, &forward_workspace);
    checkCufftError(result, "cufftGetSize (forward)");

    result = cufftGetSize(inverse_plan_, &inverse_workspace);
    checkCufftError(result, "cufftGetSize (inverse)");

    std::cout << "cuFFT workspace sizes:" << std::endl;
    std::cout << "  Forward: " << (forward_workspace / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Inverse: " << (inverse_workspace / (1024.0 * 1024.0)) << " MB" << std::endl;

    plans_created_ = true;
    std::cout << "cuFFT plans created successfully" << std::endl;
}

void FFTPlanManager::createPlansC2C(
    int nx, int ny, int nz,
    int batch,
    bool use_float64
) {
    // Destroy existing plans if any
    destroyPlans();

    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    batch_ = batch;
    use_float64_ = use_float64;
    use_r2c_ = false;

    std::cout << "Creating cuFFT C2C plans for "
              << nx << "x" << ny << "x" << nz
              << " grid (batch=" << batch << ")" << std::endl;

    cufftResult result;

    if (batch == 1) {
        // Simple 3D plan
        if (use_float64) {
            result = cufftPlan3d(&forward_plan_, nz, ny, nx, CUFFT_Z2Z);
            checkCufftError(result, "cufftPlan3d (Z2Z forward)");

            result = cufftPlan3d(&inverse_plan_, nz, ny, nx, CUFFT_Z2Z);
            checkCufftError(result, "cufftPlan3d (Z2Z inverse)");
        } else {
            result = cufftPlan3d(&forward_plan_, nz, ny, nx, CUFFT_C2C);
            checkCufftError(result, "cufftPlan3d (C2C forward)");

            result = cufftPlan3d(&inverse_plan_, nz, ny, nx, CUFFT_C2C);
            checkCufftError(result, "cufftPlan3d (C2C inverse)");
        }
    } else {
        // Batched plan
        int n[3] = {nz, ny, nx};
        int inembed[3] = {nz, ny, nx};
        int onembed[3] = {nz, ny, nx};

        if (use_float64) {
            result = cufftPlanMany(&forward_plan_, 3, n,
                                  inembed, 1, nx * ny * nz,
                                  onembed, 1, nx * ny * nz,
                                  CUFFT_Z2Z, batch);
            checkCufftError(result, "cufftPlanMany (Z2Z forward)");

            result = cufftPlanMany(&inverse_plan_, 3, n,
                                  inembed, 1, nx * ny * nz,
                                  onembed, 1, nx * ny * nz,
                                  CUFFT_Z2Z, batch);
            checkCufftError(result, "cufftPlanMany (Z2Z inverse)");
        } else {
            result = cufftPlanMany(&forward_plan_, 3, n,
                                  inembed, 1, nx * ny * nz,
                                  onembed, 1, nx * ny * nz,
                                  CUFFT_C2C, batch);
            checkCufftError(result, "cufftPlanMany (C2C forward)");

            result = cufftPlanMany(&inverse_plan_, 3, n,
                                  inembed, 1, nx * ny * nz,
                                  onembed, 1, nx * ny * nz,
                                  CUFFT_C2C, batch);
            checkCufftError(result, "cufftPlanMany (C2C inverse)");
        }
    }

    // Query actual workspace sizes
    size_t forward_workspace = 0;
    size_t inverse_workspace = 0;

    result = cufftGetSize(forward_plan_, &forward_workspace);
    checkCufftError(result, "cufftGetSize (forward)");

    result = cufftGetSize(inverse_plan_, &inverse_workspace);
    checkCufftError(result, "cufftGetSize (inverse)");

    std::cout << "cuFFT workspace sizes:" << std::endl;
    std::cout << "  Forward: " << (forward_workspace / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Inverse: " << (inverse_workspace / (1024.0 * 1024.0)) << " MB" << std::endl;

    plans_created_ = true;
    std::cout << "cuFFT plans created successfully" << std::endl;
}

void FFTPlanManager::destroyPlans() {
    if (!plans_created_) {
        return;
    }

    if (forward_plan_ != 0) {
        cufftDestroy(forward_plan_);
        forward_plan_ = 0;
    }

    if (inverse_plan_ != 0) {
        cufftDestroy(inverse_plan_);
        inverse_plan_ = 0;
    }

    plans_created_ = false;
}

void FFTPlanManager::executeForward(
    void* input,
    cufftComplex* output,
    cudaStream_t stream
) {
    if (!plans_created_) {
        throw std::runtime_error("FFT plans not created. Call createPlans* first.");
    }

    // Set stream if provided
    if (stream != 0) {
        cufftResult result = cufftSetStream(forward_plan_, stream);
        checkCufftError(result, "cufftSetStream (forward)");
    }

    cufftResult result;

    if (use_r2c_) {
        if (use_float64_) {
            result = cufftExecD2Z(forward_plan_, (cufftDoubleReal*)input,
                                 (cufftDoubleComplex*)output);
            checkCufftError(result, "cufftExecD2Z");
        } else {
            result = cufftExecR2C(forward_plan_, (cufftReal*)input, output);
            checkCufftError(result, "cufftExecR2C");
        }
    } else {
        if (use_float64_) {
            result = cufftExecZ2Z(forward_plan_, (cufftDoubleComplex*)input,
                                 (cufftDoubleComplex*)output, CUFFT_FORWARD);
            checkCufftError(result, "cufftExecZ2Z (forward)");
        } else {
            result = cufftExecC2C(forward_plan_, (cufftComplex*)input, output,
                                 CUFFT_FORWARD);
            checkCufftError(result, "cufftExecC2C (forward)");
        }
    }
}

void FFTPlanManager::executeInverse(
    cufftComplex* input,
    void* output,
    cudaStream_t stream
) {
    if (!plans_created_) {
        throw std::runtime_error("FFT plans not created. Call createPlans* first.");
    }

    // Set stream if provided
    if (stream != 0) {
        cufftResult result = cufftSetStream(inverse_plan_, stream);
        checkCufftError(result, "cufftSetStream (inverse)");
    }

    cufftResult result;

    if (use_r2c_) {
        if (use_float64_) {
            result = cufftExecZ2D(inverse_plan_, (cufftDoubleComplex*)input,
                                 (cufftDoubleReal*)output);
            checkCufftError(result, "cufftExecZ2D");
        } else {
            result = cufftExecC2R(inverse_plan_, input, (cufftReal*)output);
            checkCufftError(result, "cufftExecC2R");
        }
    } else {
        if (use_float64_) {
            result = cufftExecZ2Z(inverse_plan_, (cufftDoubleComplex*)input,
                                 (cufftDoubleComplex*)output, CUFFT_INVERSE);
            checkCufftError(result, "cufftExecZ2Z (inverse)");
        } else {
            result = cufftExecC2C(inverse_plan_, input, (cufftComplex*)output,
                                 CUFFT_INVERSE);
            checkCufftError(result, "cufftExecC2C (inverse)");
        }
    }
}

void FFTPlanManager::executeForwardInPlace(
    cufftComplex* data,
    cudaStream_t stream
) {
    if (!plans_created_) {
        throw std::runtime_error("FFT plans not created. Call createPlans* first.");
    }

    if (use_r2c_) {
        throw std::runtime_error("In-place FFT not supported for R2C/C2R plans. Use C2C plans.");
    }

    // Set stream if provided
    if (stream != 0) {
        cufftResult result = cufftSetStream(forward_plan_, stream);
        checkCufftError(result, "cufftSetStream (forward in-place)");
    }

    cufftResult result;

    if (use_float64_) {
        result = cufftExecZ2Z(forward_plan_, (cufftDoubleComplex*)data,
                             (cufftDoubleComplex*)data, CUFFT_FORWARD);
        checkCufftError(result, "cufftExecZ2Z (forward in-place)");
    } else {
        result = cufftExecC2C(forward_plan_, data, data, CUFFT_FORWARD);
        checkCufftError(result, "cufftExecC2C (forward in-place)");
    }
}

void FFTPlanManager::executeInverseInPlace(
    cufftComplex* data,
    cudaStream_t stream
) {
    if (!plans_created_) {
        throw std::runtime_error("FFT plans not created. Call createPlans* first.");
    }

    if (use_r2c_) {
        throw std::runtime_error("In-place FFT not supported for R2C/C2R plans. Use C2C plans.");
    }

    // Set stream if provided
    if (stream != 0) {
        cufftResult result = cufftSetStream(inverse_plan_, stream);
        checkCufftError(result, "cufftSetStream (inverse in-place)");
    }

    cufftResult result;

    if (use_float64_) {
        result = cufftExecZ2Z(inverse_plan_, (cufftDoubleComplex*)data,
                             (cufftDoubleComplex*)data, CUFFT_INVERSE);
        checkCufftError(result, "cufftExecZ2Z (inverse in-place)");
    } else {
        result = cufftExecC2C(inverse_plan_, data, data, CUFFT_INVERSE);
        checkCufftError(result, "cufftExecC2C (inverse in-place)");
    }
}

size_t FFTPlanManager::estimateWorkspaceSize(
    int nx, int ny, int nz,
    bool use_float64
) {
    // Create temporary plan to query actual workspace size from cuFFT
    cufftHandle temp_plan;
    size_t workspace_size = 0;
    cufftResult result;

    // Use C2C plan as it's most general (works for all cases)
    if (use_float64) {
        result = cufftPlan3d(&temp_plan, nz, ny, nx, CUFFT_Z2Z);
    } else {
        result = cufftPlan3d(&temp_plan, nz, ny, nx, CUFFT_C2C);
    }

    if (result != CUFFT_SUCCESS) {
        std::cerr << "Warning: Failed to create temporary plan for workspace estimation" << std::endl;
        // Return rough estimate as fallback
        size_t element_size = use_float64 ? sizeof(double) : sizeof(float);
        size_t grid_points = static_cast<size_t>(nx) * ny * nz;
        return grid_points * element_size * 4;  // Conservative fallback
    }

    // Query the actual workspace size cuFFT needs
    result = cufftGetSize(temp_plan, &workspace_size);

    if (result != CUFFT_SUCCESS) {
        std::cerr << "Warning: Failed to query workspace size" << std::endl;
        workspace_size = 0;
    }

    cufftDestroy(temp_plan);

    return workspace_size;
}

} // namespace bpmfwfft
