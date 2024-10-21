#include "nnls.cuh"
#include <stdio.h>
#include <float.h>

// Utility function templates
template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ double reduce(double *smem, unsigned int tID) {
    __syncthreads();
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tID < s) {
            smem[tID] += smem[tID + s];
        }
        __syncthreads();
    }
    return smem[0];
}

template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ int maxIndex(double *smem, double *smemC, unsigned int tID) {
    __syncthreads();
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tID < s) {
            double l = smem[tID], r = smem[tID + s];
            if (r > l) {
                smem[tID] = r;
                smemC[tID] = smemC[tID + s];
            }
        }
        __syncthreads();
    }
    return (int)smemC[0];
}

template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ double norml2(double *smem, double *V, unsigned int tID) {
    __syncthreads();
    smem[tID] = V[tID] * V[tID];
    return sqrt(reduce<BLOCK_SIZE>(smem, tID));
}

template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ double norml2(double *smem, double v, unsigned int tID) {
    __syncthreads();
    smem[tID] = v * v;
    return sqrt(reduce<BLOCK_SIZE>(smem, tID));
}

template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ double min(double *smem, unsigned int tID) {
    __syncthreads();
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tID < s) {
            smem[tID] = fmin(smem[tID + s], smem[tID]);
        }
        __syncthreads();
    }
    return smem[0];
}

__global__ void NNLS_MGS_GR_512(double *d_A, double *d_At, double *d_x, double *d_b,
                                double *d_R, int *nIters, int *lsIters,
                                int NSYS, int N, int M,
			        int MAX_ITER_LS, int MAX_ITER_NNLS,	
				double TOL_TERMINATION, double TOL_0) {
    
    
    extern __shared__ double sharedMem[];
    double *sDat = sharedMem;
    double *sZ = &sDat[M];
    double *sB = &sZ[M];
    int *sK = (int*)&sB[N];
    double *Qt = &((double*)&sK[M])[0];

    __shared__ int nnlsCount;
    __shared__ int lsCount;

    unsigned int tID = threadIdx.x;
    //unsigned short tID = threadIdx.x;
    unsigned int sysID = blockIdx.y * gridDim.x + blockIdx.x;

    // Ensure we don't process beyond NSYS
    if (sysID >= NSYS) return;

    __shared__ int sSysID;
    __shared__ int kCols;

    double x = 0;
    bool Z = true;

    double QtB;

    if(tID == 0) {
        nnlsCount = 0;
        lsCount = 0;
        sSysID = sysID;
        kCols = 0;
    }
    __syncthreads();

    if (tID < N) {
        sB[tID] = d_b[B_I(sysID, tID)];
    }

    do {
        double Ax;
        double Gx;
        for(short a = 0; a < N; ++a) {
            __syncthreads();
            if (tID < M) {
                Ax = d_A[A_I(sysID, a, tID)] * x;
            }
            __syncthreads();
            if (tID < M) {
                sDat[tID] = Ax;
            }
            Ax = reduce<512>(sDat, tID);
            if(tID == a && tID < N) sZ[a] = sB[a] - Ax;
	    //if(tID == a) sZ[a] = sB[tID] - Ax;
        }
        for(short a = 0; a < N; ++a) {
            __syncthreads();
            if (tID < M) {
                Ax = d_At[A_I(sysID, a, tID)] * sZ[tID];
            }
            __syncthreads();
            if (tID < M) {
                sDat[tID] = Ax;
            }
            Ax = reduce<512>(sDat, tID);
            if(tID == a) Gx = Ax;
        }

        __syncthreads();
        if (tID < M) {
            sDat[tID] = ((Z * (Gx > TOL_TERMINATION)) ? 1 : 0);
        }
        bool wAllNonPositive = !((bool)reduce<512>(sDat, tID));
        if(wAllNonPositive) break;
            
        __syncthreads();
        if (tID < M) {
            sDat[tID] = (Z ? Gx : -DBL_MAX);
        }
        unsigned short maxZinW = (unsigned short)maxIndex<512>(sDat, sZ, tID);
        if(tID == maxZinW) Z = false;
        __syncthreads();
        
        bool addColumn = true;

        do {
            if(tID == 0) ++lsCount;
            __syncthreads();
        
            if(addColumn) {
                if (tID < M) {
                    sZ[tID] = Z;
                }
                
                double newCol = d_At[A_I(sysID, maxZinW, tID)];
                __syncthreads();
                double oldCol = newCol;
                for(short k = 0; k < kCols; ++k) {
                    short a = sK[k];
                    __syncthreads();
                    double Ax = Qt[a];
                    __syncthreads();
                    if (tID < M) {
                        sDat[tID] = newCol * Ax;
                    }
                    Gx = reduce<512>(sDat, tID);
                    __syncthreads();
                    newCol -= Ax * Gx;
                    
                    if (tID < M) {
                        sDat[tID] = oldCol * Ax;
                    }
                    Gx = reduce<512>(sDat, tID);
                    if(tID == 0) d_R[R_I(sysID, a, maxZinW)] = Gx;
                }

                __syncthreads();
                Gx = norml2<512>(sDat, newCol, tID);
                newCol /= Gx;
                __syncthreads();

                if (tID < M) {
                    d_R[R_I(sysID, maxZinW, tID)] = (tID == maxZinW) * Gx;
                    Qt[maxZinW] = newCol;

                    sDat[tID] = newCol * sB[tID];
                }
                Gx = reduce<512>(sDat, tID);
                if (tID == maxZinW) QtB = Gx;

                if(tID == 0) sK[kCols++] = maxZinW;
                
                __syncthreads();

            } else {
                bool removedVars = ((int)sZ[tID] != Z);
                __syncthreads();
                if (tID < M) {
                    sZ[tID] = Z;
                }

                __shared__ bool deletedCol;

                for(short k = kCols-1; k >= 0; --k) {
                    __syncthreads();
                    short a = sK[k];
                    
                    if(tID == 0) deletedCol = false;
                    __syncthreads();
                    if(tID == a && removedVars) deletedCol = true;
                    __syncthreads();

                    if(deletedCol) {
                        __shared__ double givenc;
                        __shared__ double givens;

                        Gx = d_R[R_I(sysID, a, tID)];

                        for(short b = k+1; b < kCols; ++b) {
                            __syncthreads();
                            short tc = sK[b];
                
                            __syncthreads();
                            Ax = d_R[R_I(sysID, tc, tID)];
                            __syncthreads();

                            if(tID == tc) {    
                                if(Ax == 0 && Gx == 0) {
                                    givenc = 0;
                                    givens = 0;
                                } else if(Ax == 0) {
                                    givenc = 0;
                                    givens = 1;
                                } else if (Gx == 0) {
                                    givenc = 1;
                                    givens = 0;
                                } else if (fabs(Gx) > fabs(Ax)) {
                                    double r = -Ax / Gx;
                                    givens = 1 / sqrt(1 + r * r);
                                    givenc = -givens * r;
                                } else {
                                    double r = -Gx / Ax;
                                    givenc = 1 / sqrt(1 + r * r);
                                    givens = -givenc * r;
                                }                            
                            }
                            __syncthreads();

                            double tAx = Ax * givenc + Gx * givens;
                            double tGx = Ax * -givens + Gx * givenc;
                            __syncthreads();
                            d_R[R_I(sysID, tc, tID)] = tAx;
                            Gx = tGx;

                            __syncthreads();
                            tAx = Qt[tc];
                            tGx = Qt[a];
                            __syncthreads();
                            Qt[tc] = tAx * givenc + tGx * givens;
                            Qt[a] = tAx * -givens + tGx * givenc;

                            __shared__ double bb, bt;
                            if(tID == tc) bt = QtB;
                            else if(tID == a) bb = QtB;
                            __syncthreads();
                            if(tID == tc) QtB = bt * givenc + bb * givens;
                            else if(tID == a) QtB = bt * -givens + bb * givenc;
                        }

                        __syncthreads();
                        d_R[R_I(sysID, a, tID)] = Gx;

                        __syncthreads();
                        if(tID >= k && tID < kCols-1) Ax = sK[tID + 1];
                        __syncthreads();
                        if(tID >= k && tID < kCols-1) sK[tID] = Ax;
                        __syncthreads();
                        if(tID == 0) kCols--;
                    }
                }
                __syncthreads();
            }
            __syncthreads();    
        
            Ax = 0;
            for(short b = kCols-1; b >= 0; --b) {
                short a = sK[b];
                __syncthreads();    

                double coeff = d_R[R_I(sysID, a, tID)] * !Z;

                __syncthreads();
                if (tID < M) {
                    sDat[tID] = (!(tID == a)) * coeff * Ax;
                }
                double pSum = reduce<512>(sDat, tID);

                if((tID == a))
                    Ax = ((coeff == 0) ? (0) : (QtB - pSum) / coeff);
            }
            __syncthreads();

            if (tID < M) {
                sDat[tID] = (Z ? DBL_MAX : Ax);
            }
            if(min<512>(sDat, tID) > 0) {
                x = Ax;
                break;
            } else {
                __syncthreads();
                if (tID < M) {
                    if(!Z && Ax <= 0) sDat[tID] = x / (x - Ax);
                    else sDat[tID] = DBL_MAX;
                }
                double alpha = min<512>(sDat, tID);
                
                x += alpha * (Ax - x);

                Z = (Z) || (fabs(x) <= TOL_0);
                addColumn = false;
            }

        } while(lsCount < MAX_ITER_LS);

        if(lsCount >= MAX_ITER_LS)
            break;

        if(tID == 0) ++nnlsCount;
        __syncthreads();
    } while(nnlsCount < MAX_ITER_NNLS);

    __syncthreads();
    if (tID < M) {
        d_x[X_I(sSysID, tID)] = x;
    }

    if (tID == 0) {
        nIters[sSysID] = nnlsCount;
        lsIters[sSysID] = lsCount;
    }
}
