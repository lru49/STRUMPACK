/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */

#define STRUMPACK_NO_TRIPLET_MPI
#include "CUDAWrapper.hpp"

namespace strumpack {
  namespace gpu {

    template<typename scalar_t> __global__ void
    laswp_kernel(int n, scalar_t* dA, int lddA,
                 int npivots, int* dipiv, int inci) {
      int tid = threadIdx.x + blockDim.x*blockIdx.x;
      if (tid < n) {
        dA += tid * lddA;
        auto A1 = dA;
        for (int i1=0; i1<npivots; i1++) {
          int i2 = dipiv[i1*inci] - 1;
          auto A2 = dA + i2;
          auto temp = *A1;
          *A1 = *A2;
          *A2 = temp;
          A1++;
        }
      }
    }

    template<typename scalar_t> void
    laswp(Handle& handle, DenseMatrix<scalar_t>& dA,
          int k1, int k2, int* dipiv, int inci) {
      if (!dA.rows() || !dA.cols()) return;
      int n = dA.cols(), nt = 256;
      int grid = (n + nt - 1) / nt;
      cudaStream_t streamId;
      cublasGetStream(get_cublas_handle(handle), &streamId);
      laswp_kernel<scalar_t><<<grid, nt, 0, streamId>>>
        (n, dA.data(), dA.ld(), k2-k1+1, dipiv+k1-1, inci);
      gpu_check(cudaPeekAtLastError());
    }

    template<typename T>  __global__ void
    laswp_vbatch_kernel(int* dn, T** dA, int* lddA, int** dipiv,
                        int* npivots, unsigned int batchCount) {
      // assume dn = cols, inc = 1
      int x = blockIdx.x * blockDim.x + threadIdx.x,
        f = blockIdx.y * blockDim.y + threadIdx.y;
      if (f >= batchCount) return;
      if (x >= dn[f]) return;
      auto A = dA[f];
      auto P = dipiv[f];
      auto ldA = lddA[f];
      auto npiv = npivots[f];
      A += x * ldA;
      auto A1 = A;
      for (int i=0; i<npiv; i++) {
        auto p = P[i] - 1;
        if (p != i) {
          auto A2 = A + p;
          auto temp = *A1;
          *A1 = *A2;
          *A2 = temp;
        }
        A1++;
      }
    }

    // S-LASWP (magma-hpc, 2026-10-27): the solve's row interchanges as a parallel gather. laswp_vbatch_kernel runs ONE
    // thread per (front, column) through up to dim_sep dependent swaps (3200 for the root front, ~1.1 ms per solve
    // with nrhs = 1). The pivot sequence of every front is turned into a permutation once per factorization
    // (ipiv_to_perm), then each solve applies it with one block per front: stage the column in shared memory and
    // write x[i] = s[perm[i]]. Bitwise the same values as the swap sequence.
    __global__ void ipiv_to_perm_kernel(int** dipiv, int* npiv, int** dperm, unsigned int batchCount) {
      unsigned int f = blockIdx.x * blockDim.x + threadIdx.x;
      if (f >= batchCount) return;
      const int n = npiv[f]; const int* P = dipiv[f]; int* perm = dperm[f];
      for (int i=0; i<n; i++) perm[i] = i;
      for (int i=0; i<n; i++) {
        const int p = P[i] - 1;
        if (p != i) { const int t = perm[i]; perm[i] = perm[p]; perm[p] = t; }
      }
    }
    void ipiv_to_perm_vbatched(Handle& handle, int** dipiv, int* npiv, int** dperm, unsigned int batchCount) {
      if (!batchCount) return;
      cudaStream_t streamId;
      cublasGetStream(get_cublas_handle(handle), &streamId);
      const unsigned int nt = 128, nb = (batchCount + nt - 1) / nt;
      ipiv_to_perm_kernel<<<nb, nt, 0, streamId>>>(dipiv, npiv, dperm, batchCount);
      gpu_check(cudaPeekAtLastError());
    }
    template<typename T> __global__ void
    permute_rows_kernel(int* dn, T** dA, int* lddA, int** dperm, int* npiv, unsigned int batchCount) {
      extern __shared__ char smem_raw[];
      T* s = reinterpret_cast<T*>(smem_raw);
      const unsigned int f = blockIdx.y;
      if (f >= batchCount) return;
      const int n = npiv[f], ncol = dn[f], ld = lddA[f];
      T* A = dA[f]; const int* perm = dperm[f];
      for (int c=0; c<ncol; c++) {
        T* col = A + std::size_t(c) * ld;
        for (int i=threadIdx.x; i<n; i+=blockDim.x) s[i] = col[i];
        __syncthreads();
        for (int i=threadIdx.x; i<n; i+=blockDim.x) col[i] = s[perm[i]];
        __syncthreads();
      }
    }
    template<typename scalar_t> bool
    permute_rows_fwd_vbatched(Handle& handle, int* dn, scalar_t** dA, int* lddA, int** dperm,
                              int* npiv, int max_npiv, unsigned int batchCount) {
      if (!batchCount || max_npiv <= 0) return true;
      const std::size_t smem = std::size_t(max_npiv) * sizeof(scalar_t);
      if (smem > 48 * 1024) return false;   // caller falls back to the swap kernel
      cudaStream_t streamId;
      cublasGetStream(get_cublas_handle(handle), &streamId);
      unsigned int nt = 32; while (nt < (unsigned int)max_npiv && nt < 1024) nt *= 2;
      for (unsigned int f=0; f<batchCount; f+=MAX_BLOCKS_Y) {
        dim3 grid(1, std::min(batchCount-f, (unsigned int)MAX_BLOCKS_Y));
        permute_rows_kernel<scalar_t><<<grid, nt, smem, streamId>>>
          (dn+f, dA+f, lddA+f, dperm+f, npiv+f, batchCount-f);
        gpu_check(cudaPeekAtLastError());
      }
      return true;
    }
    template bool permute_rows_fwd_vbatched(Handle&, int*, float**, int*, int**, int*, int, unsigned int);
    template bool permute_rows_fwd_vbatched(Handle&, int*, double**, int*, int**, int*, int, unsigned int);
    template bool permute_rows_fwd_vbatched(Handle&, int*, std::complex<float>**, int*, int**, int*, int, unsigned int);
    template bool permute_rows_fwd_vbatched(Handle&, int*, std::complex<double>**, int*, int**, int*, int, unsigned int);

    // S-INVDIAG (magma-hpc, 2026-10-27) fused block-recurrence step, see GPUWrapper.hpp. Threads: 4 column lanes x NB
    // rows; lane L reads columns j = L (mod 4) of its chunk so that a warp reads 32 consecutive rows of one column
    // (coalesced); the 4 lane partials are reduced through shared memory. S > 1: each CTA stores its NB partial sums,
    // fences, bumps the front's counter; the last CTA sums the chunks (L2 loads), forms t = yin - sum and applies the
    // inverse (again 4 lanes over the columns of the NB x NB block), then resets the counter for the next step.
    template<typename T, int NB> __global__ void __launch_bounds__(4*NB)
    invdiag_step_kernel(const int* m, const int* n, T* const* A, const int* lda, T* const* x, T* const* yin,
                        T* const* inv, T** out, T* partial, int* counter, int chunk, int smax,
                        const int* m_next, const int* n_next, T* const* A_next) {
      __shared__ T red[4][NB];
      __shared__ T t[NB];
      __shared__ int s_last;
      const int i = blockIdx.x, s = blockIdx.y;
      const int r = m[i];
      if (r <= 0) return;
      const int nn = n[i];
      const int S = nn > 0 ? (nn + chunk - 1) / chunk : 1;
      if (s >= S) return;
      const int tid = threadIdx.x, row = tid % NB, lane = tid / NB;
      if (m_next) {
        // prefetch (into L2) the NEXT step's block row, which does not depend on this step's result: the S CTAs of
        // the front share its columns; one 128-byte line per thread per iteration (user suggestion, v2 brief)
        const int rn = m_next[i], cn = n_next[i];
        if (rn > 0 && cn > 0) {
          const char* An = reinterpret_cast<const char*>(A_next[i]);
          const std::size_t ldb = std::size_t(lda[i]) * sizeof(T);
          const int lines = (rn * int(sizeof(T)) + 127) / 128 + 1;
          const int cs = (cn + S - 1) / S, cb = s * cs, ce = min(cn, cb + cs);
          for (int e = tid; e < (ce - cb) * lines; e += 4*NB) {
            const int j = cb + e / lines, ln = e % lines;
            const char* p = An + std::size_t(j) * ldb + std::size_t(ln) * 128;
            asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
          }
        }
      }
      T acc = T(0);
      if (row < r && nn > 0) {
        const T* Ai = A[i] + row;
        const std::size_t ld = lda[i];
        const T* xi = x[i];
        const int c0 = s * chunk, c1 = min(nn, c0 + chunk);
        int j = c0 + lane;
        for (; j + 12 < c1; j += 16) {
          const T a0 = Ai[j*ld], a1 = Ai[(j+4)*ld], a2 = Ai[(j+8)*ld], a3 = Ai[(j+12)*ld];
          acc += a0 * xi[j]; acc += a1 * xi[j+4]; acc += a2 * xi[j+8]; acc += a3 * xi[j+12];
        }
        for (; j < c1; j += 4) acc += Ai[j*ld] * xi[j];
      }
      red[lane][row] = acc;
      __syncthreads();
      if (S == 1) {
        if (lane == 0 && row < r) t[row] = yin[i][row] - (red[0][row] + red[1][row] + red[2][row] + red[3][row]);
      } else {
        if (lane == 0) partial[(std::size_t(i)*smax + s)*NB + row] = red[0][row] + red[1][row] + red[2][row] + red[3][row];
        __threadfence();
        __syncthreads();
        if (tid == 0) s_last = (atomicAdd(&counter[i], 1) == S - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        if (lane == 0) {
          T p = T(0);
          for (int q=0; q<S; q++) p += __ldcg(&partial[(std::size_t(i)*smax + q)*NB + row]);
          if (row < r) t[row] = yin[i][row] - p;
        }
        if (tid == 0) counter[i] = 0;
      }
      __syncthreads();
      T acc2 = T(0);
      if (row < r) {
        const T* Ii = inv[i] + row;
#pragma unroll 4
        for (int j=lane; j<r; j+=4) acc2 += Ii[j*NB] * t[j];
      }
      __syncthreads();
      red[lane][row] = acc2;
      __syncthreads();
      if (lane == 0 && row < r) out[i][row] = red[0][row] + red[1][row] + red[2][row] + red[3][row];
    }
    template<typename T> bool
    invdiag_step_real(Handle& handle, int N, int S, int NB, int chunk, const int* m, const int* n,
                      T** A, const int* lda, T** x, T** yin, T** inv, T** out, T* partial, int* counter,
                      const int* m_next, const int* n_next, T** A_next) {
      if (N <= 0) return true;
      if (NB != 128 && NB != 256) return false;
      cudaStream_t streamId;
      cublasGetStream(get_cublas_handle(handle), &streamId);
      dim3 grid(N, std::max(1, S));
      if (NB == 256)
        invdiag_step_kernel<T,256><<<grid, 1024, 0, streamId>>>(m, n, A, lda, x, yin, inv, out, partial, counter, chunk, 32, m_next, n_next, A_next);
      else
        invdiag_step_kernel<T,128><<<grid, 512, 0, streamId>>>(m, n, A, lda, x, yin, inv, out, partial, counter, chunk, 32, m_next, n_next, A_next);
      gpu_check(cudaPeekAtLastError());
      return true;
    }
    template<typename scalar_t> bool
    invdiag_step(Handle&, int, int, int, int, const int*, const int*, scalar_t**, const int*, scalar_t**, scalar_t**,
                 scalar_t**, scalar_t**, scalar_t*, int*, const int*, const int*, scalar_t**) { return false; }
    template<> bool
    invdiag_step<float>(Handle& h, int N, int S, int NB, int chunk, const int* m, const int* n, float** A, const int* lda,
                        float** x, float** yin, float** inv, float** out, float* partial, int* counter,
                        const int* mn, const int* nn, float** An) {
      return invdiag_step_real<float>(h, N, S, NB, chunk, m, n, A, lda, x, yin, inv, out, partial, counter, mn, nn, An);
    }
    template<> bool
    invdiag_step<double>(Handle& h, int N, int S, int NB, int chunk, const int* m, const int* n, double** A, const int* lda,
                         double** x, double** yin, double** inv, double** out, double* partial, int* counter,
                         const int* mn, const int* nn, double** An) {
      return invdiag_step_real<double>(h, N, S, NB, chunk, m, n, A, lda, x, yin, inv, out, partial, counter, mn, nn, An);
    }
    template bool invdiag_step(Handle&, int, int, int, int, const int*, const int*, std::complex<float>**, const int*, std::complex<float>**, std::complex<float>**, std::complex<float>**, std::complex<float>**, std::complex<float>*, int*, const int*, const int*, std::complex<float>**);
    template bool invdiag_step(Handle&, int, int, int, int, const int*, const int*, std::complex<double>**, const int*, std::complex<double>**, std::complex<double>**, std::complex<double>**, std::complex<double>**, std::complex<double>*, int*, const int*, const int*, std::complex<double>**);

    // S-HOST (magma-hpc, 2026-10-27): the sparse reordering applied to the right-hand side / solution on the device,
    // dst(i,j) = src(perm[i],j) for an n x nrhs column-major block with ld = n (one thread per row). A pure gather:
    // bitwise the values of the host loops in SparseSolver::transform_b / transform_x.
    template<typename T> __global__ void
    gather_rows_kernel(int n, int nrhs, const T* src, T* dst, const int* perm) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n) return;
      const int p = perm[i];
      for (int j=0; j<nrhs; j++) dst[i + std::size_t(j) * n] = src[p + std::size_t(j) * n];
    }
    template<typename scalar_t> void
    gather_rows(Handle& handle, int n, int nrhs, const scalar_t* src, scalar_t* dst, const int* perm) {
      if (n <= 0 || nrhs <= 0) return;
      cudaStream_t streamId;
      cublasGetStream(get_cublas_handle(handle), &streamId);
      const unsigned int nt = 256, nb = (unsigned int)(n + nt - 1) / nt;
      gather_rows_kernel<scalar_t><<<nb, nt, 0, streamId>>>(n, nrhs, src, dst, perm);
      gpu_check(cudaPeekAtLastError());
    }
    template void gather_rows(Handle&, int, int, const float*, float*, const int*);
    template void gather_rows(Handle&, int, int, const double*, double*, const int*);
    template void gather_rows(Handle&, int, int, const std::complex<float>*, std::complex<float>*, const int*);
    template void gather_rows(Handle&, int, int, const std::complex<double>*, std::complex<double>*, const int*);

    template<typename scalar_t> void
    laswp_fwd_vbatched(Handle& handle, int* dn, int max_n,
                       scalar_t** dA, int* lddA, int** dipiv, int* npivots,
                       unsigned int batchCount) {
      if (max_n <= 0 || !batchCount) return;
      unsigned int nt = 512, ops = 1;
      while (nt > max_n) {
        nt /= 2;
        ops *= 2;
      }
      ops = std::min(ops, batchCount);
      unsigned int nbx = (max_n + nt - 1) / nt,
        nbf = (batchCount + ops - 1) / ops;
      dim3 block(nt, ops);
      for (unsigned int f=0; f<nbf; f+=MAX_BLOCKS_Y) {
        dim3 grid(nbx, std::min(nbf-f, MAX_BLOCKS_Y));
        cudaStream_t streamId;
        cublasGetStream(get_cublas_handle(handle), &streamId);
        auto f0 = f * ops;
        laswp_vbatch_kernel<<<grid, block, 0, streamId>>>
          (dn+f0, dA+f0, lddA+f0, dipiv+f0, npivots+f0, batchCount-f0);
        gpu_check(cudaPeekAtLastError());
      }
    }

    // explicit template instantiations
    template void laswp(Handle&, DenseMatrix<float>&, int, int, int*, int);
    template void laswp(Handle&, DenseMatrix<double>&, int, int, int*, int);
    template void laswp(Handle&, DenseMatrix<std::complex<float>>&, int, int, int*, int);
    template void laswp(Handle&, DenseMatrix<std::complex<double>>&, int, int, int*, int);

    template void laswp_fwd_vbatched(Handle&, int*, int, float**, int*, int**, int*, unsigned int);
    template void laswp_fwd_vbatched(Handle&, int*, int, double**, int*, int**, int*, unsigned int);
    template void laswp_fwd_vbatched(Handle&, int*, int, std::complex<float>**, int*, int**, int*, unsigned int);
    template void laswp_fwd_vbatched(Handle&, int*, int, std::complex<double>**, int*, int**, int*, unsigned int);

  } // end namespace gpu
} // end namespace strumpack
