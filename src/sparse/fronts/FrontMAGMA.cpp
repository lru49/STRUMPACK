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
#include <array>
#include <chrono>
#include <cstdlib>
#if defined(STRUMPACK_USE_CUDA)
#include <cuda_runtime_api.h>
#endif
#include <cstring>
#include <nvtx3/nvToolsExt.h>   // S-INVDIAG: header-only NVTX ranges for the solve trace (STRUM_SOLVE_NVTX=1)
#include <limits>
#include <type_traits>
#include <fstream>
#include <iostream>
#include <vector>

#include "FrontMAGMA.hpp"
#include "FrontGPUKernels.hpp"
#include "dense/MAGMAWrapper.hpp"

#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontMPI.hpp"
#include "FrontBLRMPI.hpp"
#endif

namespace strumpack {

  /**
   * MAGMA 2.9.0 and earlier split the row interchanges inside
   * magma_Xgetrf_vbatched into launches of at most 65535 matrices
   * (the CUDA grid z limit) without offsetting the batch arrays, so
   * the LU factors are wrong when batchCount > 65535 and a matrix is
   * larger than 32x32. Factoring at most 65535 fronts per call avoids
   * that code path.
   */
  template<typename scalar_t> void
  getrf_vbatched_chunked(int* d1, int max_d1, scalar_t** F11, int* ld1,
                         int** ipiv, int* info, void* work, int* lwork,
                         std::size_t N, gpu::Handle& handle) {
    const std::size_t max_batch = 65535;
    for (std::size_t i=0; i<N; i+=max_batch)
      gpu::magma::getrf_vbatched_max_nocheck_work
        (d1+i, d1+i, max_d1, max_d1, max_d1, max_d1*max_d1, F11+i,
         ld1+i, ipiv+i, info+i, work, lwork, std::min(max_batch, N-i),
         handle);
  }

  template<typename scalar_t, typename integer_t> class LevelInfoMAGMA {
    using F_t = Front<scalar_t,integer_t>;
    using FM_t = FrontMAGMA<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
  public:
    LevelInfoMAGMA() {}
    LevelInfoMAGMA(const std::vector<F_t*>& fronts,
                   gpu::Handle& handle, const SpMat_t* A=nullptr) {
      auto N = fronts.size();
      f.reserve(N);
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FM_t*>(F));
      std::size_t max_dsep = 0;
#pragma omp parallel for                                                \
  reduction(max:max_dsep)                                               \
  reduction(+:factor_size,Schur_size,piv_size)                          \
  reduction(+:total_upd_size)
      for (std::size_t i=0; i<N; i++) {
        const std::size_t dsep = f[i]->dim_sep();
        const std::size_t dupd = f[i]->dim_upd();
        factor_size += dsep*(dsep + 2*dupd);
        Schur_size += dupd*dupd;
        piv_size += dsep;
        total_upd_size += dupd;
        if (dsep > max_dsep) max_dsep = dsep;
      }
      Nsmall = (N > MIN_BATCH_COUNT) ? N : 0;
      if (A) {
        elems11.resize(N+1);
        elems12.resize(N+1);
        elems21.resize(N+1);
#pragma omp parallel for
        for (std::size_t i=0; i<N; i++) {
          auto& F = *(f[i]);
          A->count_front_elements
            (F.sep_begin(), F.sep_end(), F.upd(),
             elems11[i+1], elems12[i+1], elems21[i+1]);
        }
        for (std::size_t i=0; i<N; i++) {
          elems11[i+1] += elems11[i];
          elems12[i+1] += elems12[i];
          elems21[i+1] += elems21[i];
        }
      }
      Isize.resize(N+1);
      for (std::size_t i=0; i<N; i++) {
        auto& F = *(f[i]);
        Isize[i+1] = Isize[i];
        if (F.lchild_) Isize[i+1] += F.lchild_->dim_upd();
        if (F.rchild_) Isize[i+1] += F.rchild_->dim_upd();
      }
      batch_meta_bytes = gpu::round_up(4*(Nsmall+1)*sizeof(int));
      batch_meta_bytes += gpu::round_up(4*Nsmall*sizeof(scalar_t*));
      batch_meta_bytes += gpu::round_up(Nsmall*sizeof(int*));
      dev_batch_meta = gpu::HostMemory<char>(batch_meta_bytes);
      d1_batch = dev_batch_meta.template as<int>();
      d2_batch = d1_batch + Nsmall + 1;
      ld1_batch = d2_batch + Nsmall + 1;
      ld2_batch = ld1_batch + Nsmall + 1;
      F_batch = gpu::aligned_ptr<scalar_t*>(d1_batch+4*(Nsmall+1));
      ipiv_batch = gpu::aligned_ptr<int*>(F_batch+4*Nsmall);
#pragma omp parallel for                        \
  reduction(max:max_d1_small, max_d2_small)
      for (std::size_t i=0; i<Nsmall; i++) {
        const int dsep = f[i]->dim_sep();
        const int dupd = f[i]->dim_upd();
        d1_batch[i] = dsep;
        d2_batch[i] = dupd;
        max_d1_small = std::max(max_d1_small, dsep);
        max_d2_small = std::max(max_d2_small, dupd);
        ld1_batch[i] = std::max(1, dsep);
        ld2_batch[i] = std::max(1, dupd);
      }

      if (A) { // not needed for the solve
        // TODO not needed when always using MAGMA? also should now be
        // std::int64_t, and work for getrs as well?
        int getrf_work_cusolver = sizeof(scalar_t) *
          gpu::getrf_buffersize<scalar_t>(handle, max_dsep);
        getrf_work_bytes = -1;
        gpu::magma::getrf_vbatched_max_nocheck_work
          (nullptr, nullptr, max_d1_small, max_d1_small,
           max_d1_small, max_d1_small*max_d1_small,
           (scalar_t**)nullptr, nullptr, nullptr, nullptr, nullptr,
           &getrf_work_bytes, Nsmall, handle);
        getrf_work_bytes = std::max(getrf_work_bytes, getrf_work_cusolver);
        factor_bytes = gpu::round_up(sizeof(scalar_t)*factor_size);
        factor_bytes += gpu::round_up(sizeof(int)*piv_size);
      }
      work_bytes = batch_meta_bytes;
      if (A) { // not needed for the solve
        work_bytes += gpu::round_up(sizeof(int)*N);
        work_bytes += gpu::round_up(sizeof(scalar_t)*Schur_size);
        work_bytes += gpu::round_up(getrf_work_bytes);
      }
      ea_bytes = gpu::round_up(sizeof(gpu::AssembleData<scalar_t>)*f.size());
      ea_bytes += gpu::round_up(sizeof(std::size_t)*Isize.back());
      if (A)
        ea_bytes += gpu::round_up
          (sizeof(Triplet<scalar_t>) *
           (elems11.back()+elems12.back()+elems21.back()));
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << Nsmall << " small fronts, needs "
                << factor_bytes / 1.e6
                << " MB for factors, "
                << Schur_size * sizeof(scalar_t) / 1.e6
                << " MB for Schur complements" << std::endl;
    }

    void flops(long long& level_flops, long long& small_flops) {
      level_flops = small_flops = 0;
      auto N = f.size();
#pragma omp parallel for reduction(+: level_flops, small_flops)
      for (std::size_t i=0; i<N; i++) {
        auto F = f[i];
        auto flops = LU_flops(F->F11_) +
          gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                     F->F21_, F->F12_, scalar_t(1.)) +
          trsm_flops(Side::L, scalar_t(1.), F->F11_, F->F12_) +
          trsm_flops(Side::R, scalar_t(1.), F->F11_, F->F21_);
        level_flops += flops;
        if (f.size() > MIN_BATCH_COUNT)
          small_flops += flops;
      }
    }

    void set_factor_pointers(void* factors, void* pivots=nullptr) {
      auto fmem = static_cast<scalar_t*>(factors);
      for (auto F : f) {
        const std::size_t dsep = F->dim_sep();
        const std::size_t dupd = F->dim_upd();
        F->F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
        F->F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
        F->F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
      }
      int* pmem = nullptr;
      if (pivots) pmem = static_cast<int*>(pivots);
      else pmem = gpu::aligned_ptr<int>(fmem);
      for (auto F : f) {
        F->piv_ = pmem;
        pmem += F->dim_sep();
      }
    }

    void set_work_pointers(void* wmem, void* batch_mem=nullptr) {
      auto schur = gpu::aligned_ptr<scalar_t>(wmem);
      for (auto F : f) {
        auto dupd = F->dim_upd();
        if (dupd) {
          F->F22_ = DenseMW_t(dupd, dupd, schur, dupd);
          schur += dupd*dupd;
        }
      }
      auto gmem = gpu::aligned_ptr<char>(schur);
      dev_getrf_work = gmem;   gmem += getrf_work_bytes;
      auto N = f.size();
      auto imem = gpu::aligned_ptr<int>(gmem);
      dev_getrf_err = imem;    imem += N;
      set_batch_data(batch_mem ? batch_mem : imem);   // F-GATHER: batch meta may live in persistent memory
    }

    void set_batch_data(void* mem) {
      auto imem = gpu::aligned_ptr<int>(mem);
      dev_d1_batch  = imem;    imem += Nsmall+1;
      dev_d2_batch  = imem;    imem += Nsmall+1;
      dev_ld1_batch = imem;    imem += Nsmall+1;
      dev_ld2_batch = imem;    imem += Nsmall+1;
      auto fmem = gpu::aligned_ptr<scalar_t*>(imem);
      dev_F_batch = fmem;      fmem += 4 * Nsmall;
      auto ipmem = gpu::aligned_ptr<int*>(fmem);
      dev_ipiv_batch = ipmem;  ipmem += Nsmall;
#pragma omp parallel for
      for (std::size_t i=0; i<Nsmall; i++) {
        F_batch[i           ] = f[i]->F11_.data();
        F_batch[i +   Nsmall] = f[i]->F12_.data();
        F_batch[i + 2*Nsmall] = f[i]->F21_.data();
        F_batch[i + 3*Nsmall] = f[i]->F22_.data();
        ipiv_batch[i] = f[i]->piv_;
      }
      gpu::copy_host_to_device<char>
        (gpu::aligned_ptr<char>(mem), dev_batch_meta, batch_meta_bytes);
    }

    // static const int FRONT_SMALL = 10000000;
    static const int MIN_BATCH_COUNT = 0;
    std::vector<FM_t*> f;
    std::size_t factor_size = 0, Schur_size = 0, piv_size = 0,
      total_upd_size = 0, work_bytes, ea_bytes,
      factor_bytes, Nsmall = 0;
    std::vector<std::size_t> elems11, elems12, elems21, Isize;
    int max_d1_small = 0, max_d2_small = 0;
    char* dev_getrf_work = nullptr;
    int getrf_work_bytes = 0;
    int* dev_getrf_err = nullptr;
    std::size_t batch_meta_bytes = 0;
    gpu::HostMemory<char> dev_batch_meta;
    scalar_t** F_batch = nullptr;
    int** ipiv_batch = nullptr;
    int *d1_batch = nullptr, *d2_batch = nullptr,
      *ld1_batch = nullptr, *ld2_batch = nullptr;
    scalar_t **dev_F_batch = nullptr;
    int **dev_ipiv_batch = nullptr;
    int *dev_d1_batch = nullptr, *dev_d2_batch = nullptr,
      *dev_ld1_batch = nullptr, *dev_ld2_batch = nullptr;
  };


  template<typename scalar_t,typename integer_t>
  struct FrontMAGMA<scalar_t,integer_t>::SolveCache {
    SolveCache(int nrhs, std::size_t rows)
      : handle(comp_stream), nrhs(nrhs), rows(rows) {}
    gpu::Stream comp_stream;
    gpu::Handle handle;
    int nrhs;
    std::size_t rows;
    std::vector<LevelInfoMAGMA<scalar_t,integer_t>> ldata;
    std::vector<std::size_t> l_off;
    std::size_t Nmax = 0, Ntotal = 0, max_bupd_size = 0,
      max_work_mem = 0, Isize = 0;
    gpu::DeviceMemory<char> d_mem;
    gpu::HostMemory<char> h_mem;
    gpu::HostMemory<scalar_t> h_b;   // S-RHSPIN: pinned staging for the right-hand side / solution (rows x nrhs)
    gpu::DeviceMemory<scalar_t> d_braw;   // S-HOST: raw (unpermuted) rhs on the way in, permuted-back solution on the way out
    // S-LASWP: per-front row permutations of the forward solve (computed once per factorization)
    gpu::DeviceMemory<int> d_perm;
    gpu::DeviceMemory<int*> d_perm_ptrs;
    std::vector<int**> perm_batch;
    bool perm_ready = false;
    // Per-level batch meta-data (LevelInfoMAGMA::set_batch_data), one
    // private device region per level, so that it can be uploaded once
    // per cache instead of being re-uploaded (fwd + bwd) at every solve.
    gpu::DeviceMemory<char> d_meta;
    std::vector<std::size_t> meta_off;
    std::vector<scalar_t**> h_rhs_batch, h_bupd_batch,
      d_rhs_batch, d_bupd_batch;
    // Counters for STRUM_LAUNCH2_LOG / STRUM_LAUNCH2_VERIFY.
    std::size_t n_solves = 0, n_meta_uploads = 0, n_mem_uploads = 0;
    // S-INVDIAG: per converted level (nrhs == 1 only), per block step k the batch arrays of the block recurrence
    // (K*N entries each, step k at offset k*N): rows of block k (mL), columns of the L update (nLu) / U update (nUu),
    // and the pointers to the F11 block rows (AL / AU), the rhs block (rhs), the scratch block (scr) and the inverses.
    struct InvSolve {
      bool on = false, U = false;
      int nb = 0, nblk = 0;
      int *mL = nullptr, *nLu = nullptr, *nUu = nullptr, *ldnb = nullptr;
      scalar_t **AL = nullptr, **AU = nullptr, **rhs = nullptr, **scr = nullptr, **Li = nullptr, **Ui = nullptr;
      std::vector<int> maxLu, maxUu;
    };
    std::vector<InvSolve> inv;
    gpu::DeviceMemory<int> inv_ints;
    gpu::DeviceMemory<scalar_t*> inv_ptrs;
    gpu::DeviceMemory<scalar_t> d_scr;
    bool inv_synced = false;                 // this cache's stream has waited for the current factorization's inverses
    gpu::DeviceMemory<scalar_t> d_partial;   // fused step: N x 64 x NB partial sums
    gpu::DeviceMemory<int> d_counter;        // fused step: per-front arrival counters (self-resetting)
  };

  // S-INVDIAG (magma-hpc, 2026-10-27): the top-of-tree fronts (d1 of several hundred to a few thousand rows) solve
  // their L11 / U11 triangular systems with nrhs = 1 as a sequential chain of 32-row pipelined steps (~9 us each,
  // latency-bound, 40 % of the solve). Instead, once per factorization, the NB x NB diagonal blocks of L11 (unit
  // lower) and U11 (upper) of every front of a level whose largest front has d1 >= STRUM_SOLVE_INVDIAG_MIN are
  // inverted explicitly (MAGMA trtri_diag_vbatched gives 128-blocks; 256 / 512 are composed from them with two
  // vbatched gemms per doubling), and the solve becomes the block recurrence
  //   x_k = Linv_kk (b_k - L_{k,0:k} x_{0:k}),  k = 0..K-1   and   x_k = Uinv_kk (y_k - U_{k,k+1:} x_{k+1:}),  k = K-1..0
  // i.e. 2 vbatched gemv launches per block step over all fronts of the level (K = ceil(d1/NB) steps per side).
  // Knobs: STRUM_SOLVE_INVDIAG=0 disables everything (bit-for-bit the previous path), STRUM_SOLVE_INVDIAG_MIN
  // (default 512), STRUM_SOLVE_INVDIAG_NB (128 / 256 / 512, default 256), STRUM_SOLVE_INVDIAG_U=0 keeps the U11
  // solve on the trsm path, STRUM_INVDIAG_VERBOSE=1 prints sizes and the inversion time per factorization,
  // =2 also the max over all blocks of ||I - inv * block||_max (host check after each factorization).
  namespace invdiag {
    inline int env_int(const char* n, int def) { const char* e = std::getenv(n); return e ? std::atoi(e) : def; }
    inline bool enabled() { static const bool v = env_int("STRUM_SOLVE_INVDIAG", 1) != 0; return v; }
    inline int min_d1() { static const int v = env_int("STRUM_SOLVE_INVDIAG_MIN", 512); return v; }
    inline int nb() {
      static const int v = []() { int n = env_int("STRUM_SOLVE_INVDIAG_NB", 256);
        return (n == 128 || n == 256 || n == 512) ? n : 256; }();
      return v;
    }
    inline bool use_U() { static const bool v = env_int("STRUM_SOLVE_INVDIAG_U", 1) != 0; return v; }
    inline int verbose() { static const int v = env_int("STRUM_INVDIAG_VERBOSE", 0); return v; }
    inline bool own_stream() { static const bool v = env_int("STRUM_INVDIAG_STREAM", 1) != 0; return v; }   // v2: inversion on a second stream
    inline bool fused() { static const bool v = env_int("STRUM_SOLVE_INVDIAG_FUSED", 1) != 0; return v; }      // v2: one kernel per step
    inline int chunk() { static const int v = std::max(16, env_int("STRUM_SOLVE_INVDIAG_CHUNK", 64)); return v; }
    inline bool nvtx() { static const bool v = env_int("STRUM_SOLVE_NVTX", 0) != 0; return v; }
    inline bool prefetch() { static const bool v = env_int("STRUM_SOLVE_INVDIAG_PREFETCH", 0) != 0; return v; }   // v2 trace: +0.14 ms with the single-CTA apply; re-measured with v2b
    struct Range { bool on; Range(const char* s) : on(nvtx()) { if (on) nvtxRangePushA(s); } ~Range() { if (on) nvtxRangePop(); } };
  }

  // S-HOST: device copy of the sparse reordering permutation, cached across solves. Rebuilt after every
  // factorization (devperm_ is reset there; a changed reordering always implies a new factorization before the
  // next solve) and whenever the host vectors move or change size.
  template<typename scalar_t,typename integer_t>
  struct FrontMAGMA<scalar_t,integer_t>::DevPerm {
    const integer_t* key_iperm = nullptr;
    const integer_t* key_perm = nullptr;
    std::size_t n = 0;
    gpu::DeviceMemory<int> d_iperm, d_perm;
  };

  // STRUM_LAUNCH2_LOG=1: print one line per solve telling whether the
  // meta-data upload was done (cache miss) or skipped (cache hit).
  // STRUM_LAUNCH2_VERIFY=1: additionally read the cached device
  // meta-data back after every solve and compare it with the host copy.
  static bool launch2_env(const char* name) {
    const char* e = std::getenv(name);
    return e && e[0] && e[0] != '0';
  }

  template<typename scalar_t,typename integer_t>
  FrontMAGMA<scalar_t,integer_t>::FrontMAGMA
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t>
  FrontMAGMA<scalar_t,integer_t>::~FrontMAGMA() {
#if defined(STRUMPACK_COUNT_FLOPS)
    const std::size_t dupd = dim_upd();
    const std::size_t dsep = dim_sep();
    STRUMPACK_SUB_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
#endif
  }

  template<typename scalar_t,typename integer_t> scalar_t*
  FrontMAGMA<scalar_t,integer_t>::get_device_F22(scalar_t* dF22) {
    return F22_.data();
  }

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::
  release_work_memory(VectorPool<scalar_t>& workspace) {
    if (dev_Schur_) workspace.restore(*dev_Schur_);
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontMPI<scalar_t,integer_t>* pa) const {
    DenseM_t F22(dim_upd(), dim_upd());
    gpu::copy_device_to_host(F22, F22_);
    ExtendAdd<scalar_t,integer_t>::
      extend_add_seq_copy_to_buffers(F22, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::extadd_blr_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontBLRMPI<scalar_t,integer_t>* pa) const {
    DenseM_t F22(dim_upd(), dim_upd());
    gpu::copy_device_to_host(F22, F22_);
    BLR::BLRExtendAdd<scalar_t,integer_t>::
      seq_copy_to_buffers(F22, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::extadd_blr_copy_to_buffers_col
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontBLRMPI<scalar_t,integer_t>* pa,
   integer_t begin_col, integer_t end_col, const Opts_t& opts) const {
    DenseM_t F22(dim_upd(), dim_upd());
    gpu::copy_device_to_host(F22, F22_);
    BLR::BLRExtendAdd<scalar_t,integer_t>::
      seq_copy_to_buffers_col(F22, sbuf, pa, this, begin_col, end_col);
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, VectorPool<scalar_t>& workspace, int task_depth) {
    const std::size_t dupd = dim_upd();
    DenseM_t F22(dupd, dupd);
    gpu::copy_device_to_host(F22, F22_);
    this->extend_add(paF11, paF12, paF21, paF22, F22, p);
    release_work_memory(workspace);
  }

  template<typename scalar_t, typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::front_assembly
  (const SpMat_t& A, LInfo_t& L, char* hea_mem, char* dea_mem) {
    using Trip_t = Triplet<scalar_t>;
    auto N = L.f.size();
    auto hasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(hea_mem);
    auto Iptr   = gpu::aligned_ptr<std::size_t>(hasmbl + N);
    auto e11    = gpu::aligned_ptr<Trip_t>(Iptr + L.Isize.back());
    auto e12    = e11 + L.elems11.back();
    auto e21    = e12 + L.elems12.back();
    auto dasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(dea_mem);
    auto dIptr  = gpu::aligned_ptr<std::size_t>(dasmbl + N);
    auto de11   = gpu::aligned_ptr<Trip_t>(dIptr + L.Isize.back());
    auto de12   = de11 + L.elems11.back();
    auto de21   = de12 + L.elems12.back();

#pragma omp parallel for
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      A.set_front_elements
        (f.sep_begin_, f.sep_end_, f.upd_,
         e11+L.elems11[n], e12+L.elems12[n], e21+L.elems21[n]);
      hasmbl[n] = gpu::AssembleData<scalar_t>
        (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
         f.F21_.data(), f.F22_.data(),
         L.elems11[n+1]-L.elems11[n], L.elems12[n+1]-L.elems12[n],
         L.elems21[n+1]-L.elems21[n],
         de11+L.elems11[n], de12+L.elems12[n], de21+L.elems21[n]);
      auto fIptr = Iptr + L.Isize[n];
      auto fdIptr = dIptr + L.Isize[n];
      if (f.lchild_) {
        auto c = dynamic_cast<FM_t*>(f.lchild_.get());
        hasmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
        fIptr += c->dim_upd();
        fdIptr += c->dim_upd();
      }
      if (f.rchild_) {
        auto c = dynamic_cast<FM_t*>(f.rchild_.get());
        hasmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
      }
    }
    gpu::copy_host_to_device<char>(dea_mem, hea_mem, L.ea_bytes);
    gpu::assemble(N, hasmbl, dasmbl);
  }

  // F-GATHER (magma-hpc F track, 2026-10-27): factorization structure cache.
  // The sparse-matrix entries of every front were extracted on the host into
  // (row,col,value) triplets and copied to the device at EVERY factorization
  // (918 MB for the JOREK proxy), and the level meta-data (sizes, batch
  // pointers, assembly descriptors) was rebuilt each time. A refactorization
  // with the same sparsity (JOREK refactorizes its preconditioner every few
  // time steps, same structure) now reuses all of it: the device buffers for
  // factors/work/meta-data stay allocated, so every device pointer in the
  // cached meta-data remains valid, and a per-level gather map
  // (dst offset into the factor memory, src index into the CSR values) built
  // once scatters the new values with one kernel per level from a single
  // device copy of A.val(). STRUM_FACTOR_CACHE=0 restores the old path.

  // F-TOPLU (magma-hpc F track): GPU-only blocked LU of ONE large front through MAGMA's native path
  // (GPU panel kernels + open magmablas trsm/gemm in our fork). Used for levels whose fronts are taller
  // than the fused batched panel can handle (the vbatched path then falls back to one idamax/swap/ger
  // launch triple per column: 8448 launches for the 3200-row root of the JOREK proxy).
  static inline void large_getrf_native(magma_int_t n, float* A, magma_int_t ld, magma_int_t* piv, magma_int_t* info)
  { magma_sgetrf_native(n, n, A, ld, piv, info); }
  static inline void large_getrf_native(magma_int_t n, double* A, magma_int_t ld, magma_int_t* piv, magma_int_t* info)
  { magma_dgetrf_native(n, n, A, ld, piv, info); }
  static inline void large_getrf_native(magma_int_t n, std::complex<float>* A, magma_int_t ld, magma_int_t* piv, magma_int_t* info)
  { magma_cgetrf_native(n, n, reinterpret_cast<magmaFloatComplex*>(A), ld, piv, info); }
  static inline void large_getrf_native(magma_int_t n, std::complex<double>* A, magma_int_t ld, magma_int_t* piv, magma_int_t* info)
  { magma_zgetrf_native(n, n, reinterpret_cast<magmaDoubleComplex*>(A), ld, piv, info); }


  // F-GEMM-TOP (magma-hpc F track): Schur update of ONE large front with the fixed-size open magmablas gemm kernel
  // (90-98 % of the FP64 peak at these sizes) instead of the vbatched kernel (60-85 % on the top levels).
  static inline void large_gemm(magma_int_t m, magma_int_t n, magma_int_t k, const float* A, magma_int_t lda, const float* B, magma_int_t ldb, float* C, magma_int_t ldc, magma_queue_t q)
  { magmablas_sgemm(MagmaNoTrans, MagmaNoTrans, m, n, k, -1.f, A, lda, B, ldb, 1.f, C, ldc, q); }
  static inline void large_gemm(magma_int_t m, magma_int_t n, magma_int_t k, const double* A, magma_int_t lda, const double* B, magma_int_t ldb, double* C, magma_int_t ldc, magma_queue_t q)
  { magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, m, n, k, -1., A, lda, B, ldb, 1., C, ldc, q); }
  static inline void large_gemm(magma_int_t m, magma_int_t n, magma_int_t k, const std::complex<float>* A, magma_int_t lda, const std::complex<float>* B, magma_int_t ldb, std::complex<float>* C, magma_int_t ldc, magma_queue_t q)
  { magmablas_cgemm(MagmaNoTrans, MagmaNoTrans, m, n, k, MAGMA_C_MAKE(-1.f, 0.f), reinterpret_cast<const magmaFloatComplex*>(A), lda, reinterpret_cast<const magmaFloatComplex*>(B), ldb, MAGMA_C_ONE, reinterpret_cast<magmaFloatComplex*>(C), ldc, q); }
  static inline void large_gemm(magma_int_t m, magma_int_t n, magma_int_t k, const std::complex<double>* A, magma_int_t lda, const std::complex<double>* B, magma_int_t ldb, std::complex<double>* C, magma_int_t ldc, magma_queue_t q)
  { magmablas_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, k, MAGMA_Z_MAKE(-1., 0.), reinterpret_cast<const magmaDoubleComplex*>(A), lda, reinterpret_cast<const magmaDoubleComplex*>(B), ldb, MAGMA_Z_ONE, reinterpret_cast<magmaDoubleComplex*>(C), ldc, q); }

  template<typename scalar_t,typename integer_t>
  struct FrontMAGMA<scalar_t,integer_t>::FactorCache {
    std::size_t N = 0, nnz = 0;
    int lvls = 0;
    std::vector<LevelInfoMAGMA<scalar_t,integer_t>> ldata;
    std::size_t peak_work_dmem = 0, total_factor_dmem = 0, nmap = 0;
    gpu::DeviceMemory<char> work_dmem;   // Schur complements + getrf work (two parity halves)
    gpu::DeviceMemory<char> meta_dmem;   // per level: batch meta | AssembleData | Iptr (uploaded once)
    gpu::HostMemory<char> meta_hmem;     // host mirror (gpu::assemble reads the host AssembleData)
    std::vector<char*> level_factors, level_work;
    std::vector<gpu::AssembleData<scalar_t>*> h_asmbl, d_asmbl;
    gpu::DeviceMemory<scalar_t> d_vals;  // A.val() on the device, refreshed every factorization (host-copy path only)
    gpu::DeviceMemory<int> d_src;        // gather map, all levels concatenated
    gpu::DeviceMemory<unsigned> d_dst;
    std::vector<std::size_t> map_off;    // lvls+1 offsets into d_src/d_dst
    // F-VALCOPY: the CSR value array is page-locked in place (cudaHostRegister) so the 455 MB per-refactorization
    // upload runs at PCIe speed instead of through the driver's pageable staging; registration is kept while the
    // same host buffer (STRUMPACK keeps the permuted matrix across value updates) is passed again.
    // v2: cudaHostRegister of A.val() in place was WRONG (window 103 arm V): STRUMPACK replaces the matrix
    // buffer on a structure-changing update, the registration then maps freed pages and the next upload reads
    // stale memory. Now: a STRUMPACK-owned pinned staging buffer, filled chunk by chunk (OpenMP memcpy) while the
    // previous chunk's async DMA runs on the compute stream.
    gpu::HostMemory<scalar_t> h_vals;    // pinned staging, nnz values
    // S-INVDIAG: per level the explicit inverses of the NB x NB diagonal blocks of L11 / U11 (persistent, refilled
    // every factorization) and the batch meta of their construction.
    struct InvRound {   // composition s -> 2s of one side, one entry per 2s-block of the level
      int s = 0, nblk = 0, max_rB = 0;
      int *m_rB = nullptr, *c_s = nullptr, *c_2s = nullptr, *c_ld = nullptr, *c_sB = nullptr;
      scalar_t **a = nullptr, **b = nullptr, **oa = nullptr, **ob = nullptr, **M = nullptr, **T = nullptr, **oX = nullptr;
    };
    struct InvLevel {
      bool on = false;
      int nb = 0, nblk_max = 0;
      std::size_t side_elems = 0, scratch_elems = 0;
      gpu::DeviceMemory<scalar_t> d_inv;    // [ L blocks | U blocks ], final layout: block k of front i at inv_i + k*nb*nb, ld nb
      gpu::DeviceMemory<int> d_ints;
      gpu::DeviceMemory<scalar_t*> d_ptrs;
      std::vector<scalar_t*> h128[2];            // per side, per front: trtri_diag output (128 layout; host copy)
      std::vector<InvRound> rounds[2];
    };
    std::vector<InvLevel> inv;
    gpu::DeviceMemory<scalar_t> inv_scratch;   // intermediate layouts + gemm temporaries (max over levels)
    // v2: the inversion of a level runs on its own (non-blocking) stream right after the level's LU, overlapping the
    // level's trsm / Schur gemm on the compute stream; inv_ready[l] = LU done (compute -> inv stream), inv_done[l] =
    // inverses ready (inv stream -> the solve's stream and the next factorization's compute stream).
    gpu::Stream inv_stream;
    gpu::Handle inv_handle;
    std::vector<std::unique_ptr<gpu::Event>> inv_ready, inv_done;
    std::vector<cudaEvent_t> inv_t0, inv_t1;   // verbose timing per level (on the inversion stream)
    std::vector<float> inv_level_ms;
    FactorCache() : inv_stream(true), inv_handle(inv_stream) {}
    ~FactorCache() { for (auto e : inv_t0) if (e) cudaEventDestroy(e); for (auto e : inv_t1) if (e) cudaEventDestroy(e); }
    double inv_ms = 0.;                         // inversion time of the last factorization (verbose)
    double inv_resid = 0.;                      // max ||I - inv * block|| of the last factorization (verbose >= 2)
    std::size_t inv_bytes = 0;
  };

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontMAGMA<scalar_t,integer_t>::factors_on_device
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata,
   std::size_t total_dmem) {
    static const bool use_cache = !(std::getenv("STRUM_FACTOR_CACHE") &&
                                    std::atoi(std::getenv("STRUM_FACTOR_CACHE")) == 0);
    const bool reuse = factor_cache_ && factor_cache_->N == A.size() &&
      factor_cache_->nnz == A.nnz();
    if (!use_cache || (!reuse && ldata.empty()))
      return factors_on_device_legacy(A, opts, ldata, total_dmem);
    if (!reuse) {
      // structure changed (or first factorization): the gather map needs
      // 32-bit element offsets and 32-bit CSR indices
      std::size_t total_factor_dmem = 0;
      for (auto& L : ldata) total_factor_dmem += L.factor_bytes;
      if (total_factor_dmem / sizeof(scalar_t) >= (std::size_t(1) << 32) ||
          std::size_t(A.nnz()) >= (std::size_t(1) << 31)) {
        if (opts.verbose())
          std::cout << "# F-GATHER: problem too large for 32-bit maps, legacy path" << std::endl;
        factor_cache_.reset();
        return factors_on_device_legacy(A, opts, ldata, total_dmem);
      }
      build_factor_cache(A, ldata);
      solve_cache_.reset();   // S-INVDIAG: new inverse buffers / front pointers -> the solve workspace is rebuilt
    }
    ReturnCode err_code = ReturnCode::SUCCESS;
    static const bool pack_verbose = std::getenv("STRUM_DEVUPDATE_VERBOSE") && std::atoi(std::getenv("STRUM_DEVUPDATE_VERBOSE")) != 0;
    const auto t_f0 = std::chrono::steady_clock::now();
    static const bool keep_solve_cache = !(std::getenv("STRUM_SOLVE_CACHE_KEEP") &&
                                           std::atoi(std::getenv("STRUM_SOLVE_CACHE_KEEP")) == 0);
    if (keep_solve_cache && solve_cache_) {
      // S-CACHE-KEEP (magma-hpc F track): same structure and the same persistent factor memory, so the solve
      // workspace, its meta-data and the device permutation stay valid; only the per-front pivot permutations
      // are rebuilt at the next solve. This also removes the cudaFree/cudaFreeHost here, whose implicit device
      // synchronization would wait for the asynchronous value upload of F-DEVUPDATE.
      solve_cache_->perm_ready = false;
      solve_cache_->inv_synced = false;   // S-INVDIAG: wait for this factorization's inversions at the next solve
    } else {
      solve_cache_.reset(); // new factors: rebuild the solve workspace
      devperm_.reset();     // S-HOST: and the device permutation
    }
    const auto t_f1 = std::chrono::steady_clock::now();
    auto& C = *factor_cache_;
    gpu::Stream comp_stream;
    gpu::Handle handle(comp_stream);
    if (pack_verbose) {   // item-5 audit: host time before the values are touched
      const auto t_f2 = std::chrono::steady_clock::now();
      std::cout << "# F-PACK factor_start solve_cache_reset_ms "
                << std::chrono::duration<double,std::milli>(t_f1 - t_f0).count()
                << " stream_handle_create_ms " << std::chrono::duration<double,std::milli>(t_f2 - t_f1).count() << std::endl;
    }
    // F-DEVUPDATE: when the value update ran on the device, the permuted values are already there
    // (the source waits for its upload and verifies it); nothing is copied at the start of the factorization
    const scalar_t* vals = nullptr;
    DeviceValueSource<scalar_t>* dsrc = A.device_value_source();
    unsigned long long dgen = 0;
    if (dsrc) vals = dsrc->device_values(C.nnz, &dgen);
    if (vals) {
      C.d_vals.release();   // the host-copy buffers are not needed while the device source is current
      C.h_vals.release();
      if (opts.verbose())
        std::cout << "# F-DEVUPDATE: CSR values taken from the device (generation " << dgen
                  << "), no host copy" << std::endl;
    } else {
      dsrc = nullptr;
      if (C.d_vals.size() < C.nnz) C.d_vals = gpu::DeviceMemory<scalar_t>(C.nnz);
      vals = C.d_vals;
      static const bool pin = !(std::getenv("STRUM_PIN_VALUES") &&
                                std::atoi(std::getenv("STRUM_PIN_VALUES")) == 0);
      if (pin) {
        if (C.h_vals.size() < C.nnz) C.h_vals = gpu::HostMemory<scalar_t>(C.nnz);
        const scalar_t* src = A.val();
        scalar_t* stg = C.h_vals;
        scalar_t* dst = C.d_vals;
        const std::size_t nchunk = 8, chunk = (C.nnz + nchunk - 1) / nchunk;
        for (std::size_t c=0; c<nchunk; c++) {
          const std::size_t lo = c*chunk, hi = std::min(C.nnz, lo+chunk);
          if (lo >= hi) break;
#pragma omp parallel for
          for (std::size_t k=lo; k<hi; k++) stg[k] = src[k];
          gpu::copy_host_to_device_async<scalar_t>(dst+lo, stg+lo, hi-lo, comp_stream);
        }
        comp_stream.synchronize();   // the gathers below run on the default stream
      } else
        gpu::copy_host_to_device<scalar_t>(C.d_vals, A.val(), C.nnz);
    }
    scalar_t* fbase = dev_factors_->template as<scalar_t>();
    C.inv_ms = 0.; C.inv_resid = 0.;
    // S-INVDIAG v2: the previous factorization's inversions must be done before F11 is overwritten (the memsets /
    // gathers below run on the legacy default stream, which is ordered after comp_stream)
    for (int l=0; l<int(C.inv.size()); l++) if (C.inv[l].on && C.inv_done[l]) C.inv_done[l]->wait(comp_stream);
    for (int l=C.lvls-1; l>=0; l--) {
      auto& L = C.ldata[l];
      try {
        gpu::memset<scalar_t>(C.level_work[l], 0, L.Schur_size);
        gpu::memset<scalar_t>(C.level_factors[l], 0, L.factor_size);
        // default stream: gather of this level's sparse entries, then extend-add
        gpu::gather_front_elements<scalar_t>
          (C.map_off[l+1]-C.map_off[l], C.d_src + C.map_off[l],
           C.d_dst + C.map_off[l], vals, fbase);
        gpu::assemble(L.f.size(), C.h_asmbl[l], C.d_asmbl[l]);

        std::size_t N = L.f.size(), Nsmall = L.Nsmall;
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F11 = L.dev_F_batch;   auto F12 = F11 + Nsmall;
        auto F21 = F12 + Nsmall;    auto F22 = F21 + Nsmall;

        static const int large_lu = std::getenv("STRUM_LARGE_LU") ?
          std::atoi(std::getenv("STRUM_LARGE_LU")) : 0;   // default OFF (window 102: +3.4 %, NO-GO); 0 = off
        if (large_lu > 0 && L.max_d1_small > large_lu) {
          // F-TOPLU: every front of this level individually (levels with such fronts hold 1-8 fronts)
          std::vector<magma_int_t> piv;
          for (std::size_t i=0; i<N; i++) {
            auto& F11m = L.f[i]->F11_;
            const magma_int_t n = F11m.rows();
            if (!n) continue;
            magma_int_t info = 0;
            piv.resize(n);
            large_getrf_native(n, F11m.data(), F11m.ld(), piv.data(), &info);
            gpu::copy_host_to_device<int>(L.f[i]->piv_, piv.data(), n);
          }
        } else
        getrf_vbatched_chunked
          (d1, L.max_d1_small, F11, ld1, L.dev_ipiv_batch,
           L.dev_getrf_err, L.dev_getrf_work, &L.getrf_work_bytes,
           Nsmall, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrf
            (handle, L.f[i]->F11_, (scalar_t*)L.dev_getrf_work,
             L.getrf_work_bytes / sizeof(scalar_t),
             L.f[i]->piv_, L.dev_getrf_err+i);
        if (opts.replace_tiny_pivots()) {
          gpu::replace_pivots_vbatched
            (handle, d1, L.max_d1_small, F11, ld1,
             opts.pivot_threshold(), Nsmall);
          for (std::size_t i=Nsmall; i<N; i++)
            gpu::replace_pivots
              (L.f[i]->F11_.rows(), L.f[i]->F11_.data(),
               opts.pivot_threshold());
        }
        if (l < int(C.inv.size()) && C.inv[l].on) {   // S-INVDIAG: F11 is final here
          if (invdiag::own_stream()) {
            C.inv_ready[l]->record(comp_stream);
            C.inv_ready[l]->wait(C.inv_stream);
            invdiag_build(l, C.inv_handle);
            C.inv_done[l]->record(C.inv_stream);
          } else {
            invdiag_build(l, handle);
            C.inv_done[l]->record(comp_stream);
          }
        }
        gpu::laswp_fwd_vbatched
          (handle, d2, L.max_d2_small, F12, ld1, L.dev_ipiv_batch,
           d1, Nsmall);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, handle);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrs
            (handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_,
             L.f[i]->F12_, L.dev_getrf_err+i);
        static const int large_gemm_d2 = std::getenv("STRUM_LARGE_GEMM") ?
          std::atoi(std::getenv("STRUM_LARGE_GEMM")) : 0;   // default OFF (window 102: +4.2 %, NO-GO); 0 = off
        if (large_gemm_d2 > 0 && L.max_d2_small > large_gemm_d2) {
          // F-GEMM-TOP: one fixed-size gemm launch per front (levels with such fronts hold 1-250 fronts)
          auto q = gpu::get_magma_queue(handle);
          for (std::size_t i=0; i<Nsmall; i++) {
            auto& f = *(L.f[i]);
            const magma_int_t dsep = f.dim_sep(), dupd = f.dim_upd();
            if (!dsep || !dupd) continue;
            large_gemm(dupd, dupd, dsep, f.F21_.data(), f.F21_.ld(), f.F12_.data(), f.F12_.ld(),
                       f.F22_.data(), f.F22_.ld(), q);
          }
        } else
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans, d2, d2, d1, scalar_t(-1.),
           F21, ld2, F12, ld1, scalar_t(1.), F22, ld2, Nsmall,
           L.max_d2_small, L.max_d2_small, L.max_d1_small, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::gemm
            (handle, Trans::N, Trans::N, scalar_t(-1.),
             L.f[i]->F21_, L.f[i]->F12_, scalar_t(1.), L.f[i]->F22_);
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      long long level_flops, small_flops;
      L.flops(level_flops, small_flops);
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      STRUMPACK_FLOPS(small_flops);
    }
    {
      // the solve's (blocking) stream and the F-DEVUPDATE write-back must follow the factorization work issued
      // on comp_stream: an event wait on the legacy default stream is a barrier for every blocking stream
      cudaEvent_t ev;
      gpu_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
      gpu_check(cudaEventRecord(ev, gpu::get_cuda_stream(comp_stream)));
      gpu_check(cudaStreamWaitEvent(0, ev, 0));
      gpu_check(cudaEventDestroy(ev));
    }
    if (dsrc) dsrc->factor_enqueued();   // F-DEVUPDATE: host write-back now, overlapping the kernels above
    if (invdiag::verbose() && C.inv_bytes) {
      std::cout << "# INVDIAG factorization";
      for (int l=0; l<C.lvls; l++) {
        if (!C.inv[l].on) continue;
        cudaEventSynchronize(C.inv_t1[l]);
        float ms = 0.f; cudaEventElapsedTime(&ms, C.inv_t0[l], C.inv_t1[l]);
        C.inv_ms += ms;
        std::cout << " l" << l << "_ms " << ms;
      }
      std::cout << " inversion_ms " << C.inv_ms << " inv_MB " << C.inv_bytes / 1e6
                << " stream " << (invdiag::own_stream() ? "own" : "compute");
      if (invdiag::verbose() >= 2) std::cout << " max_resid " << C.inv_resid;
      std::cout << std::endl;
    }
    return err_code;
  }

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::build_factor_cache
  (const SpMat_t& A, std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata) {
    factor_cache_.reset(new FactorCache());
    auto& C = *factor_cache_;
    C.N = A.size(); C.nnz = A.nnz(); C.lvls = ldata.size();
    C.ldata = std::move(ldata);
    const int lvls = C.lvls;
    std::size_t meta_bytes = 0;
    C.map_off.assign(lvls+1, 0);
    for (int l=0; l<lvls; l++) {
      auto& L = C.ldata[l];
      std::size_t level_mem = L.work_bytes;
      if (l+1 < lvls) level_mem += C.ldata[l+1].work_bytes;
      C.peak_work_dmem = std::max(C.peak_work_dmem, level_mem);
      C.total_factor_dmem += L.factor_bytes;
      meta_bytes += L.batch_meta_bytes;
      meta_bytes += gpu::round_up(sizeof(gpu::AssembleData<scalar_t>)*L.f.size());
      meta_bytes += gpu::round_up(sizeof(std::size_t)*L.Isize.back());
      C.map_off[l+1] = C.map_off[l] +
        L.elems11.back() + L.elems12.back() + L.elems21.back();
    }
    C.nmap = C.map_off[lvls];
    C.work_dmem = gpu::DeviceMemory<char>(C.peak_work_dmem);
    dev_factors_.reset(new gpu::DeviceMemory<char>(C.total_factor_dmem));
    C.meta_dmem = gpu::DeviceMemory<char>(meta_bytes);
    C.meta_hmem = gpu::HostMemory<char>(meta_bytes);
    // C.d_vals (A.val() on the device) is allocated on first use: not needed when the values come from the device
    C.level_factors.assign(lvls, nullptr); C.level_work.assign(lvls, nullptr);
    C.h_asmbl.assign(lvls, nullptr); C.d_asmbl.assign(lvls, nullptr);
    std::vector<int> src(C.nmap);
    std::vector<unsigned> dst(C.nmap);
    char* factors_dptr = dev_factors_->template as<char>();
    const scalar_t* fbase = dev_factors_->template as<scalar_t>();
    char* meta_d = C.meta_dmem;
    char* meta_h = C.meta_hmem;
    const integer_t* Aptr = A.ptr();
    const integer_t* Aind = A.ind();
    for (int l=lvls-1; l>=0; l--) {
      auto& L = C.ldata[l];
      char* work_dptr = (l % 2) ? C.work_dmem.template as<char>()
        : C.work_dmem.template as<char>() + C.peak_work_dmem - L.work_bytes;
      C.level_work[l] = work_dptr;
      C.level_factors[l] = factors_dptr;
      L.set_factor_pointers(factors_dptr);
      factors_dptr += L.factor_bytes;
      L.set_work_pointers(work_dptr, meta_d);   // uploads the batch meta into the persistent slot
      meta_d += L.batch_meta_bytes; meta_h += L.batch_meta_bytes;
      const std::size_t N = L.f.size();
      auto hasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(meta_h);
      auto dasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(meta_d);
      auto Iptr   = gpu::aligned_ptr<std::size_t>(hasmbl + N);
      auto dIptr  = gpu::aligned_ptr<std::size_t>(dasmbl + N);
      const std::size_t asm_bytes =
        gpu::round_up(sizeof(gpu::AssembleData<scalar_t>)*N) +
        gpu::round_up(sizeof(std::size_t)*L.Isize.back());
      C.h_asmbl[l] = hasmbl; C.d_asmbl[l] = dasmbl;
      const std::size_t off = C.map_off[l],
        E11 = L.elems11.back(), E12 = L.elems12.back();
#pragma omp parallel for
      for (std::size_t n=0; n<N; n++) {
        auto& f = *(L.f[n]);
        hasmbl[n] = gpu::AssembleData<scalar_t>
          (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
           f.F21_.data(), f.F22_.data(), 0, 0, 0, nullptr, nullptr, nullptr);
        auto fIptr = Iptr + L.Isize[n];
        auto fdIptr = dIptr + L.Isize[n];
        if (f.lchild_) {
          auto c = dynamic_cast<FM_t*>(f.lchild_.get());
          hasmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), fdIptr);
          c->upd_to_parent(&f, fIptr);
          fIptr += c->dim_upd();
          fdIptr += c->dim_upd();
        }
        if (f.rchild_) {
          auto c = dynamic_cast<FM_t*>(f.rchild_.get());
          hasmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), fdIptr);
          c->upd_to_parent(&f, fIptr);
        }
        // gather map of this front (same walk as CSRMatrix::set_front_elements)
        const integer_t slo = f.sep_begin(), shi = f.sep_end();
        const auto& upd = f.upd();
        const integer_t ds = shi - slo, du = upd.size();
        const unsigned o11 = f.F11_.data() - fbase, o12 = f.F12_.data() - fbase,
          o21 = f.F21_.data() - fbase;
        int* s11 = src.data() + off + L.elems11[n];
        int* s12 = src.data() + off + E11 + L.elems12[n];
        int* s21 = src.data() + off + E11 + E12 + L.elems21[n];
        unsigned* t11 = dst.data() + off + L.elems11[n];
        unsigned* t12 = dst.data() + off + E11 + L.elems12[n];
        unsigned* t21 = dst.data() + off + E11 + E12 + L.elems21[n];
        for (integer_t row=0; row<ds; row++) {
          integer_t upd_ptr = 0;
          const auto hij = Aptr[row+slo+1];
          for (integer_t j=Aptr[row+slo]; j<hij; j++) {
            const integer_t col = Aind[j];
            if (col >= slo) {
              if (col < shi) { *s11++ = j; *t11++ = o11 + row + (col-slo)*ds; }
              else {
                while (upd_ptr<du && upd[upd_ptr]<col) upd_ptr++;
                if (upd_ptr == du) break;
                if (upd[upd_ptr] == col) { *s12++ = j; *t12++ = o12 + row + upd_ptr*ds; }
              }
            }
          }
        }
        for (integer_t i=0; i<du; i++) {
          const auto row = upd[i];
          const auto hij = Aptr[row+1];
          for (integer_t j=Aptr[row]; j<hij; j++) {
            const integer_t col = Aind[j];
            if (col >= slo) {
              if (col < shi) { *s21++ = j; *t21++ = o21 + i + (col-slo)*du; }
              else break;
            }
          }
        }
      }
      gpu::copy_host_to_device<char>(reinterpret_cast<char*>(dasmbl),
                                     reinterpret_cast<char*>(hasmbl), asm_bytes);
      meta_d += asm_bytes; meta_h += asm_bytes;
    }
    C.d_src = gpu::DeviceMemory<int>(C.nmap);
    C.d_dst = gpu::DeviceMemory<unsigned>(C.nmap);
    gpu::copy_host_to_device<int>(C.d_src, src.data(), C.nmap);
    gpu::copy_host_to_device<unsigned>(C.d_dst, dst.data(), C.nmap);
    invdiag_plan(C.ldata);   // S-INVDIAG
  }

  // S-INVDIAG: decide the converted levels, allocate the inverse buffers and build the (structure-only) batch
  // meta of the inversion: the trtri_diag_vbatched output pointers and, for nb > 128, per doubling s -> 2s and per
  // 2s-block the lacpy / gemm arrays of   inv([A 0; C B]) = [Ai 0; -Bi C Ai, Bi]   (L, unit lower) and
  // inv([A B; 0 D]) = [Ai, -Ai B Di; 0, Di]   (U), with the s-inverses Ai / Bi(Di) taken from the previous layout.
  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::invdiag_plan(std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata) {
    auto& C = *factor_cache_;
    const int lvls = C.lvls;
    C.inv.clear(); C.inv.resize(lvls);
    C.inv_scratch.release(); C.inv_bytes = 0;
    for (int l=0; l<lvls; l++)
      for (auto f : ldata[l].f) { f->invL_ = nullptr; f->invU_ = nullptr; }
    if (!invdiag::enabled()) return;
    const int NB = invdiag::nb(), mind1 = invdiag::min_d1();
    auto blocks = [](int d1, int s) { return std::size_t((d1 + s - 1) / s); };
    std::size_t scratch_max = 0;
    for (int l=0; l<lvls; l++) {
      auto& L = ldata[l]; auto& V = C.inv[l];
      if (!L.Nsmall || L.Nsmall != L.f.size() || L.max_d1_small < mind1 || L.max_d1_small <= 0) continue;
      V.on = true; V.nb = NB;
      std::size_t fin = 0, scr = 0, tmax = 0;
      for (int s=128; s<=NB; s*=2) {
        std::size_t e = 0;
        for (auto f : L.f) e += blocks(f->dim_sep(), s) * s * s;
        if (s == NB) fin = e; else scr += 2 * e;
        if (s < NB) { std::size_t t = 0; for (auto f : L.f) t += blocks(f->dim_sep(), 2*s) * s * s; tmax = std::max(tmax, t); }
      }
      V.side_elems = fin; V.scratch_elems = scr + tmax;
      V.d_inv = gpu::DeviceMemory<scalar_t>(2 * fin);
      C.inv_bytes += 2 * fin * sizeof(scalar_t);
      scratch_max = std::max(scratch_max, V.scratch_elems);
      for (auto f : L.f) V.nblk_max = std::max<int>(V.nblk_max, blocks(f->dim_sep(), NB));
    }
    if (scratch_max) C.inv_scratch = gpu::DeviceMemory<scalar_t>(scratch_max);
    C.inv_ready.clear(); C.inv_done.clear(); C.inv_ready.resize(lvls); C.inv_done.resize(lvls);
    C.inv_t0.assign(lvls, nullptr); C.inv_t1.assign(lvls, nullptr); C.inv_level_ms.assign(lvls, 0.f);
    for (int l=0; l<lvls; l++) {
      auto& V = C.inv[l];
      if (!V.on) continue;
      C.inv_ready[l].reset(new gpu::Event()); C.inv_done[l].reset(new gpu::Event());
      cudaEventCreate(&C.inv_t0[l]); cudaEventCreate(&C.inv_t1[l]);
      auto& L = ldata[l];
      const std::size_t N = L.f.size();
      std::vector<int> ints;
      std::vector<scalar_t*> ptrs;
      // layout base pointers: index li = log2(s/128); the final layout lives in d_inv, the others in the scratch
      scalar_t* scr = C.inv_scratch;
      scalar_t* base[2][4] = {{nullptr}};
      int nlay = 0;
      for (int side=0; side<2; side++) {
        int li = 0;
        for (int s=128; s<=NB; s*=2, li++) {
          if (s == NB) base[side][li] = V.d_inv.template as<scalar_t>() + side * V.side_elems;
          else {
            base[side][li] = scr;
            std::size_t e = 0;
            for (auto f : L.f) e += blocks(f->dim_sep(), s) * s * s;
            scr += e;
          }
        }
        nlay = li;
      }
      scalar_t* Tbase = scr;   // gemm temporaries (rounds and sides run one after the other)
      std::vector<std::size_t> off[4];   // per layout: element offset of front i
      {
        int li = 0;
        for (int s=128; s<=NB; s*=2, li++) {
          off[li].assign(N+1, 0);
          for (std::size_t i=0; i<N; i++) off[li][i+1] = off[li][i] + blocks(L.f[i]->dim_sep(), s) * s * s;
        }
      }
      for (std::size_t i=0; i<N; i++) {
        L.f[i]->invL_ = base[0][nlay-1] + off[nlay-1][i];
        L.f[i]->invU_ = base[1][nlay-1] + off[nlay-1][i];
      }
      for (int side=0; side<2; side++) {
        V.h128[side].resize(N);
        for (std::size_t i=0; i<N; i++) V.h128[side][i] = base[side][0] + off[0][i];
      }
      struct RO { std::size_t m_rB, c_s, c_2s, c_ld, c_sB, a, b, oa, ob, M, T, oX; int s, nblk, max_rB; };
      std::vector<RO> ro[2];
      for (int side=0; side<2; side++) {
        int li = 0;
        for (int s=128; s<NB; s*=2, li++) {
          const int s2 = 2 * s;
          std::vector<int> m_rB, c_ld, c_sB;
          std::vector<scalar_t*> a, b, oa, ob, M, T, oX;
          std::size_t j = 0;
          for (std::size_t i=0; i<N; i++) {
            auto& f = *(L.f[i]);
            const int d1 = f.dim_sep(), ld = f.F11_.ld();
            scalar_t* F = f.F11_.data();
            scalar_t* in = base[side][li] + off[li][i];
            scalar_t* out = base[side][li+1] + off[li+1][i];
            const int nJ = blocks(d1, s2);
            for (int J=0; J<nJ; J++, j++) {
              const int r = std::min(s2, d1 - s2*J), rB = std::max(0, r - s);
              scalar_t* ain = in + std::size_t(2*J) * s * s;
              scalar_t* bin = rB > 0 ? in + std::size_t(2*J+1) * s * s : ain;
              scalar_t* o = out + std::size_t(J) * s2 * s2;
              m_rB.push_back(rB); c_ld.push_back(ld); c_sB.push_back(rB > 0 ? s : 0);
              a.push_back(ain); b.push_back(bin); oa.push_back(o); ob.push_back(o + s + std::size_t(s) * s2);
              T.push_back(Tbase + j * std::size_t(s) * s);
              if (side == 0) {   // L: M = C (rows rB, cols s) below the diagonal block A; X -> o(s.., 0..)
                M.push_back(rB > 0 ? F + std::size_t(s2)*J + s + std::size_t(s2)*J*ld : F);
                oX.push_back(o + s);
              } else {           // U: M = B (rows s, cols rB) right of the diagonal block A; X -> o(0.., s..)
                M.push_back(rB > 0 ? F + std::size_t(s2)*J + (std::size_t(s2)*J + s)*ld : F);
                oX.push_back(o + std::size_t(s) * s2);
              }
            }
          }
          RO r; r.s = s; r.nblk = int(j); r.max_rB = 0;
          for (int v : m_rB) r.max_rB = std::max(r.max_rB, v);
          auto push_i = [&](const std::vector<int>& v) { std::size_t o = ints.size(); ints.insert(ints.end(), v.begin(), v.end()); ints.push_back(0); return o; };
          auto push_p = [&](const std::vector<scalar_t*>& v) { std::size_t o = ptrs.size(); ptrs.insert(ptrs.end(), v.begin(), v.end()); ptrs.push_back(nullptr); return o; };
          r.m_rB = push_i(m_rB); r.c_s = push_i(std::vector<int>(j, s)); r.c_2s = push_i(std::vector<int>(j, s2));
          r.c_ld = push_i(c_ld); r.c_sB = push_i(c_sB);
          r.a = push_p(a); r.b = push_p(b); r.oa = push_p(oa); r.ob = push_p(ob); r.M = push_p(M); r.T = push_p(T); r.oX = push_p(oX);
          ro[side].push_back(r);
        }
      }
      V.d_ints = gpu::DeviceMemory<int>(ints.size() + 1);
      V.d_ptrs = gpu::DeviceMemory<scalar_t*>(ptrs.size() + 1);
      if (!ints.empty()) gpu::copy_host_to_device<int>(V.d_ints, ints.data(), ints.size());
      gpu::copy_host_to_device<scalar_t*>(V.d_ptrs, ptrs.data(), ptrs.size());
      int* di = V.d_ints; scalar_t** dp = V.d_ptrs;
      for (int side=0; side<2; side++) {
        for (auto& r : ro[side]) {
          typename FactorCache::InvRound R;
          R.s = r.s; R.nblk = r.nblk; R.max_rB = r.max_rB;
          R.m_rB = di + r.m_rB; R.c_s = di + r.c_s; R.c_2s = di + r.c_2s; R.c_ld = di + r.c_ld; R.c_sB = di + r.c_sB;
          R.a = dp + r.a; R.b = dp + r.b; R.oa = dp + r.oa; R.ob = dp + r.ob; R.M = dp + r.M; R.T = dp + r.T; R.oX = dp + r.oX;
          V.rounds[side].push_back(R);
        }
      }
      if (invdiag::verbose())
        std::cout << "# INVDIAG plan level " << l << " fronts " << N << " max_d1 " << L.max_d1_small
                  << " nb " << NB << " nblk_max " << V.nblk_max << " inv_MB " << 2. * V.side_elems * sizeof(scalar_t) / 1e6
                  << " scratch_MB " << V.scratch_elems * sizeof(scalar_t) / 1e6 << std::endl;
    }
  }

  // S-INVDIAG: fill the inverses of level l (called right after the LU of the level, every factorization)
  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::invdiag_build(int l, gpu::Handle& handle) {
    auto& C = *factor_cache_;
    auto& V = C.inv[l];
    auto& L = C.ldata[l];
    const magma_int_t N = L.Nsmall;
    const int verbose = invdiag::verbose();
    cudaStream_t st = magma_queue_get_cuda_stream(gpu::get_magma_queue(handle));
    if (verbose) cudaEventRecord(C.inv_t0[l], st);
    cudaMemsetAsync(V.d_inv.template as<scalar_t>(), 0, 2 * V.side_elems * sizeof(scalar_t), st);
    // the intermediate layouts in the scratch must be zero too: the lacpy of a composed s-block copies its whole
    // s x s square, including the quadrant that round never writes (NB = 512 read the previous level's data there)
    if (V.scratch_elems) cudaMemsetAsync(C.inv_scratch.template as<scalar_t>(), 0, V.scratch_elems * sizeof(scalar_t), st);
    for (int side=0; side<2; side++) {
      // MAGMA's trtri_diag_vbatched returns wrong blocks (checked standalone, stock 2.10.0 too): one nonbatched
      // call per front (~10 launches each, once per factorization)
      for (magma_int_t i=0; i<N; i++) {
        auto& f = *(L.f[i]);
        gpu::magma::trtri_diag(side == 0 ? MagmaLower : MagmaUpper, side == 0 ? MagmaUnit : MagmaNonUnit,
                               f.dim_sep(), f.F11_.data(), f.F11_.ld(), V.h128[side][i], handle);
      }
      for (auto& R : V.rounds[side]) {
        const int s = R.s;
        gpu::magma::lacpy_vbatched(MagmaFull, s, s, R.c_s, R.c_s, (const scalar_t* const*)R.a, R.c_s, R.oa, R.c_2s, R.nblk, handle);
        gpu::magma::lacpy_vbatched(MagmaFull, s, s, R.c_sB, R.c_sB, (const scalar_t* const*)R.b, R.c_s, R.ob, R.c_2s, R.nblk, handle);
        if (R.max_rB <= 0) continue;
        if (side == 0) {   // T = C Ai ; X = -Bi T
          gpu::magma::gemm_vbatched_max_nocheck
            (MagmaNoTrans, MagmaNoTrans, R.m_rB, R.c_s, R.c_s, scalar_t(1.), R.M, R.c_ld, R.a, R.c_s,
             scalar_t(0.), R.T, R.c_s, R.nblk, R.max_rB, s, s, handle);
          gpu::magma::gemm_vbatched_max_nocheck
            (MagmaNoTrans, MagmaNoTrans, R.m_rB, R.c_s, R.m_rB, scalar_t(-1.), R.b, R.c_s, R.T, R.c_s,
             scalar_t(0.), R.oX, R.c_2s, R.nblk, R.max_rB, s, R.max_rB, handle);
        } else {           // T = B Di ; X = -Ai T
          gpu::magma::gemm_vbatched_max_nocheck
            (MagmaNoTrans, MagmaNoTrans, R.c_s, R.m_rB, R.m_rB, scalar_t(1.), R.M, R.c_ld, R.b, R.c_s,
             scalar_t(0.), R.T, R.c_s, R.nblk, s, R.max_rB, R.max_rB, handle);
          gpu::magma::gemm_vbatched_max_nocheck
            (MagmaNoTrans, MagmaNoTrans, R.c_s, R.m_rB, R.c_s, scalar_t(-1.), R.a, R.c_s, R.T, R.c_s,
             scalar_t(0.), R.oX, R.c_2s, R.nblk, s, R.max_rB, s, handle);
        }
      }
    }
    if (verbose) cudaEventRecord(C.inv_t1[l], st);
    if (verbose >= 2) {   // host check: max over blocks of ||I - inv * block||_max
      cudaStreamSynchronize(st);
      const int NB = V.nb;
      double worst = 0.;
      for (int side=0; side<2; side++)
        for (magma_int_t i=0; i<N; i++) {
          auto& f = *(L.f[i]);
          const int d1 = f.dim_sep(), ld = f.F11_.ld(), K = (d1 + NB - 1) / NB;
          if (!d1) continue;
          std::vector<scalar_t> A(std::size_t(ld) * d1), Inv(std::size_t(K) * NB * NB);
          gpu::copy_device_to_host<scalar_t>(A.data(), f.F11_.data(), A.size());
          gpu::copy_device_to_host<scalar_t>(Inv.data(), side == 0 ? f.invL_ : f.invU_, Inv.size());
          for (int k=0; k<K; k++) {
            const int r = std::min(NB, d1 - k*NB);
            const scalar_t* B = Inv.data() + std::size_t(k) * NB * NB;
            double w = 0.;
#pragma omp parallel for reduction(max:w) collapse(2)
            for (int q=0; q<r; q++)
              for (int p=0; p<r; p++) {
                scalar_t acc(0.);
                for (int t=0; t<r; t++) {
                  const int gr = k*NB + t, gc = k*NB + q;
                  scalar_t blk(0.);
                  if (side == 0) { if (t > q) blk = A[gr + std::size_t(gc)*ld]; else if (t == q) blk = scalar_t(1.); }
                  else if (t <= q) blk = A[gr + std::size_t(gc)*ld];
                  acc += B[p + std::size_t(t)*NB] * blk;
                }
                if (p == q) acc -= scalar_t(1.);
                w = std::max(w, double(std::abs(acc)));
              }
            worst = std::max(worst, w);
          }
        }
      C.inv_resid = std::max(C.inv_resid, worst);
      std::cout << "# INVDIAG check level " << l << " max_resid " << worst << std::endl;
    }
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontMAGMA<scalar_t,integer_t>::factors_on_device_legacy
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata,
   std::size_t total_dmem) {
    ReturnCode err_code = ReturnCode::SUCCESS;
    factor_cache_.reset();
    solve_cache_.reset(); // new factors: rebuild the solve workspace
    devperm_.reset();     // S-HOST: and the device permutation
    int lvls = ldata.size();
    gpu::Stream comp_stream, copy_stream;
    gpu::Handle handle(comp_stream);
    std::size_t peak_ea_hmem = 0, total_factor_dmem = 0,
      peak_work_dmem = 0;
    for (int l=0; l<lvls; l++) {
      peak_ea_hmem = std::max(peak_ea_hmem, ldata[l].ea_bytes);
      std::size_t level_mem = ldata[l].work_bytes + ldata[l].ea_bytes;
      if (l+1 < lvls)
        level_mem += ldata[l+1].work_bytes;
      peak_work_dmem = std::max(peak_work_dmem, level_mem);
      total_factor_dmem += ldata[l].factor_bytes;
    }
    gpu::HostMemory<char> ea_hmem(peak_ea_hmem);
    gpu::DeviceMemory<char> work_dmem(peak_work_dmem);
    dev_factors_.reset(new gpu::DeviceMemory<char>(total_factor_dmem));
    char* factors_dptr = dev_factors_->as<char>();

    for (int l=lvls-1; l>=0; l--) {
      // TaskTimer tl("");
      // tl.start();
      auto& L = ldata[l];
      // if (opts.verbose()) L.print_info(l, lvls);
      try {
        char *work_dptr = nullptr, *ea_dptr = nullptr;
        if (l % 2) {
          work_dptr = work_dmem;
          ea_dptr = work_dptr + L.work_bytes;
        } else {
          work_dptr = work_dmem + peak_work_dmem - L.work_bytes;
          ea_dptr = work_dptr - L.ea_bytes;
        }
        gpu::memset<scalar_t>(work_dptr, 0, L.Schur_size);
        gpu::memset<scalar_t>(factors_dptr, 0, L.factor_size);
        L.set_factor_pointers(factors_dptr);
        factors_dptr += L.factor_bytes;
        L.set_work_pointers(work_dptr);

        // default stream
        front_assembly(A, L, ea_hmem, ea_dptr);

        std::size_t N = L.f.size(), Nsmall = L.Nsmall;
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F11 = L.dev_F_batch;   auto F12 = F11 + Nsmall;
        auto F21 = F12 + Nsmall;    auto F22 = F21 + Nsmall;

        getrf_vbatched_chunked
          (d1, L.max_d1_small, F11, ld1, L.dev_ipiv_batch,
           L.dev_getrf_err, L.dev_getrf_work, &L.getrf_work_bytes,
           Nsmall, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrf
            (handle, L.f[i]->F11_, (scalar_t*)L.dev_getrf_work,
             L.getrf_work_bytes / sizeof(scalar_t),
             L.f[i]->piv_, L.dev_getrf_err+i);
        // TODO check error code
        if (opts.replace_tiny_pivots()) {
          gpu::replace_pivots_vbatched
            (handle, d1, L.max_d1_small, F11, ld1,
             opts.pivot_threshold(), Nsmall);
          for (std::size_t i=Nsmall; i<N; i++)
            gpu::replace_pivots
              (L.f[i]->F11_.rows(), L.f[i]->F11_.data(),
               opts.pivot_threshold());
        }
        gpu::laswp_fwd_vbatched
          (handle, d2, L.max_d2_small, F12, ld1, L.dev_ipiv_batch,
           d1, Nsmall);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, handle);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrs
            (handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_,
             L.f[i]->F12_, L.dev_getrf_err+i);
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans, d2, d2, d1, scalar_t(-1.),
           F21, ld2, F12, ld1, scalar_t(1.), F22, ld2, Nsmall,
           L.max_d2_small, L.max_d2_small, L.max_d1_small, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::gemm
            (handle, Trans::N, Trans::N, scalar_t(-1.),
             L.f[i]->F21_, L.f[i]->F12_, scalar_t(1.), L.f[i]->F22_);
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      long long level_flops, small_flops;
      L.flops(level_flops, small_flops);
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      STRUMPACK_FLOPS(small_flops);
      static const bool level_info = std::getenv("STRUM_LEVEL_INFO") != nullptr;
      if (level_info) {
        std::size_t max_d1 = 0, max_d2 = 0, sum_d1 = 0, sum_d2 = 0;
        double gemm_flops = 0., getrf_flops = 0., trsm_flops = 0., sum_d1d2 = 0., sum_d1sq = 0.;
        for (auto f : L.f) {
          const double d1 = f->dim_sep(), d2 = f->dim_upd();
          max_d1 = std::max<std::size_t>(max_d1, d1); max_d2 = std::max<std::size_t>(max_d2, d2);
          sum_d1 += d1; sum_d2 += d2; sum_d1d2 += d1 * d2; sum_d1sq += d1 * d1;
          gemm_flops += 2. * d2 * d2 * d1; getrf_flops += 2. / 3. * d1 * d1 * d1; trsm_flops += 2. * d1 * d1 * d2;
        }
        std::cout << "# LEVEL_INFO level " << l << " fronts " << L.f.size() << " max_d1 " << max_d1 << " max_d2 " << max_d2
                  << " sum_d1 " << sum_d1 << " sum_d2 " << sum_d2 << " gemm_gflop " << gemm_flops / 1e9
                  << " getrf_gflop " << getrf_flops / 1e9 << " trsm_gflop " << trsm_flops / 1e9
                  << " level_flops " << level_flops << " small_flops " << small_flops
                  << " factor_MB " << L.factor_bytes / 1e6 << " work_MB " << L.work_bytes / 1e6 << " ea_MB " << L.ea_bytes / 1e6
                  << " elems " << L.elems11.back() + L.elems12.back() + L.elems21.back()
                  << " sum_d1d2 " << sum_d1d2 << " sum_d1sq " << sum_d1sq << std::endl;
      }
    }
    return err_code;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontMAGMA<scalar_t,integer_t>::split_smaller
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    const std::size_t dupd = dim_upd(), dsep = dim_sep();
    ReturnCode err_code = ReturnCode::SUCCESS;
    if (lchild_) {
      auto el = lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (el != ReturnCode::SUCCESS) err_code = el;
    }
    if (rchild_) {
      auto er = rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (er != ReturnCode::SUCCESS) err_code = er;
    }
    STRUMPACK_ADD_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
    STRUMPACK_ADD_MEMORY(dupd*dupd*sizeof(scalar_t));
    host_factors_.reset(new scalar_t[dsep*(dsep+2*dupd)]);
    {
      auto fmem = host_factors_.get();
      F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
      F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
      F21_ = DenseMW_t(dupd, dsep, fmem, dupd);
    }
    DenseM_t F22(dupd, dupd);
    F11_.zero(); F12_.zero();
    F21_.zero(); F22.zero();
    A.extract_front
      (F11_, F12_, F21_, this->sep_begin_, this->sep_end_,
       this->upd_, task_depth);
    if (lchild_) {
#pragma omp parallel
#pragma omp single
      lchild_->extend_add_to_dense(F11_, F12_, F21_, F22, this, 0);
    }
    if (rchild_) {
#pragma omp parallel
#pragma omp single
      rchild_->extend_add_to_dense(F11_, F12_, F21_, F22, this, 0);
    }
    // TaskTimer tl("");
    // tl.start();
    if (dsep) {
      gpu::Handle h;
      auto lwork = gpu::getrf_buffersize<scalar_t>(h, dsep);
      gpu::DeviceMemory<scalar_t> dm11(dsep*dsep + lwork);
      gpu::DeviceMemory<int> dpiv(dsep+1); // and ierr
      DenseMW_t dF11(dsep, dsep, dm11, dsep);
      gpu::copy_host_to_device(dF11, F11_);
      gpu::getrf(h, dF11, dm11 + dsep*dsep, lwork, dpiv, dpiv+dsep);
      // TODO check return code!
      if (opts.replace_tiny_pivots())
        gpu::replace_pivots
          (F11_.rows(), dF11.data(), opts.pivot_threshold());
      pivot_mem_.resize(dsep);
      piv_ = pivot_mem_.data();
      gpu::copy_device_to_host(piv_, dpiv.as<int>(), dsep);
      gpu::copy_device_to_host(F11_, dF11);
      if (dupd) {
        gpu::DeviceMemory<scalar_t> dm12(dsep*dupd);
        DenseMW_t dF12(dsep, dupd, dm12, dsep);
        gpu::copy_host_to_device(dF12, F12_);
        gpu::getrs(h, Trans::N, dF11, dpiv, dF12, dpiv+dsep);
        gpu::copy_device_to_host(F12_, dF12);
        dm11.release();
        gpu::DeviceMemory<scalar_t> dm21(dsep*dupd);
        DenseMW_t dF21(dupd, dsep, dm21, dupd);
        gpu::copy_host_to_device(dF21, F21_);
        dev_Schur_.reset(new gpu::DeviceMemory<char>(dupd*dupd*sizeof(scalar_t)));
        F22_ = DenseMW_t(dupd, dupd, dev_Schur_->template as<scalar_t>(), dupd);
        gpu::copy_host_to_device(F22_, F22);
        gpu::gemm(h, Trans::N, Trans::N, scalar_t(-1.),
                  dF21, dF12, scalar_t(1.), F22_);
      }
    }
    // count flops
    auto level_flops = LU_flops(F11_) +
      gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
      trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
      trsm_flops(Side::R, scalar_t(1.), F11_, F21_);
    (void)level_flops;
    STRUMPACK_FULL_RANK_FLOPS(level_flops);
    // if (opts.verbose()) {
    //   auto level_time = tl.elapsed();
    //   std::cout << "#   GPU Factorization complete, took: "
    //             << level_time << " seconds, "
    //             << level_flops / 1.e9 << " GFLOPS, "
    //             << (float(level_flops) / level_time) / 1.e9
    //             << " GFLOP/s" << std::endl;
    // }
    return err_code;
  }


  // check the device memory required for doing it level by level when
  // moving the factors back to the host after each level
  template<typename scalar_t,typename integer_t>
  std::size_t level_peak_device_memory_MAGMA
  (const std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata) {
    std::size_t peak_dmem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      auto& L = ldata[l];
      // memory needed on this level: factors,
      // schur updates, pivot vectors, cuSOLVER work space,
      // assembly data (indices, sparse elements)
      std::size_t level_mem = L.factor_bytes + L.work_bytes + L.ea_bytes;
      // the contribution blocks of the previous level are still
      // needed for the extend-add
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_dmem = std::max(peak_dmem, level_mem);
    }
    return peak_dmem;
  }

  // check if we can keep everything on the device, so the factors
  // reside on the device and can be used in the solve
  template<typename scalar_t,typename integer_t>
  std::size_t total_peak_device_memory_MAGMA
  (const std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata) {
    std::size_t peak_dmem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      std::size_t level_mem = ldata[l].work_bytes + ldata[l].ea_bytes;
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_dmem = std::max(peak_dmem, level_mem);
    }
    for (std::size_t l=0; l<ldata.size(); l++)
      peak_dmem += ldata[l].factor_bytes;
    return peak_dmem;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontMAGMA<scalar_t,integer_t>::factor
  (const SpMat_t& A, const Opts_t& opts, VectorPool<scalar_t>& workspace,
   int etree_level, int task_depth) {
    ReturnCode err_code = ReturnCode::SUCCESS;
    if (etree_level == 0 && factor_cache_ && factor_cache_->N == A.size() &&
        factor_cache_->nnz == A.nnz()) {   // F-GATHER: same structure, reuse everything
      std::vector<LInfo_t> none;
      return factors_on_device(A, opts, none, 0);
    }
    gpu::Stream comp_stream, copy_stream;
    gpu::Handle handle(comp_stream);
    const int lvls = this->levels();
    std::vector<LInfo_t> ldata(lvls);
    {
      std::vector<std::vector<F_t*>> fp(lvls);
      try {
        this->get_level_fronts_gpu(fp);
      } catch (...) {
        return split_smaller(A, opts, etree_level, task_depth);
      }
      for (int l=lvls-1; l>=0; l--)
        ldata[l] = LInfo_t(fp[l], handle, &A);
    }

#if 1 // enable for solve on GPU (if factors fit on device)
    if (etree_level == 0) {
      auto total_dmem = total_peak_device_memory_MAGMA(ldata);
      if (opts.verbose())
        std::cout << "# Need " << total_dmem / 1.e6 << " MB "
                  << "of device mem, "
                  << gpu::available_memory() / 1.e6
                  << " MB available" << std::endl;
      if (total_dmem < 0.9 * gpu::available_memory())
        return factors_on_device(A, opts, ldata, total_dmem);
    }
#endif

    auto peak_dmem = level_peak_device_memory_MAGMA(ldata);
    if (peak_dmem >= 0.9 * gpu::available_memory()) {
      if (opts.verbose())
        std::cout << "# Factorization does not fit in GPU memory, "
          "splitting in smaller traversals." << std::endl;
      return split_smaller(A, opts, etree_level, task_depth);
    }

    std::size_t pinned_size = 0;
    for (int l=lvls-1; l>=0; l--)
      pinned_size = std::max(pinned_size, ldata[l].factor_size);
    auto pinned = workspace.get_pinned(pinned_size);

    std::size_t peak_hea_mem = 0;
    for (int l=lvls-1; l>=0; l--)
      peak_hea_mem = std::max(peak_hea_mem, ldata[l].ea_bytes);
    gpu::HostMemory<char> hea_mem(peak_hea_mem);
    auto all_dmem = workspace.get_device_bytes(peak_dmem);
    char* old_work = nullptr;

    for (int l=lvls-1; l>=0; l--) {
      // TaskTimer tl("");
      // tl.start();
      auto& L = ldata[l];
      // if (opts.verbose()) L.print_info(l, lvls);
      try {
        char *work_mem = nullptr, *dea_mem = nullptr;
        scalar_t* dev_factors = nullptr;
        if (l % 2) {
          work_mem = all_dmem;
          dea_mem = work_mem + L.work_bytes;
          dev_factors = gpu::aligned_ptr<scalar_t>(dea_mem + L.ea_bytes);
        } else {
          work_mem = all_dmem + peak_dmem - L.work_bytes;
          dea_mem = work_mem - L.ea_bytes;
          dev_factors = gpu::aligned_ptr<scalar_t>(dea_mem - L.factor_bytes);
        }
        gpu::memset<scalar_t>(work_mem, 0, L.Schur_size);
        gpu::memset<scalar_t>(dev_factors, 0, L.factor_size);
        L.set_factor_pointers(dev_factors);
        L.set_work_pointers(work_mem);
        old_work = work_mem;

        // default stream
        front_assembly(A, L, hea_mem, dea_mem);

        std::size_t N = L.f.size(), Nsmall = L.Nsmall;
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F11 = L.dev_F_batch;   auto F12 = F11 + Nsmall;
        auto F21 = F12 + Nsmall;    auto F22 = F21 + Nsmall;

        getrf_vbatched_chunked
          (d1, L.max_d1_small, F11, ld1, L.dev_ipiv_batch,
           L.dev_getrf_err, L.dev_getrf_work, &L.getrf_work_bytes,
           Nsmall, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrf
            (handle, L.f[i]->F11_, (scalar_t*)L.dev_getrf_work,
             L.getrf_work_bytes / sizeof(scalar_t),
             L.f[i]->piv_, L.dev_getrf_err+i);
        // TODO check error code
        if (opts.replace_tiny_pivots()) {
          gpu::replace_pivots_vbatched
            (handle, d1, L.max_d1_small, F11, ld1,
             opts.pivot_threshold(), Nsmall);
          for (std::size_t i=Nsmall; i<N; i++)
            gpu::replace_pivots
              (L.f[i]->F11_.rows(), L.f[i]->F11_.data(),
               opts.pivot_threshold());
        }
        gpu::laswp_fwd_vbatched
          (handle, d2, L.max_d2_small, F12, ld1, L.dev_ipiv_batch,
           d1, Nsmall);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, handle);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrs
            (handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_,
             L.f[i]->F12_, L.dev_getrf_err+i);

        STRUMPACK_ADD_MEMORY(L.factor_bytes);
        L.f[0]->host_factors_.reset(new scalar_t[L.factor_size]);
        L.f[0]->pivot_mem_.resize(L.piv_size);

        comp_stream.synchronize();
        gpu::copy_device_to_host_async<scalar_t>
          (pinned, dev_factors, L.factor_size, copy_stream);

        // use max_nocheck to overlap this with copy above
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans, d2, d2, d1, scalar_t(-1.),
           F21, ld2, F12, ld1, scalar_t(1.), F22, ld2, Nsmall,
           L.max_d2_small, L.max_d2_small, L.max_d1_small, handle);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::gemm
            (handle, Trans::N, Trans::N, scalar_t(-1.),
             L.f[i]->F21_, L.f[i]->F12_, scalar_t(1.), L.f[i]->F22_);
        copy_stream.synchronize();
        auto host_factors = L.f[0]->host_factors_.get();
#pragma omp parallel for
        for (std::size_t i=0; i<L.factor_size; i++)
          host_factors[i] = pinned[i];

        gpu::copy_device_to_host<int>
          (L.f[0]->pivot_mem_.data(), L.f[0]->piv_, L.piv_size);
        L.set_factor_pointers
          (L.f[0]->host_factors_.get(), L.f[0]->pivot_mem_.data());
        comp_stream.synchronize();
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      long long level_flops, small_flops;
      L.flops(level_flops, small_flops);
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      STRUMPACK_FLOPS(small_flops);
      // if (opts.verbose()) {
      //   auto level_time = tl.elapsed();
      //   std::cout << "#   GPU Factorization complete, took: "
      //             << level_time << " seconds, "
      //             << level_flops / 1.e9 << " GFLOPS, "
      //             << (float(level_flops) / level_time) / 1.e9
      //             << " GFLOP/s" << std::endl;
      // }
    }
    const std::size_t dupd = dim_upd();
    if (dupd) { // get the contribution block from the device
      dev_Schur_.reset(new gpu::DeviceMemory<char>
                       (workspace.get_device_bytes
                        (dupd*dupd*sizeof(scalar_t))));
      auto dF22 = dev_Schur_->template as<scalar_t>();
      gpu::copy_device_to_device(dF22, (scalar_t*)(old_work), dupd*dupd);
      F22_ = DenseMW_t(dupd, dupd, dF22, dupd);
    }
    workspace.restore(all_dmem);
    workspace.restore(pinned);
    return err_code;
  }

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& b) const {
    if (dev_factors_) gpu_solve(b);
    else {
#if defined(STRUMPACK_USE_MPI)
      if (!mpi_rank())
#endif
        std::cerr << "WARNING: Solve is performed on CPU" << std::endl;
      // factors are not on the device, solve on CPU
      Front<scalar_t,integer_t>::multifrontal_solve(b);
    }
  }

  template<typename scalar_t,typename integer_t> bool
  FrontMAGMA<scalar_t,integer_t>::multifrontal_solve_devperm
  (const DenseM_t& b, DenseM_t& x, const std::vector<integer_t>& iperm,
   const std::vector<integer_t>& perm) const {
    if (!dev_factors_) return false;
    const std::size_t n = x.rows();
    if (std::size_t(b.rows()) != n || b.cols() != x.cols() ||
        iperm.size() != n || perm.size() != n ||
        n > std::size_t(std::numeric_limits<int>::max())) return false;
    if (!devperm_ || devperm_->key_iperm != iperm.data() ||
        devperm_->key_perm != perm.data() || devperm_->n != n) {
      devperm_.reset(new DevPerm());
      auto& D = *devperm_;
      D.key_iperm = iperm.data();  D.key_perm = perm.data();  D.n = n;
      D.d_iperm = gpu::DeviceMemory<int>(n);
      D.d_perm = gpu::DeviceMemory<int>(n);
      if (std::is_same<integer_t,int>::value) {
        gpu::copy_host_to_device<int>(D.d_iperm, reinterpret_cast<const int*>(iperm.data()), n);
        gpu::copy_host_to_device<int>(D.d_perm, reinterpret_cast<const int*>(perm.data()), n);
      } else {
        std::vector<int> t(n);
        for (std::size_t i=0; i<n; i++) t[i] = int(iperm[i]);
        gpu::copy_host_to_device<int>(D.d_iperm, t.data(), n);
        for (std::size_t i=0; i<n; i++) t[i] = int(perm[i]);
        gpu::copy_host_to_device<int>(D.d_perm, t.data(), n);
      }
    }
    DevPermArgs dp{&b, devperm_->d_iperm, devperm_->d_perm};
    gpu_solve(x, &dp);
    return true;
  }

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      F11_.solve_LU_in_place(bloc, piv_, task_depth);
      if (dim_upd()) {
        if (b.cols() == 1)
          gemv(Trans::N, scalar_t(-1.), F21_, bloc,
               scalar_t(1.), bupd, task_depth);
        else
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, bloc,
               scalar_t(1.), bupd, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      if (y.cols() == 1) {
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12_, yupd,
               scalar_t(1.), yloc, task_depth);
      } else {
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12_, yupd,
               scalar_t(1.), yloc, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontMAGMA<scalar_t,integer_t>::gpu_solve(DenseM_t& b, const DevPermArgs* dp) const {
    int nrhs = b.cols();
    if (!solve_cache_ || solve_cache_->nrhs != nrhs ||
        solve_cache_->rows != std::size_t(b.rows()))
      solve_cache_.reset(new SolveCache(nrhs, b.rows()));
    auto& C = *solve_cache_;
    // first solve with this cache: build level info and metadata
    const bool fresh = C.ldata.empty();
    static const bool pack_verbose = std::getenv("STRUM_DEVUPDATE_VERBOSE") && std::atoi(std::getenv("STRUM_DEVUPDATE_VERBOSE")) != 0;
    const auto t_s0 = std::chrono::steady_clock::now();
    auto& handle = C.handle;
    const int lvls = this->levels();
    auto& ldata = C.ldata;
    auto& l_off = C.l_off;
    auto& Nmax = C.Nmax;   auto& Ntotal = C.Ntotal;
    auto& max_bupd_size = C.max_bupd_size;
    auto& max_work_mem = C.max_work_mem;
    auto& Isize = C.Isize;
    if (fresh) {
      ldata.resize(lvls);
      for (int l=lvls-1; l>=0; l--) {
        std::vector<F_t*> fp;
        const_cast<FM_t*>(this)->get_level_fronts(fp, l);
        ldata[l] = LInfo_t(fp, handle);
      }
      l_off.assign(lvls, 0);
      for (int l=0; l<lvls; l++) {
        auto& L = ldata[l];
        auto N = L.f.size();
        if (l != lvls-1) l_off[l+1] = l_off[l] + N;
        Nmax = std::max(Nmax, N);
        Ntotal += N;
        Isize += L.Isize.back();
        max_bupd_size = std::max(max_bupd_size, L.total_upd_size*nrhs);
        max_work_mem = std::max(max_work_mem, L.work_bytes);
      }
    }
    std::size_t d_mem_bytes = 0;
    d_mem_bytes += gpu::round_up(Ntotal*sizeof(gpu::AssembleData<scalar_t>));
    d_mem_bytes += gpu::round_up(Isize*sizeof(std::size_t));
    d_mem_bytes += gpu::round_up((3*(Nmax+1)+1)*sizeof(int));
    d_mem_bytes += gpu::round_up(2*Ntotal*sizeof(scalar_t*));
    auto h_mem_bytes = d_mem_bytes;
    d_mem_bytes += gpu::round_up(max_work_mem);
    d_mem_bytes += gpu::round_up((nrhs*b.rows()+2*max_bupd_size)*sizeof(scalar_t));
    if (fresh) {
      C.d_mem = gpu::DeviceMemory<char>(d_mem_bytes);
      C.h_mem = gpu::HostMemory<char>(h_mem_bytes);
    }
    // Device regions for the per-level batch meta-data (one per level).
    auto& meta_off = C.meta_off;
    if (fresh) {
      std::size_t meta_bytes = 0;
      meta_off.assign(lvls, 0);
      for (int l=0; l<lvls; l++) {
        meta_off[l] = meta_bytes;
        meta_bytes += gpu::round_up(ldata[l].batch_meta_bytes);
      }
      // + slack for the alignment inside set_batch_data
      C.d_meta = gpu::DeviceMemory<char>(meta_bytes + gpu::round_up(1));
    }
    auto& d_mem = C.d_mem;
    auto d_asmbl      = d_mem.template as<gpu::AssembleData<scalar_t>>();
    auto d_I          = gpu::aligned_ptr<std::size_t>(d_asmbl+Ntotal);
    auto d_batch_int  = gpu::aligned_ptr<int>(d_I+Isize);
    auto d_batch_ptrs = gpu::aligned_ptr<scalar_t*>(d_batch_int+3*(Nmax+1)+1);
    auto dwork_mem    = gpu::aligned_ptr<char>(d_batch_ptrs+2*Ntotal);
    auto d_rhs_mem    = gpu::aligned_ptr<scalar_t>(dwork_mem+max_work_mem);
    auto& h_mem = C.h_mem;
    auto h_asmbl      = h_mem.template as<gpu::AssembleData<scalar_t>>();
    auto h_I          = gpu::aligned_ptr<std::size_t>(h_asmbl+Ntotal);
    auto h_batch_int  = gpu::aligned_ptr<int>(h_I+Isize);
    auto h_batch_ptrs = gpu::aligned_ptr<scalar_t*>(h_batch_int+3*(Nmax+1)+1);
    // scalar device data
    auto d_b = d_rhs_mem;
    auto d_bupd_odd = d_b + nrhs * b.rows();
    auto d_bupd_even = d_bupd_odd + max_bupd_size;
    // int device data
    auto nrhs_batch = d_batch_int;
    auto ldrhs_batch = nrhs_batch + Nmax+1;
    auto inc_batch = ldrhs_batch + Nmax+1;
    auto getrs_err = inc_batch + Nmax+1;
    auto& h_rhs_batch = C.h_rhs_batch;   auto& h_bupd_batch = C.h_bupd_batch;
    auto& d_rhs_batch = C.d_rhs_batch;   auto& d_bupd_batch = C.d_bupd_batch;
    if (fresh) {
      for (std::size_t i=0; i<Nmax+1; i++) {
        h_batch_int[i           ] = nrhs;
        h_batch_int[i+  (Nmax+1)] = b.rows();
        h_batch_int[i+2*(Nmax+1)] = 1;
      }
      h_rhs_batch.resize(lvls); h_bupd_batch.resize(lvls);
      d_rhs_batch.resize(lvls); d_bupd_batch.resize(lvls);
#pragma omp parallel for schedule(static,1)
      for (std::size_t l=0; l<std::size_t(lvls); l++) {
        d_rhs_batch[l] = d_batch_ptrs + l_off[l];
        d_bupd_batch[l] = d_batch_ptrs + Ntotal + l_off[l];
        h_rhs_batch[l] = h_batch_ptrs + l_off[l];
        h_bupd_batch[l] = h_batch_ptrs + Ntotal + l_off[l];
        auto bu = (l % 2) ? d_bupd_odd : d_bupd_even;
        auto& L = ldata[l];
        for (std::size_t i=0, pos=l_off[l]; i<L.f.size(); i++, pos++) {
          h_batch_ptrs[pos] = d_b + L.f[i]->sep_begin();
          h_batch_ptrs[Ntotal+pos] = bu;
          bu += nrhs * L.f[i]->dim_upd();
        }
      }
      for (std::size_t l=0, Ipos=0; l<std::size_t(lvls); l++) {
        auto& L = ldata[l];
#pragma omp parallel for
        for (std::size_t i=0; i<L.f.size(); i++) {
          auto& f = *(L.f[i]);
          h_asmbl[l_off[l]+i] = gpu::AssembleData<scalar_t>
            (f.dim_sep(), f.dim_upd(), h_rhs_batch[l][i], h_bupd_batch[l][i]);
          auto hI = h_I+Ipos+L.Isize[i];
          if (f.lchild_) {
            f.lchild_->upd_to_parent(&f, hI);
            hI += f.lchild_->dim_upd();
          }
          if (f.rchild_)
            f.rchild_->upd_to_parent(&f, hI);
        }
        for (std::size_t i=0, ch=0; i<L.f.size(); i++) {
          auto& f = *(L.f[i]);
          auto dI = d_I+Ipos+L.Isize[i];
          if (f.lchild_) {
            auto dupd = f.lchild_->dim_upd();
            h_asmbl[l_off[l]+i].set_ext_add_left
              (dupd, h_bupd_batch[l+1][ch++], dI);
            dI += dupd;
          }
          if (f.rchild_)
            h_asmbl[l_off[l]+i].set_ext_add_right
              (f.rchild_->dim_upd(), h_bupd_batch[l+1][ch++], dI);
        }
        Ipos += L.Isize.back();
      }
    }
    if (fresh) {
      // Everything in h_mem / d_mem[0,h_mem_bytes) and the per-level batch
      // meta-data only depends on the factorization, nrhs and the number of
      // rows (the d_b / d_bupd pointers point into d_mem, which lives as
      // long as this cache). The cache is dropped on refactorization
      // (factors_on_device) and rebuilt when nrhs or rows change, so upload
      // once here instead of at every solve.
      // copy all meta-data at once, from pinned memory
      gpu::copy_host_to_device<char>(d_mem, h_mem, h_mem_bytes);
      C.n_mem_uploads++;
      // per-level batch meta-data; set_batch_data also sets L.dev_*_batch
      // to point into this level's private region of d_meta
      for (int l=lvls-1; l>=0; l--) {
        ldata[l].set_batch_data(C.d_meta.template as<char>() + meta_off[l]);
        C.n_meta_uploads++;
      }
      // S-INVDIAG: per-step batch arrays of the block recurrence for the converted levels (nrhs == 1 only)
      C.inv.clear();
      if (nrhs == 1 && factor_cache_ && int(factor_cache_->inv.size()) == lvls && invdiag::enabled()) {
        C.inv.resize(lvls);
        std::size_t scr_max = 0;
        for (int l=0; l<lvls; l++) {
          if (!factor_cache_->inv[l].on) continue;
          std::size_t sum = 0;
          for (auto f : ldata[l].f) sum += f->dim_sep();
          scr_max = std::max(scr_max, sum);
        }
        if (scr_max) C.d_scr = gpu::DeviceMemory<scalar_t>(scr_max);
        scalar_t* scr_base = C.d_scr;
        std::vector<int> ints;
        std::vector<scalar_t*> ptrs;
        struct Off { std::size_t mL, nLu, nUu, ldnb, AL, AU, rhs, scr, Li, Ui; };
        std::vector<Off> offs(lvls);
        for (int l=0; l<lvls; l++) {
          auto& FV = factor_cache_->inv[l];
          if (!FV.on) continue;
          auto& L = ldata[l];
          const std::size_t N = L.f.size();
          bool ok = N == L.Nsmall && N > 0;
          for (std::size_t i=0; i<N && ok; i++) ok = L.f[i]->invL_ && L.f[i]->invU_;
          if (!ok) continue;
          auto& S = C.inv[l];
          S.on = true; S.U = invdiag::use_U(); S.nb = FV.nb; S.nblk = FV.nblk_max;
          const int NB = S.nb, K = S.nblk;
          const std::size_t NBB = std::size_t(NB) * NB;
          std::vector<std::size_t> scr_off(N+1, 0);
          for (std::size_t i=0; i<N; i++) scr_off[i+1] = scr_off[i] + L.f[i]->dim_sep();
          std::vector<int> mL, nLu, nUu, ldnb(N, NB);
          std::vector<scalar_t*> AL, AU, rhs, scr, Li, Ui;
          S.maxLu.assign(K, 0); S.maxUu.assign(K, 0);
          for (int k=0; k<K; k++)
            for (std::size_t i=0; i<N; i++) {
              auto& f = *(L.f[i]);
              const int d1 = f.dim_sep(), ld = f.F11_.ld();
              const int r = std::max(0, std::min(NB, d1 - k*NB));
              const int nU = r > 0 ? std::max(0, d1 - (k+1)*NB) : 0;
              scalar_t* F = f.F11_.data();
              scalar_t* b_i = d_b + f.sep_begin();
              mL.push_back(r); nLu.push_back(r > 0 ? k*NB : 0); nUu.push_back(nU);
              AL.push_back(r > 0 ? F + k*NB : F);
              AU.push_back(nU > 0 ? F + k*NB + std::size_t(k+1)*NB*ld : F);
              rhs.push_back(r > 0 ? b_i + k*NB : b_i);
              scr.push_back(scr_base + scr_off[i] + (r > 0 ? k*NB : 0));
              Li.push_back(r > 0 ? f.invL_ + k*NBB : f.invL_);
              Ui.push_back(r > 0 ? f.invU_ + k*NBB : f.invU_);
              if (r > 0) S.maxLu[k] = k*NB;
              S.maxUu[k] = std::max(S.maxUu[k], nU);
            }
          auto push_i = [&](const std::vector<int>& v) { std::size_t o = ints.size(); ints.insert(ints.end(), v.begin(), v.end()); ints.push_back(0); return o; };
          auto push_p = [&](const std::vector<scalar_t*>& v) { std::size_t o = ptrs.size(); ptrs.insert(ptrs.end(), v.begin(), v.end()); ptrs.push_back(nullptr); return o; };
          auto& O = offs[l];
          O.mL = push_i(mL); O.nLu = push_i(nLu); O.nUu = push_i(nUu); O.ldnb = push_i(ldnb);
          O.AL = push_p(AL); O.AU = push_p(AU); O.rhs = push_p(rhs); O.scr = push_p(scr); O.Li = push_p(Li); O.Ui = push_p(Ui);
          if (invdiag::verbose())
            std::cout << "# INVDIAG solve level " << l << " fronts " << N << " nb " << NB << " nblk " << K
                      << " U " << (S.U ? "inv" : "trsm") << std::endl;
        }
        if (!ints.empty()) {
          std::size_t part_max = 0, nmax_on = 0;
          for (int l=0; l<lvls; l++) if (C.inv[l].on) {
            part_max = std::max(part_max, ldata[l].f.size() * 64 * std::size_t(C.inv[l].nb));
            nmax_on = std::max(nmax_on, ldata[l].f.size());
          }
          if (invdiag::fused()) {
            C.d_partial = gpu::DeviceMemory<scalar_t>(part_max);
            C.d_counter = gpu::DeviceMemory<int>(nmax_on);
            gpu::memset<int>(C.d_counter, 0, nmax_on);
          }
          C.inv_ints = gpu::DeviceMemory<int>(ints.size());
          C.inv_ptrs = gpu::DeviceMemory<scalar_t*>(ptrs.size());
          gpu::copy_host_to_device<int>(C.inv_ints, ints.data(), ints.size());
          gpu::copy_host_to_device<scalar_t*>(C.inv_ptrs, ptrs.data(), ptrs.size());
          int* di = C.inv_ints; scalar_t** dp = C.inv_ptrs;
          for (int l=0; l<lvls; l++) {
            auto& S = C.inv[l];
            if (!S.on) continue;
            auto& O = offs[l];
            S.mL = di + O.mL; S.nLu = di + O.nLu; S.nUu = di + O.nUu; S.ldnb = di + O.ldnb;
            S.AL = dp + O.AL; S.AU = dp + O.AU; S.rhs = dp + O.rhs; S.scr = dp + O.scr; S.Li = dp + O.Li; S.Ui = dp + O.Ui;
          }
        }
      }
      if (pack_verbose)
        std::cout << "# F-PACK solve_cache_build_ms "
                  << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t_s0).count()
                  << " (level info + meta-data + uploads; once per structure with S-CACHE-KEEP)" << std::endl;
    }
    // S-LASWP: build the per-front permutations from the pivots (once per factorization: the factorization
    // clears perm_ready; S-CACHE-KEEP keeps the rest of this cache)
    static const bool solve_perm = !(std::getenv("STRUM_SOLVE_PERM") && std::atoi(std::getenv("STRUM_SOLVE_PERM")) == 0);
    if (solve_perm && !C.perm_ready) {
      const auto t_p0 = std::chrono::steady_clock::now();
      if (!C.d_perm.size()) {
        std::size_t tot = 0;
        for (int l=0; l<lvls; l++) tot += ldata[l].piv_size;
        C.d_perm = gpu::DeviceMemory<int>(tot + 1);
        C.d_perm_ptrs = gpu::DeviceMemory<int*>(Ntotal + 1);
        std::vector<int*> hp(Ntotal + 1, nullptr);
        C.perm_batch.assign(lvls, nullptr);
        int* base = C.d_perm;
        std::size_t off = 0;
        for (int l=0; l<lvls; l++) {
          auto& L = ldata[l];
          C.perm_batch[l] = C.d_perm_ptrs.template as<int*>() + l_off[l];
          for (std::size_t i=0; i<L.f.size(); i++) { hp[l_off[l]+i] = base + off; off += L.f[i]->dim_sep(); }
        }
        gpu::copy_host_to_device<int*>(C.d_perm_ptrs, hp.data(), Ntotal + 1);
      }
      for (int l=lvls-1; l>=0; l--) {
        auto& L = ldata[l];
        if (L.Nsmall) gpu::ipiv_to_perm_vbatched(handle, L.dev_ipiv_batch, L.dev_d1_batch, C.perm_batch[l], L.Nsmall);
      }
      C.perm_ready = true;
      if (pack_verbose)
        std::cout << "# F-PACK solve_perm_rebuild_ms "
                  << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t_p0).count() << std::endl;
    }
    // S-INVDIAG: the inverses of THIS factorization must be complete before the solve's stream touches them (the
    // inversion stream is non-blocking and S-CACHE-KEEP keeps this cache across refactorizations: per epoch)
    if (!C.inv_synced) {
      if (factor_cache_)
        for (int l=0; l<int(factor_cache_->inv.size()); l++)
          if (factor_cache_->inv[l].on && factor_cache_->inv_done[l]) factor_cache_->inv_done[l]->wait(C.comp_stream);
      C.inv_synced = true;
    }
    C.n_solves++;
    // copy rhs, from pageable memory (input)
    // S-RHSPIN (magma-hpc, 2026-10-27): the 1.6 MB rhs went through a pageable cudaMemcpy each way (driver-side
    // staging + a full device sync); now memcpy into a pinned buffer kept in the solve cache and an async copy on
    // the compute stream. STRUM_RHS_PIN=0 restores the pageable copies.
    static const bool rhs_pin = !(std::getenv("STRUM_RHS_PIN") && std::atoi(std::getenv("STRUM_RHS_PIN")) == 0);
    const std::size_t nb_elems = std::size_t(b.rows()) * nrhs;
    if (dp) {
      // S-HOST (magma-hpc, 2026-10-27): the right-hand side arrives RAW; stage it pinned, upload it and apply the
      // sparse reordering on the device, d_b(i,:) = braw(iperm[i],:), instead of the host pass in transform_b.
      const DenseM_t& br = *dp->braw;
      if (C.h_b.size() < nb_elems) C.h_b = gpu::HostMemory<scalar_t>(nb_elems);
      if (C.d_braw.size() < nb_elems) C.d_braw = gpu::DeviceMemory<scalar_t>(nb_elems);
      scalar_t* hb = C.h_b;
      scalar_t* d_braw = C.d_braw;
      if (br.ld() == br.rows()) std::copy(br.data(), br.data() + nb_elems, hb);
      else for (int j=0; j<nrhs; j++) std::copy(br.ptr(0, j), br.ptr(0, j) + br.rows(), hb + std::size_t(j) * br.rows());
      gpu::copy_host_to_device_async<scalar_t>(d_braw, hb, nb_elems, C.comp_stream);
      gpu::gather_rows<scalar_t>(handle, b.rows(), nrhs, d_braw, d_b, dp->d_iperm);
    } else if (rhs_pin) {
      if (C.h_b.size() < nb_elems) C.h_b = gpu::HostMemory<scalar_t>(nb_elems);
      scalar_t* hb = C.h_b;
      if (b.ld() == b.rows()) std::copy(b.data(), b.data() + nb_elems, hb);
      else for (int j=0; j<nrhs; j++) std::copy(b.ptr(0, j), b.ptr(0, j) + b.rows(), hb + std::size_t(j) * b.rows());
      gpu::copy_host_to_device_async<scalar_t>(d_b, hb, nb_elems, C.comp_stream);
    } else
      gpu::copy_host_to_device<scalar_t>(d_b, b);

    // S-NOSYNC (magma-hpc, 2026-10-27): the per-level host-side synchronizations below are not needed for
    // correctness: comp_stream is a blocking stream (cudaStreamCreate), so its kernels are ordered against the
    // memsets / extend-add / extract kernels on the legacy default stream implicitly. Each host sync drained the
    // GPU and exposed the next level's launch latency (61 syncs, ~2-4 ms of idle per solve). STRUM_SOLVE_SYNC=1
    // restores them (A/B arm).
    static const bool solve_sync = std::getenv("STRUM_SOLVE_SYNC") && std::atoi(std::getenv("STRUM_SOLVE_SYNC")) == 1;
    // forward solve
    for (int l=lvls-1; l>=0; l--) {
      if (solve_sync) gpu::synchronize_default_stream();
      std::string rname = "fwd l=" + std::to_string(l);
      invdiag::Range rlev(rname.c_str());
      auto& L = ldata[l];
      std::size_t N = L.f.size(), Nsmall = L.Nsmall;
      // batch meta-data was uploaded once, see above (L.dev_*_batch)
      auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
      auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
      auto F11 = L.dev_F_batch;   auto F21 = F11 + 2*Nsmall;
      {
      invdiag::Range rea("ea");
      gpu::memset<scalar_t>
        ((l % 2) ? d_bupd_odd : d_bupd_even, 0, L.total_upd_size*nrhs);
      if (l != lvls-1)
        gpu::extend_add_rhs
          (b.rows(), nrhs, N, &h_asmbl[l_off[l]], d_asmbl+l_off[l]);
      }
      // TODO remove?
      if (solve_sync) gpu::synchronize_default_stream();
      invdiag::Range* rsub = new invdiag::Range("perm");
      if (!(C.perm_ready && Nsmall &&
            gpu::permute_rows_fwd_vbatched<scalar_t>
            (handle, nrhs_batch+Nmax-Nsmall, d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
             C.perm_batch[l], d1, L.max_d1_small, Nsmall)))
        gpu::laswp_fwd_vbatched
          (handle, nrhs_batch+Nmax-Nsmall, nrhs,
           d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
           L.dev_ipiv_batch, d1, Nsmall);
      delete rsub;
      const auto* IV = (nrhs == 1 && !C.inv.empty() && C.inv[l].on) ? &C.inv[l] : nullptr;
      if (IV) {
        // v2 fused path: one kernel per step (real types, NB 128/256), otherwise the two vbatched gemvs per step
        const bool fused = invdiag::fused() && (IV->nb == 128 || IV->nb == 256);
        const int chunk0 = invdiag::chunk();
        const bool pf = invdiag::prefetch();
        auto fused_step = [&](const int* m, const int* n, int maxn, scalar_t** A, scalar_t** x, scalar_t** yin,
                              scalar_t** inv, scalar_t** out, const int* mn, const int* nn, scalar_t** An, bool lower) {
          const int chunk = std::max(chunk0, (maxn + 63) / 64);
          const int S = maxn > 0 ? (maxn + chunk - 1) / chunk : 0;
          return gpu::invdiag_step<scalar_t>(handle, N, S, IV->nb, chunk, m, n, A, ld1, x, yin, inv, out,
                                             C.d_partial, C.d_counter, pf ? mn : nullptr, nn, An, lower);
        };
        // S-INVDIAG: block recurrence with the explicit inverses of the diagonal blocks (all fronts of the level
        // in one vbatched gemv per step). L11: x_k = Linv_kk (b_k - L_{k,0:k} x_{0:k}), the b blocks are updated
        // in place in the rhs and x goes to the scratch vector; U11: x_k = Uinv_kk (y_k - U_{k,k+1:} x_{k+1:}),
        // y in the scratch updated in place, x back into the rhs.
        const int NB = IV->nb, K = IV->nblk;
        auto inc = inc_batch+Nmax-Nsmall;
        invdiag::Range* rl = new invdiag::Range(fused ? "L11 fused" : "L11 vbatched");
        for (int k=0; k<K; k++) {
          if (fused && fused_step(IV->mL+k*N, IV->nLu+k*N, IV->maxLu[k], IV->AL+k*N, IV->scr, IV->rhs+k*N,
                                  IV->Li+k*N, IV->scr+k*N,
                                  k+1 < K ? IV->mL+(k+1)*N : nullptr, IV->nLu+std::min(k+1,K-1)*N, IV->AL+std::min(k+1,K-1)*N, true)) continue;
          if (k > 0 && IV->maxLu[k] > 0)
            gpu::magma::gemv_vbatched_max_nocheck
              (MagmaNoTrans, IV->mL+k*N, IV->nLu+k*N, scalar_t(-1.), IV->AL+k*N, ld1, IV->scr, inc,
               scalar_t(1.), IV->rhs+k*N, inc, N, NB, IV->maxLu[k], handle);
          gpu::magma::gemv_vbatched_max_nocheck
            (MagmaNoTrans, IV->mL+k*N, IV->mL+k*N, scalar_t(1.), IV->Li+k*N, IV->ldnb, IV->rhs+k*N, inc,
             scalar_t(0.), IV->scr+k*N, inc, N, NB, NB, handle);
        }
        delete rl;
        if (IV->U) {
          invdiag::Range ru(fused ? "U11 fused" : "U11 vbatched");
          for (int k=K-1; k>=0; k--) {
            bool upd = k < K-1 && IV->maxUu[k] > 0;
#if defined(STRUM_PLANT_INVDIAG)
            if (l == 0 && k == 0) upd = false;   // PLANT invdiag-skip-root-u0: the root's first block misses its U update
#endif
            if (fused && fused_step(IV->mL+k*N, upd ? IV->nUu+k*N : IV->nLu, upd ? IV->maxUu[k] : 0, IV->AU+k*N,
                                    IV->rhs+std::min(k+1, K-1)*N, IV->scr+k*N, IV->Ui+k*N, IV->rhs+k*N,
                                    k > 0 ? IV->mL+(k-1)*N : nullptr, IV->nUu+std::max(k-1,0)*N, IV->AU+std::max(k-1,0)*N, false)) continue;
            if (upd)
              gpu::magma::gemv_vbatched_max_nocheck
                (MagmaNoTrans, IV->mL+k*N, IV->nUu+k*N, scalar_t(-1.), IV->AU+k*N, ld1, IV->rhs+(k+1)*N, inc,
                 scalar_t(1.), IV->scr+k*N, inc, N, NB, IV->maxUu[k], handle);
            gpu::magma::gemv_vbatched_max_nocheck
              (MagmaNoTrans, IV->mL+k*N, IV->mL+k*N, scalar_t(1.), IV->Ui+k*N, IV->ldnb, IV->scr+k*N, inc,
               scalar_t(0.), IV->rhs+k*N, inc, N, NB, NB, handle);
          }
        } else {
          // y = L^{-1} b back into the rhs, then the U11 solve on the trsm path
          invdiag::Range ru("U11 trsm");
          gpu::magma::lacpy_vbatched
            (MagmaFull, L.max_d1_small, 1, d1, nrhs_batch+Nmax-Nsmall, (const scalar_t* const*)IV->scr, ld1,
             IV->rhs, ld1, N, handle);
          gpu::magma::trsm_vbatched_max_nocheck
            (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
             L.max_d1_small, nrhs, d1, nrhs_batch+Nmax-Nsmall, scalar_t(1.),
             F11, ld1, d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
             Nsmall, handle);
        }
      } else {
      invdiag::Range rt("L11+U11 trsm");
      gpu::magma::trsm_vbatched_max_nocheck
        (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
         L.max_d1_small, nrhs, d1, nrhs_batch+Nmax-Nsmall, scalar_t(1.),
         F11, ld1, d_rhs_batch[l], ldrhs_batch+Nmax-
         Nsmall, Nsmall, handle);
      gpu::magma::trsm_vbatched_max_nocheck
        (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
         L.max_d1_small, nrhs, d1, nrhs_batch+Nmax-Nsmall, scalar_t(1.),
         F11, ld1, d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
         Nsmall, handle);
      }
      for (std::size_t i=Nsmall; i<N; i++) {
        DenseMW_t fb(L.f[i]->dim_sep(), nrhs, h_rhs_batch[l][i], b.rows());
        gpu::getrs(handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_, fb, getrs_err);
      }
      if (l == 0) break;
      invdiag::Range rg("F21");
      if (nrhs == 1)
        gpu::magma::gemv_vbatched_max_nocheck
          (MagmaNoTrans, d2, d1, scalar_t(-1.),
           F21, ld2, d_rhs_batch[l], inc_batch+Nmax-Nsmall, scalar_t(1.),
           d_bupd_batch[l], inc_batch+Nmax-Nsmall, Nsmall,
           L.max_d2_small, L.max_d1_small, handle);
      else
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans, d2, nrhs_batch+Nmax-Nsmall, d1,
           scalar_t(-1.), F21, ld2, d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
           scalar_t(1.), d_bupd_batch[l], ld2, Nsmall,
           L.max_d2_small, nrhs, L.max_d1_small, handle);
      for (std::size_t i=Nsmall; i<N; i++) {
        DenseMW_t fb(L.f[i]->dim_sep(), nrhs, h_rhs_batch[l][i], b.rows()),
          fbu(L.f[i]->dim_upd(), nrhs, h_bupd_batch[l][i], L.f[i]->dim_upd());
        if (nrhs == 1)
          gpu::gemv(handle, Trans::N, scalar_t(-1.),
                    L.f[i]->F21_, fb, scalar_t(1.), fbu);
        else
          gpu::gemm(handle, Trans::N, Trans::N, scalar_t(-1.),
                    L.f[i]->F21_, fb, scalar_t(1.), fbu);
      }
    }
    if (solve_sync) gpu::synchronize_default_stream();


    // backward solve
    for (int l=0; l<lvls; l++) {
      std::string rname = "bwd l=" + std::to_string(l);
      invdiag::Range rlev(rname.c_str());
      auto& L = ldata[l];
      std::size_t N = L.f.size(), Nsmall = L.Nsmall;
      auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
      auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
      auto F12 = L.dev_F_batch + Nsmall;
      if (solve_sync) gpu::synchronize_default_stream();
      if (l != 0) {
        invdiag::Range rg("F12");
        if (nrhs == 1)
          gpu::magma::gemv_vbatched_max_nocheck
            (MagmaNoTrans, d1, d2, scalar_t(-1.), F12, ld1,
             d_bupd_batch[l], inc_batch+Nmax-Nsmall, scalar_t(1.),
             d_rhs_batch[l], inc_batch+Nmax-Nsmall, Nsmall,
             L.max_d1_small, L.max_d2_small, handle);
        else
          gpu::magma::gemm_vbatched_max_nocheck
            (MagmaNoTrans, MagmaNoTrans, d1, nrhs_batch+Nmax-Nsmall, d2,
             scalar_t(-1.), F12, ld1, d_bupd_batch[l], ld2, scalar_t(1.),
             d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall, Nsmall,
             L.max_d1_small, nrhs, L.max_d2_small, handle);
        for (std::size_t i=Nsmall; i<N; i++) {
          DenseMW_t fb(L.f[i]->dim_sep(), nrhs, h_rhs_batch[l][i], b.rows()),
            fbu(L.f[i]->dim_upd(), nrhs, h_bupd_batch[l][i], L.f[i]->dim_upd());
          if (nrhs == 1)
            gpu::gemv(handle, Trans::N, scalar_t(-1.),
                      L.f[i]->F12_, fbu, scalar_t(1.), fb);
          else
            gpu::gemm(handle, Trans::N, Trans::N, scalar_t(-1.),
                      L.f[i]->F12_, fbu, scalar_t(1.), fb);
        }
      }
      if (solve_sync) gpu::synchronize_default_stream();
      if (l != lvls-1) {
        invdiag::Range rx("extract");
        gpu::extract_rhs
          (b.rows(), nrhs, N, &h_asmbl[l_off[l]], d_asmbl+l_off[l]);
      }
    }
    if (dp) {
      // S-HOST: undo the reordering on the device, x(i,:) = y(perm[i],:), then download (replaces transform_x)
      scalar_t* hb = C.h_b;
      scalar_t* d_braw = C.d_braw;
      gpu::gather_rows<scalar_t>(handle, b.rows(), nrhs, d_b, d_braw, dp->d_perm);
      gpu::copy_device_to_host_async<scalar_t>(hb, d_braw, nb_elems, C.comp_stream);
      C.comp_stream.synchronize();
      if (b.ld() == b.rows()) std::copy(hb, hb + nb_elems, b.data());
      else for (int j=0; j<nrhs; j++) std::copy(hb + std::size_t(j) * b.rows(), hb + std::size_t(j+1) * b.rows(), b.ptr(0, j));
    } else if (rhs_pin) {
      scalar_t* hb = C.h_b;
      gpu::copy_device_to_host_async<scalar_t>(hb, d_b, nb_elems, C.comp_stream);
      C.comp_stream.synchronize();
      if (b.ld() == b.rows()) std::copy(hb, hb + nb_elems, b.data());
      else for (int j=0; j<nrhs; j++) std::copy(hb + std::size_t(j) * b.rows(), hb + std::size_t(j+1) * b.rows(), b.ptr(0, j));
    } else {
      gpu::synchronize_default_stream();
      gpu::copy_device_to_host<scalar_t>(b, d_b);
    }
    static const bool l2log = launch2_env("STRUM_LAUNCH2_LOG");
    static const bool l2ver = launch2_env("STRUM_LAUNCH2_VERIFY");
    if (l2log || l2ver)
      std::cout << "# LAUNCH2 solve " << C.n_solves << " nrhs " << nrhs
                << " rows " << b.rows() << " cache "
                << (fresh ? "MISS(upload)" : "HIT(no upload)")
                << " mem_uploads " << C.n_mem_uploads << " meta_uploads "
                << C.n_meta_uploads << std::endl;
    if (l2ver) {
      // read the cached device meta-data back and compare with the host
      // copy; the one int at d_batch_int[3*(Nmax+1)] is the getrs error
      // flag written by the device, it is not input and is skipped
      std::vector<char> back(h_mem_bytes);
      gpu::copy_device_to_host<char>(back.data(), d_mem, h_mem_bytes);
      const std::size_t skip = reinterpret_cast<char*>
        (h_batch_int + 3*(Nmax+1)) - h_mem.data();
      std::size_t bad_mem = 0, bad_meta = 0;
      for (std::size_t i=0; i<h_mem_bytes; i++)
        if (!(i >= skip && i < skip+sizeof(int)) && back[i] != h_mem.data()[i])
          bad_mem++;
      for (int l=0; l<lvls; l++) {
        auto& L = ldata[l];
        std::vector<char> mb(L.batch_meta_bytes);
        gpu::copy_device_to_host<char>
          (mb.data(), C.d_meta.template as<char>() + meta_off[l],
           L.batch_meta_bytes);
        bad_meta += std::size_t(std::memcmp
          (mb.data(), L.dev_batch_meta.data(), L.batch_meta_bytes) != 0);
      }
      std::cout << "# LAUNCH2 verify solve " << C.n_solves
                << " mem_bytes_differ " << bad_mem << " levels_meta_differ "
                << bad_meta << std::endl;
    }
  }

  // explicit template instantiations
  template class FrontMAGMA<float,int>;
  template class FrontMAGMA<double,int>;
  template class FrontMAGMA<std::complex<float>,int>;
  template class FrontMAGMA<std::complex<double>,int>;

  template class FrontMAGMA<float,long int>;
  template class FrontMAGMA<double,long int>;
  template class FrontMAGMA<std::complex<float>,long int>;
  template class FrontMAGMA<std::complex<double>,long int>;

  template class FrontMAGMA<float,long long int>;
  template class FrontMAGMA<double,long long int>;
  template class FrontMAGMA<std::complex<float>,long long int>;
  template class FrontMAGMA<std::complex<double>,long long int>;

} // end namespace strumpack
