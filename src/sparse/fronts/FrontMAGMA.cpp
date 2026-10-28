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
#include <string>
#include <cstdlib>
#if defined(STRUMPACK_USE_CUDA)
#include <cuda_runtime_api.h>
#endif
#include <cstring>
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

  // F-GRAPH (magma-hpc F track, 2026-10-27): CUDA-graph capture/replay of the
  // factorization levels and of the solve. Every refactorization with the same
  // sparsity issues the same kernel sequence with the same arguments (the
  // factor/work/meta device buffers are persistent since F-GATHER), so each
  // level is captured once (cudaStreamBeginCapture .. EndCapture ->
  // cudaGraphInstantiate) on the first factorization and replayed with one
  // cudaGraphLaunch afterwards; same for the whole solve sequence.
  //   STRUM_GRAPH=1       per-level factorization graphs (default 0 = off)
  //   STRUM_GRAPH=2       one graph for the whole factorization
  //   STRUM_GRAPH_SOLVE   solve graph (default = STRUM_GRAPH != 0)
  //   STRUM_GRAPH_RELAXED capture mode relaxed instead of thread-local
  //   STRUM_GRAPH_LOG=1   print graph sizes / instantiate times
  static int graph_env_int(const char* name, int def) {
    const char* e = std::getenv(name);
    return (e && e[0]) ? std::atoi(e) : def;
  }
  static int graph_mode() {
    static const int m = graph_env_int("STRUM_GRAPH", 0);
    return m;
  }
  static bool graph_solve_on() {
    static const bool on = graph_env_int("STRUM_GRAPH_SOLVE", graph_mode() != 0 ? 1 : 0) != 0;
    return on;
  }
  static bool graph_log() {
    static const bool on = graph_env_int("STRUM_GRAPH_LOG", 0) != 0;
    return on;
  }
#if defined(STRUMPACK_USE_CUDA)
  struct GraphExecHolder {
    cudaGraphExec_t exec = nullptr;
    std::size_t nodes = 0;
    double instantiate_ms = 0.;
    std::size_t launches = 0;
    GraphExecHolder() = default;
    GraphExecHolder(const GraphExecHolder&) = delete;
    GraphExecHolder& operator=(const GraphExecHolder&) = delete;
    GraphExecHolder(GraphExecHolder&& o) noexcept { *this = std::move(o); }
    GraphExecHolder& operator=(GraphExecHolder&& o) noexcept {
      if (this != &o) { reset(); exec = o.exec; nodes = o.nodes; instantiate_ms = o.instantiate_ms;
        launches = o.launches; o.exec = nullptr; }
      return *this;
    }
    ~GraphExecHolder() { reset(); }
    bool valid() const { return exec != nullptr; }
    void reset() {
      if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
      nodes = 0; launches = 0;
    }
  };
  static bool graph_begin(gpu::Stream& s) {
    static const bool relaxed = graph_env_int("STRUM_GRAPH_RELAXED", 0) != 0;
    auto e = cudaStreamBeginCapture
      (gpu::get_cuda_stream(s), relaxed ? cudaStreamCaptureModeRelaxed : cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
      std::cerr << "# F-GRAPH: cudaStreamBeginCapture failed: " << cudaGetErrorString(e) << std::endl;
      (void)cudaGetLastError();
      return false;
    }
    gpu::set_capture_tolerant(true);
    return true;
  }
  // ends the capture and instantiates; false (and nothing executed) if the capture was invalidated
  static bool graph_end(gpu::Stream& s, GraphExecHolder& G, const char* what) {
    cudaGraph_t g = nullptr;
    auto e = cudaStreamEndCapture(gpu::get_cuda_stream(s), &g);
    gpu::set_capture_tolerant(false);
    if (e != cudaSuccess || !g) {
      std::cerr << "# F-GRAPH: capture of " << what << " failed: " << cudaGetErrorString(e)
                << " (running the sequence eagerly)" << std::endl;
      (void)cudaGetLastError();
      if (g) cudaGraphDestroy(g);
      return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    cudaGraphGetNodes(g, nullptr, &G.nodes);
    e = cudaGraphInstantiate(&G.exec, g, 0);
    cudaGraphDestroy(g);
    if (e != cudaSuccess) {
      std::cerr << "# F-GRAPH: cudaGraphInstantiate of " << what << " failed: " << cudaGetErrorString(e) << std::endl;
      (void)cudaGetLastError();
      G.exec = nullptr; G.nodes = 0;
      return false;
    }
    G.instantiate_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (graph_log())
      std::cout << "# F-GRAPH " << what << " nodes " << G.nodes << " instantiate_ms " << G.instantiate_ms << std::endl;
    return true;
  }
  // STRUM_GRAPH_TRACE=1: report the capture status after each step (debugging aid)
  static void graph_trace(const char* where, gpu::Stream& s) {
    static const bool on = graph_env_int("STRUM_GRAPH_TRACE", 0) != 0;
    if (!on) return;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    auto e = cudaStreamIsCapturing(gpu::get_cuda_stream(s), &st);
    std::cout << "# F-GRAPH trace " << where << " status " << int(st) << " err " << int(e)
              << " last " << int(cudaPeekAtLastError()) << std::endl;
  }
  static void graph_launch(GraphExecHolder& G, gpu::Stream& s) {
    gpu_check(cudaGraphLaunch(G.exec, gpu::get_cuda_stream(s)));
    G.launches++;
  }
#endif

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
#if defined(STRUMPACK_USE_CUDA)
    GraphExecHolder graph;        // the whole solve (H2D of the staged rhs .. D2H of the solution)
    const void* gkey[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};   // pointers the captured graph bakes in
#endif
  };

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
    // F-GRAPH: persistent stream + handle (the cuBLAS/cuSOLVER/MAGMA handles and their workspaces were created and
    // destroyed at every factorization: the 7 cudaMalloc / 22 cudaFree per refactorization), the per-level (or
    // whole-factorization) executable graphs and a factorization counter (replays start with the second one).
    std::unique_ptr<gpu::Stream> stream;
    std::unique_ptr<gpu::Handle> handle;
#if defined(STRUMPACK_USE_CUDA)
    std::vector<GraphExecHolder> graphs;
#endif
    const scalar_t* graph_vals = nullptr;   // value pointer the graphs were captured with
    std::size_t nfact = 0;
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
      solve_cache_.reset();   // new factor memory: the solve workspace (and its F-GRAPH graph) must be rebuilt
      devperm_.reset();
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
      // synchronization would wait for the asynchronous value upload of F-DEVUPDATE. (F-GRAPH's solve graph
      // bakes the cache's device pointers in, so it survives with the cache.)
      solve_cache_->perm_ready = false;
    } else {
      solve_cache_.reset(); // new factors: rebuild the solve workspace
      devperm_.reset();     // S-HOST: and the device permutation
    }
    const auto t_f1 = std::chrono::steady_clock::now();
    auto& C = *factor_cache_;
    const int gmode = graph_mode();           // 0 off, 1 per-level graphs, 2 one graph for all levels
    const bool gsolve = graph_solve_on();
    // F-GRAPH: persistent stream + handle in graph mode (the graphs are replayed on the same stream; the
    // per-factorization cuBLAS/cuSOLVER/MAGMA handle creation and its allocations go away)
    if ((gmode != 0 || gsolve) && !C.stream) {
      C.stream.reset(new gpu::Stream());
      C.handle.reset(new gpu::Handle(*C.stream));
    }
    std::unique_ptr<gpu::Stream> tmp_stream;
    std::unique_ptr<gpu::Handle> tmp_handle;
    if (!C.stream) {
      tmp_stream.reset(new gpu::Stream());
      tmp_handle.reset(new gpu::Handle(*tmp_stream));
    }
    gpu::Stream& comp_stream = C.stream ? *C.stream : *tmp_stream;
    gpu::Handle& handle = C.handle ? *C.handle : *tmp_handle;
    // gs != nullptr: every device operation of a level is issued on comp_stream (capturable); otherwise the
    // original mix of legacy-default-stream kernels (memset/gather/assemble) and MAGMA on comp_stream is kept.
    gpu::Stream* gs = gmode ? &comp_stream : nullptr;
    if (pack_verbose) {   // item-5 audit: host time before the values are touched
      const auto t_f2 = std::chrono::steady_clock::now();
      std::cout << "# F-PACK factor_start solve_cache_reset_ms "
                << std::chrono::duration<double,std::milli>(t_f1 - t_f0).count()
                << " stream_handle_create_ms " << std::chrono::duration<double,std::milli>(t_f2 - t_f1).count() << std::endl;
    }
    // F-DEVUPDATE: when the value update ran on the device, the permuted values are already there
    // (the source waits for its upload and verifies it); nothing is copied at the start of the factorization.
    // F-GRAPH: the capture stream waits for the upload instead of the host; the verification is completed
    // after the graphs are launched (a repair then re-launches them).
    const scalar_t* vals = nullptr;
    DeviceValueSource<scalar_t>* dsrc = A.device_value_source();
    unsigned long long dgen = 0;
    if (dsrc) vals = gs ? dsrc->device_values_on_stream(C.nnz, &dgen, gpu::get_cuda_stream(*gs))
                        : dsrc->device_values(C.nnz, &dgen);
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
        if (!gs) comp_stream.synchronize();   // the gathers below run on the default stream
      } else
        gpu::copy_host_to_device<scalar_t>(C.d_vals, A.val(), C.nnz);
    }
    scalar_t* fbase = dev_factors_->template as<scalar_t>();
    static const int large_lu = std::getenv("STRUM_LARGE_LU") ?
      std::atoi(std::getenv("STRUM_LARGE_LU")) : 0;   // default OFF (window 102: +3.4 %, NO-GO); 0 = off
    static const int large_gemm_d2 = std::getenv("STRUM_LARGE_GEMM") ?
      std::atoi(std::getenv("STRUM_LARGE_GEMM")) : 0;   // default OFF (window 102: +4.2 %, NO-GO); 0 = off
    // F-TOPLU goes through magma_Xgetrf_native (own queues, host pivot copies): not capturable
    auto level_capturable = [&](int l) {
      return !(large_lu > 0 && C.ldata[l].max_d1_small > large_lu);
    };
    // the GPU work of one level: zero, gather the sparse entries, extend-add, LU, laswp, trsm, Schur update
    auto run_level = [&](int l) {
      auto& L = C.ldata[l];
      try {
        if (gs) {
          gpu::memset_async<scalar_t>(C.level_work[l], 0, L.Schur_size, *gs);
          gpu::memset_async<scalar_t>(C.level_factors[l], 0, L.factor_size, *gs);
          gpu::gather_front_elements<scalar_t>
            (C.map_off[l+1]-C.map_off[l], C.d_src + C.map_off[l],
             C.d_dst + C.map_off[l], vals, fbase, *gs);
          gpu::assemble(L.f.size(), C.h_asmbl[l], C.d_asmbl[l], *gs);
          graph_trace("level:assemble", *gs);
        } else {
          gpu::memset<scalar_t>(C.level_work[l], 0, L.Schur_size);
          gpu::memset<scalar_t>(C.level_factors[l], 0, L.factor_size);
          // default stream: gather of this level's sparse entries, then extend-add
          gpu::gather_front_elements<scalar_t>
            (C.map_off[l+1]-C.map_off[l], C.d_src + C.map_off[l],
             C.d_dst + C.map_off[l], vals, fbase);
          gpu::assemble(L.f.size(), C.h_asmbl[l], C.d_asmbl[l]);
        }

        std::size_t N = L.f.size(), Nsmall = L.Nsmall;
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F11 = L.dev_F_batch;   auto F12 = F11 + Nsmall;
        auto F21 = F12 + Nsmall;    auto F22 = F21 + Nsmall;

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
               opts.pivot_threshold(), gs);
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
    };
    auto count_flops = [&](int l) {
      long long level_flops, small_flops;
      C.ldata[l].flops(level_flops, small_flops);
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      STRUMPACK_FLOPS(small_flops);
    };
#if defined(STRUMPACK_USE_CUDA)
    // the graphs bake the value pointer in: a switch between the device source and the host copy (or a
    // reallocated source buffer) invalidates them
    if (gmode != 0 && vals != C.graph_vals) { C.graphs.clear(); C.graph_vals = vals; }
#endif
    auto run_all = [&]() {
#if defined(STRUMPACK_USE_CUDA)
    if (gmode == 2) {
      // one graph for the whole factorization
      if (C.graphs.size() < 1) C.graphs.resize(1);
      auto& G = C.graphs[0];
      bool all_capturable = true;
      for (int l=0; l<C.lvls; l++) all_capturable = all_capturable && level_capturable(l);
      if (G.valid()) graph_launch(G, comp_stream);
      else {
        bool captured = all_capturable && graph_begin(comp_stream);
        if (captured) {
          for (int l=C.lvls-1; l>=0; l--) run_level(l);
          captured = graph_end(comp_stream, G, "factorization");
        }
        if (captured) graph_launch(G, comp_stream);
        else for (int l=C.lvls-1; l>=0; l--) run_level(l);
      }
      for (int l=C.lvls-1; l>=0; l--) count_flops(l);
    } else if (gmode != 0) {
      // one graph per level
      if (C.graphs.size() < std::size_t(C.lvls)) C.graphs.resize(C.lvls);
      for (int l=C.lvls-1; l>=0; l--) {
        auto& G = C.graphs[l];
        if (G.valid()) {
          int lg = l;
#if defined(STRUM_PLANT_GRAPH)
          // PLANT: on replays, level lvls/2 is "updated" with the graph of the level below it (idempotent for
          // that level, so level lvls/2 keeps the factors of the previous factorization): must FAIL the compare
          if (C.nfact > 0 && l == C.lvls/2 && l >= 1 && C.graphs[l-1].valid()) lg = l-1;
#endif
          graph_launch(C.graphs[lg], comp_stream);
        } else {
          bool captured = level_capturable(l) && graph_begin(comp_stream);
          if (captured) {
            run_level(l);
            captured = graph_end(comp_stream, G, (std::string("level ") + std::to_string(l)).c_str());
          }
          if (captured) graph_launch(G, comp_stream);
          else run_level(l);
        }
        count_flops(l);
      }
    } else
#endif
    for (int l=C.lvls-1; l>=0; l--) {
      run_level(l);
      count_flops(l);
    }
    };
    run_all();
    C.nfact++;
    if (dsrc && gs && !dsrc->verify_pending_values()) {
      // the deferred verification repaired the device values: the graphs above read wrong ones, replay
      std::cerr << "# F-GRAPH: device values were repaired after the launch, re-issuing the factorization" << std::endl;
      run_all();
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
    if (gmode) {
      // F-GRAPH: with 15 graph launches instead of ~20k kernel launches the host no longer runs at the GPU's
      // pace (launch-queue backpressure made the eager path return with the GPU nearly done): wait here so that
      // factor() keeps its "factors ready on return" timing semantics and the first solve does not absorb the
      // factorization. STRUM_GRAPH_NOSYNC=1 returns asynchronously.
      static const bool nosync = graph_env_int("STRUM_GRAPH_NOSYNC", 0) != 0;
      if (!nosync) comp_stream.synchronize();
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
#if defined(STRUMPACK_USE_CUDA)
    C.graphs.clear();   // F-GRAPH: captured on the first factorization with this structure
#endif
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
    C.n_solves++;
    // copy rhs, from pageable memory (input)
    // S-RHSPIN (magma-hpc, 2026-10-27): the 1.6 MB rhs went through a pageable cudaMemcpy each way (driver-side
    // staging + a full device sync); now memcpy into a pinned buffer kept in the solve cache and an async copy on
    // the compute stream. STRUM_RHS_PIN=0 restores the pageable copies.
    static const bool rhs_pin = !(std::getenv("STRUM_RHS_PIN") && std::atoi(std::getenv("STRUM_RHS_PIN")) == 0);
    const std::size_t nb_elems = std::size_t(b.rows()) * nrhs;
    // S-NOSYNC (magma-hpc, 2026-10-27): the per-level host-side synchronizations below are not needed for
    // correctness: comp_stream is a blocking stream (cudaStreamCreate), so its kernels are ordered against the
    // memsets / extend-add / extract kernels on the legacy default stream implicitly. Each host sync drained the
    // GPU and exposed the next level's launch latency (61 syncs, ~2-4 ms of idle per solve). STRUM_SOLVE_SYNC=1
    // restores them (A/B arm).
    static const bool solve_sync = std::getenv("STRUM_SOLVE_SYNC") && std::atoi(std::getenv("STRUM_SOLVE_SYNC")) == 1;
    // F-GRAPH: with the rhs staged in the persistent pinned buffer, the whole device sequence (H2D, reorder,
    // forward, backward, reorder back, D2H) has invariant arguments: capture it once per solve cache, replay it.
    static const bool gsolve = graph_solve_on();
    // the first solve of a cache runs eagerly: MAGMA creates its workspace memory pools lazily on first use
    // (cudaMemPoolCreate inside a capture invalidates it); the capture is taken from the second solve on
    bool gcap = gsolve && (dp || rhs_pin) && C.n_solves > 1;
    // staging buffers (allocated outside any capture; their addresses are baked into the graph)
    if (dp || rhs_pin) { if (C.h_b.size() < nb_elems) C.h_b = gpu::HostMemory<scalar_t>(nb_elems); }
    if (dp) { if (C.d_braw.size() < nb_elems) C.d_braw = gpu::DeviceMemory<scalar_t>(nb_elems); }
    scalar_t* hb = (dp || rhs_pin) ? static_cast<scalar_t*>(C.h_b) : nullptr;
    scalar_t* d_braw = dp ? static_cast<scalar_t*>(C.d_braw) : nullptr;
    if (dp) {
      // S-HOST (magma-hpc, 2026-10-27): the right-hand side arrives RAW; stage it pinned, upload it and apply the
      // sparse reordering on the device, d_b(i,:) = braw(iperm[i],:), instead of the host pass in transform_b.
      const DenseM_t& br = *dp->braw;
      if (br.ld() == br.rows()) std::copy(br.data(), br.data() + nb_elems, hb);
      else for (int j=0; j<nrhs; j++) std::copy(br.ptr(0, j), br.ptr(0, j) + br.rows(), hb + std::size_t(j) * br.rows());
    } else if (rhs_pin) {
      if (b.ld() == b.rows()) std::copy(b.data(), b.data() + nb_elems, hb);
      else for (int j=0; j<nrhs; j++) std::copy(b.ptr(0, j), b.ptr(0, j) + b.rows(), hb + std::size_t(j) * b.rows());
    }

    // the device part of the solve; on_stream: everything on comp_stream (capturable), else the original mix
    auto issue_device = [&](bool on_stream) {
      if (dp) {
        gpu::copy_host_to_device_async<scalar_t>(d_braw, hb, nb_elems, C.comp_stream);
        graph_trace("solve:h2d", C.comp_stream);
        gpu::gather_rows<scalar_t>(handle, b.rows(), nrhs, d_braw, d_b, dp->d_iperm);
        graph_trace("solve:gather", C.comp_stream);
      } else if (rhs_pin) {
        gpu::copy_host_to_device_async<scalar_t>(d_b, hb, nb_elems, C.comp_stream);
        graph_trace("solve:h2d", C.comp_stream);
      } else
        gpu::copy_host_to_device<scalar_t>(d_b, b);
      const bool sync = solve_sync && !on_stream;
      // forward solve
      for (int l=lvls-1; l>=0; l--) {
        if (sync) gpu::synchronize_default_stream();
        auto& L = ldata[l];
        std::size_t N = L.f.size(), Nsmall = L.Nsmall;
        // batch meta-data was uploaded once, see above (L.dev_*_batch)
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F11 = L.dev_F_batch;   auto F21 = F11 + 2*Nsmall;
        if (on_stream) {
          gpu::memset_async<scalar_t>
            ((l % 2) ? d_bupd_odd : d_bupd_even, 0, L.total_upd_size*nrhs, C.comp_stream);
          if (l != lvls-1)
            gpu::extend_add_rhs
              (b.rows(), nrhs, N, &h_asmbl[l_off[l]], d_asmbl+l_off[l], C.comp_stream);
        } else {
          gpu::memset<scalar_t>
            ((l % 2) ? d_bupd_odd : d_bupd_even, 0, L.total_upd_size*nrhs);
          if (l != lvls-1)
            gpu::extend_add_rhs
              (b.rows(), nrhs, N, &h_asmbl[l_off[l]], d_asmbl+l_off[l]);
        }
        // TODO remove?
        if (sync) gpu::synchronize_default_stream();
        graph_trace("solve:fwd memset/extend_add", C.comp_stream);
        if (!(C.perm_ready && Nsmall &&
              gpu::permute_rows_fwd_vbatched<scalar_t>
              (handle, nrhs_batch+Nmax-Nsmall, d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
               C.perm_batch[l], d1, L.max_d1_small, Nsmall)))
          gpu::laswp_fwd_vbatched
            (handle, nrhs_batch+Nmax-Nsmall, nrhs,
             d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
             L.dev_ipiv_batch, d1, Nsmall);
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
        for (std::size_t i=Nsmall; i<N; i++) {
          DenseMW_t fb(L.f[i]->dim_sep(), nrhs, h_rhs_batch[l][i], b.rows());
          gpu::getrs(handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_, fb, getrs_err);
        }
        graph_trace("solve:fwd perm/trsm", C.comp_stream);
        if (l == 0) break;
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
      if (sync) gpu::synchronize_default_stream();

      // backward solve
      for (int l=0; l<lvls; l++) {
        auto& L = ldata[l];
        std::size_t N = L.f.size(), Nsmall = L.Nsmall;
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F12 = L.dev_F_batch + Nsmall;
        if (sync) gpu::synchronize_default_stream();
        if (l != 0) {
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
        if (sync) gpu::synchronize_default_stream();
        if (l != lvls-1) {
          if (on_stream)
            gpu::extract_rhs
              (b.rows(), nrhs, N, &h_asmbl[l_off[l]], d_asmbl+l_off[l], C.comp_stream);
          else
            gpu::extract_rhs
              (b.rows(), nrhs, N, &h_asmbl[l_off[l]], d_asmbl+l_off[l]);
        }
      }
      if (dp) {
        // S-HOST: undo the reordering on the device, x(i,:) = y(perm[i],:), then download (replaces transform_x)
        gpu::gather_rows<scalar_t>(handle, b.rows(), nrhs, d_b, d_braw, dp->d_perm);
        gpu::copy_device_to_host_async<scalar_t>(hb, d_braw, nb_elems, C.comp_stream);
      } else if (rhs_pin)
        gpu::copy_device_to_host_async<scalar_t>(hb, d_b, nb_elems, C.comp_stream);
      else {
        gpu::synchronize_default_stream();
        gpu::copy_device_to_host<scalar_t>(b, d_b);
      }
    };

#if defined(STRUMPACK_USE_CUDA)
    if (gcap) {
      const void* key[5] = { dp ? static_cast<const void*>(dp) : nullptr, dp ? dp->d_iperm : nullptr,
                             dp ? dp->d_perm : nullptr, hb, d_braw };
      key[0] = dp ? reinterpret_cast<const void*>(1) : nullptr;
      if (C.graph.valid() && std::memcmp(key, C.gkey, sizeof(key)) != 0) C.graph.reset();
      if (C.graph.valid()) graph_launch(C.graph, C.comp_stream);
      else {
        bool captured = graph_begin(C.comp_stream);
        if (captured) {
          issue_device(true);
          captured = graph_end(C.comp_stream, C.graph, "solve");
        }
        if (captured) {
          std::memcpy(C.gkey, key, sizeof(key));
          graph_launch(C.graph, C.comp_stream);
        } else
          issue_device(false);
      }
    } else
#endif
      issue_device(false);
    if (dp || rhs_pin) {
      C.comp_stream.synchronize();
      if (b.ld() == b.rows()) std::copy(hb, hb + nb_elems, b.data());
      else for (int j=0; j<nrhs; j++) std::copy(hb + std::size_t(j) * b.rows(), hb + std::size_t(j+1) * b.rows(), b.ptr(0, j));
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
