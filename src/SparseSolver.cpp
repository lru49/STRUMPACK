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
 */

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "StrumpackSparseSolver.hpp"

#if defined(STRUMPACK_USE_PAPI)
#include <papi.h>
#endif

#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/ordering/MatrixReordering.hpp"
#include "sparse/EliminationTree.hpp"
#include "iterative/IterativeSolvers.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  SparseSolver<scalar_t,integer_t>::SparseSolver
  (bool verbose, bool root) : SparseSolverBase<scalar_t,integer_t>
    (verbose, root) { }

  template<typename scalar_t,typename integer_t>
  SparseSolver<scalar_t,integer_t>::SparseSolver
  (int argc, char* argv[], bool verbose, bool root)
    : SparseSolverBase<scalar_t,integer_t>
    (argc, argv, verbose, root) { }

  template<typename scalar_t,typename integer_t>
  SparseSolver<scalar_t,integer_t>::~SparseSolver() = default;


  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::setup_tree() {
    tree_.reset(new EliminationTree<scalar_t,integer_t>
                (opts_, *mat_, nd_->tree()));
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::setup_reordering() {
    nd_.reset(new MatrixReordering<scalar_t,integer_t>(matrix()->size()));
  }

  template<typename scalar_t,typename integer_t> int
  SparseSolver<scalar_t,integer_t>::compute_reordering
  (const int* p, int base, int nx, int ny, int nz,
   int components, int width) {
    if (p) return nd_->set_permutation(opts_, *mat_, p, base);
    return nd_->nested_dissection
      (opts_, *mat_, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::separator_reordering() {
    nd_->separator_reordering(opts_, *mat_, tree_->root());
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::set_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    mat_.reset(new CSRMatrix<scalar_t,integer_t>(A));
    factored_ = reordered_ = false;
  }

  template <typename scalar_t, typename integer_t>
  void SparseSolver<scalar_t, integer_t>::set_lower_triangle_matrix
  (const CSRMatrix<scalar_t, integer_t> &A) {
    auto ptr = A.ptr();
    auto index = A.ind();
    auto value = A.val();
    std::vector<integer_t> mat_ptr = {0};
    std::vector<integer_t> mat_ind;
    std::vector<scalar_t> mat_val;
    for (int row = 0; row < A.size(); ++row) {
      mat_ptr.push_back(integer_t(0));
      for (int j = ptr[row]; j < ptr[row + 1]; ++j) {
        if (index[j] <= row) {
          mat_ind.push_back(index[j]);
          mat_val.push_back(value[j]);
          mat_ptr[row + 1]++;
        }
      }
      mat_ptr[row + 1] += mat_ptr[row];
    }
    mat_.reset(new CSRMatrix<scalar_t, integer_t>(
        integer_t(mat_ptr.size() - 1), mat_ptr.data(), mat_ind.data(),
        mat_val.data()));
    factored_ = reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::update_matrix_values
  (const CSRMatrix<scalar_t,integer_t>& A) {
    if (!(mat_ && A.size() == mat_->size() && A.nnz() <= mat_->nnz())) {
      // matrix() has been made symmetric, can have more nonzeros
      this->print_wrong_sparsity_error();
      return;
    }
    if (update_values_cached(A.size(), A.ptr(), A.ind(), A.val())) return;
    mat_.reset(new CSRMatrix<scalar_t,integer_t>(A));
    permute_matrix_values();
    build_update_cache(A.size(), A.ptr(), A.ind(), A.symm_sparse());
  }

  template<typename scalar_t,typename integer_t> bool
  SparseSolver<scalar_t,integer_t>::update_values_cached
  (integer_t N, const integer_t* ptr, const integer_t* ind, const scalar_t* val) {
    if (!upd_valid_ || !reordered_ || N != upd_N_ ||
        ptr[N] != upd_ptr_[N]) return false;
    if (!std::equal(upd_ptr_.begin(), upd_ptr_.end(), ptr) ||
        !std::equal(upd_ind_.begin(), upd_ind_.end(), ind)) {
      upd_valid_ = false;   // structure changed: rebuild through the full path
      return false;
    }
    const std::size_t nnz = mat_->nnz();
    auto v = mat_->val();
    const auto map = upd_map_.data();
#pragma omp parallel for
    for (std::size_t k=0; k<nnz; k++)
      v[k] = map[k] > 0 ? val[map[k]-1] : scalar_t(0.);
    factored_ = false;
    return true;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::build_update_cache
  (integer_t N, const integer_t* ptr, const integer_t* ind, bool symm_sparsity) {
    upd_valid_ = false;
    static const bool enabled = !(std::getenv("STRUM_UPDATE_CACHE") &&
                                  std::atoi(std::getenv("STRUM_UPDATE_CACHE")) == 0);
    if (!enabled || !reordered_ || matching_.job != MatchingJob::NONE ||
        equil_.type != EquilibrationType::NONE ||
        opts_.compression() != CompressionType::NONE) return;
    const std::size_t nnz_in = ptr[N];
    if (nnz_in >= (std::size_t(1) << 52)) return;
    std::vector<double> idx(nnz_in);
#pragma omp parallel for
    for (std::size_t k=0; k<nnz_in; k++) idx[k] = double(k+1);
    CSRMatrix<double,integer_t> Mi(N, ptr, ind, idx.data(), symm_sparsity);
    Mi.symmetrize_sparsity();
    Mi.permute(reordering()->iperm(), reordering()->perm());
    const std::size_t nnz = mat_->nnz();
    if (Mi.nnz() != integer_t(nnz) ||
        !std::equal(Mi.ptr(), Mi.ptr()+N+1, mat_->ptr()) ||
        !std::equal(Mi.ind(), Mi.ind()+nnz, mat_->ind())) return;
    upd_map_.resize(nnz);
    const double* mv = Mi.val();
#pragma omp parallel for
    for (std::size_t k=0; k<nnz; k++) upd_map_[k] = integer_t(std::llround(mv[k]));
    // self-check: the map applied to the input values must reproduce the permuted values bitwise
    // (checked on the matrix we just permuted: its values are the input values moved around)
    upd_ptr_.assign(ptr, ptr+N+1);
    upd_ind_.assign(ind, ind+nnz_in);
    upd_N_ = N;
    upd_valid_ = true;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    mat_.reset(new CSRMatrix<scalar_t,integer_t>
               (N, row_ptr, col_ind, values, symmetric_pattern));
    factored_ = reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::update_matrix_values
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    if (!(mat_ && N == mat_->size() && row_ptr[N] <= mat_->nnz())) {
      // matrix() has been made symmetric, can have more nonzeros
      this->print_wrong_sparsity_error();
      return;
    }
    if (update_values_cached(N, row_ptr, col_ind, values)) return;
    mat_.reset(new CSRMatrix<scalar_t,integer_t>
               (N, row_ptr, col_ind, values, symmetric_pattern));
    permute_matrix_values();
    build_update_cache(N, row_ptr, col_ind, symmetric_pattern);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::permute_matrix_values() {
    if (reordered_) {
      matrix()->apply_matching(matching_);
      matrix()->equilibrate(equil_);
      matrix()->symmetrize_sparsity();
      matrix()->permute(reordering()->iperm(), reordering()->perm());
      if (opts_.compression() != CompressionType::NONE)
        separator_reordering();
    }
    factored_ = false;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolver<scalar_t,integer_t>::solve_internal
  (const scalar_t* b, scalar_t* x, bool use_initial_guess) {
    auto N = matrix()->size();
    auto B = ConstDenseMatrixWrapperPtr(N, 1, b, N);
    DenseMW_t X(N, 1, x, N);
    return this->solve(*B, X, use_initial_guess);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::transform_x0
  (DenseM_t& x, DenseM_t& xtmp) {
    integer_t N = matrix()->size(), d = x.cols();
    auto& P = reordering()->iperm();
    if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          x(i, j) = x(i, j) / matching_.C[i];
    if (opts_.matching() == MatchingJob::NONE)
      xtmp.copy(x);
    else
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          xtmp(i, j) = x(matching_.Q[i], j);
    if (this->equil_.type == EquilibrationType::COLUMN ||
        this->equil_.type == EquilibrationType::BOTH)
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          xtmp(i, j) = xtmp(i, j) / equil_.C[i];
    for (integer_t j=0; j<d; j++)
#pragma omp parallel for
      for (integer_t i=0; i<N; i++)
        x(i, j) = xtmp(P[i], j);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::transform_x
  (DenseM_t& x, DenseM_t& xtmp) {
    integer_t N = matrix()->size(), d = x.cols();
    auto& Pi = reordering()->perm();
    for (integer_t j=0; j<d; j++)
#pragma omp parallel for
      for (integer_t i=0; i<N; i++)
        xtmp(i, j) = x(Pi[i], j);
    if (this->equil_.type == EquilibrationType::COLUMN ||
        this->equil_.type == EquilibrationType::BOTH)
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          xtmp(i, j) = equil_.C[i] * xtmp(i, j);
    if (opts_.matching() == MatchingJob::NONE)
      x.copy(xtmp);
    else {
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          x(matching_.Q[i], j) = xtmp(i, j);
      if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
        for (integer_t j=0; j<d; j++)
#pragma omp parallel for
          for (integer_t i=0; i<N; i++)
            x(i, j) = matching_.C[i] * x(i, j);
    }
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::transform_b
  (const DenseM_t& b, DenseM_t& bloc) {
    using real_t = typename RealType<scalar_t>::value_type;
    integer_t N = matrix()->size(), d = b.cols();
    auto& P = reordering()->iperm();
    std::vector<real_t> R(N, 1.);
    if (equil_.type == EquilibrationType::ROW ||
        equil_.type == EquilibrationType::BOTH)
      for (integer_t i=0; i<N; i++)
        R[i] *= equil_.R[i];
    if (this->reordered_ &&
        opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
      for (integer_t i=0; i<N; i++)
        R[i] *= matching_.R[i];
    for (integer_t j=0; j<d; j++)
#pragma omp parallel for
      for (integer_t i=0; i<N; i++) {
        auto p = P[i];
        bloc(i, j) = R[p] * b(p, j);
      }
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolver<scalar_t,integer_t>::solve_internal
  (const DenseM_t& b, DenseM_t& x, bool use_initial_guess) {
    // reordering has to be called, even for the iterative solvers
    if (!this->reordered_) {
      ReturnCode ierr = this->reorder();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    // factor needs to be called, except for the non-preconditioned
    // solvers
    if (!this->factored_ &&
        (opts_.Krylov_solver() != KrylovSolver::GMRES) &&
        (opts_.Krylov_solver() != KrylovSolver::BICGSTAB)) {
      ReturnCode ierr = this->factor();
      // TODO there could be zero pivots, but replaced, and this
      // should still continue!!
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }

    TaskTimer t("solve");
    this->perf_counters_start();
    t.start();
    assert(b.cols() == x.cols());

    integer_t d = b.cols();
    assert(matrix()->size() < std::numeric_limits<int>::max());

    // S-HOST (magma-hpc, 2026-10-27): for the direct solve without matching and equilibration, transform_b
    // (bloc(i) = b(iperm[i])) and transform_x (x(i) = y(perm[i])) are pure permutations of the rhs / solution.
    // Let the GPU front do them on the device (gather kernels around the tree solve, device copy of the
    // permutation cached across solves) and skip the bloc allocation, the two host passes and the x = bloc copy.
    // Falls through to the host path when the tree cannot (CPU fronts, factors not on the device).
    // STRUM_SOLVE_DEVPERM=0 restores the host path.
    static const bool devperm = !(std::getenv("STRUM_SOLVE_DEVPERM") &&
                                  std::atoi(std::getenv("STRUM_SOLVE_DEVPERM")) == 0);
    if (devperm && opts_.Krylov_solver() == KrylovSolver::DIRECT &&
        opts_.matching() == MatchingJob::NONE &&
        equil_.type == EquilibrationType::NONE &&
        tree()->multifrontal_solve_devperm
        (b, x, reordering()->iperm(), reordering()->perm())) {
      Krylov_its_ = 0;
      t.stop();
      this->perf_counters_stop("DIRECT/GMRES solve");
      this->print_solve_stats(t);
      return ReturnCode::SUCCESS;
    }

    DenseM_t bloc(b.rows(), d);

    auto spmv = [&](const scalar_t* x, scalar_t* y)
                { matrix()->spmv(x, y); };
    Krylov_its_ = 0;

    if (use_initial_guess &&
        opts_.Krylov_solver() != KrylovSolver::DIRECT)
      transform_x0(x, bloc);
    transform_b(b, bloc);

    auto MFsolve =
      [&](scalar_t* w) {
        DenseMW_t X(x.rows(), 1, w, x.ld());
#if !defined(STRUMPACK_USE_MAGMA)
        if (opts_.use_gpu())
          std::cerr
            << "-------------------------------------------------------" << std::endl
            << "WARNING: sparse multifrontal solve is done on CPU," << std::endl
            << "GPU solve requires MAGMA, see" << std::endl
            << "   https://bitbucket.org/icl/magma/src/master/" << std::endl
            << "Configure with -DTPL_ENABLE_MAGMA=ON" << std::endl
            << "and set the MAGMA_DIR environment variable" << std::endl
            << "-------------------------------------------------------" << std::endl;
#endif
        tree()->multifrontal_solve(X);
      };

    // DIRECT apart, every branch below runs an iterative solver that stops
    // either because it converged or because it ran out of iterations
    bool iterative_solve = true;

    switch (opts_.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if (opts_.compression() != CompressionType::NONE && x.cols() == 1)
        iterative::GMRes<scalar_t>
          (spmv, MFsolve, x.rows(), x.data(), bloc.data(),
           opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
           opts_.gmres_restart(), opts_.GramSchmidt_type(),
           use_initial_guess, opts_.verbose() && is_root_);
      else
        iterative::IterativeRefinement<scalar_t,integer_t>
          (*matrix(), [&](DenseM_t& w) { tree()->multifrontal_solve(w); },
           x, bloc, opts_.rel_tol(), opts_.abs_tol(),
           Krylov_its_, opts_.maxit(), use_initial_guess,
           opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::DIRECT: {
      x = bloc;
      tree()->multifrontal_solve(x);
      iterative_solve = false;
    }; break;
    case KrylovSolver::REFINE: {
      iterative::IterativeRefinement<scalar_t,integer_t>
        (*matrix(), [&](DenseM_t& w) { tree()->multifrontal_solve(w); },
         x, bloc, opts_.rel_tol(), opts_.abs_tol(),
         Krylov_its_, opts_.maxit(), use_initial_guess,
         opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::PREC_GMRES: {
      assert(x.cols() == 1);
      iterative::GMRes<scalar_t>
        (spmv, MFsolve, x.rows(), x.data(), bloc.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         opts_.gmres_restart(), opts_.GramSchmidt_type(),
         use_initial_guess, opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::PREC_BICGSTAB: {
      assert(x.cols() == 1);
      iterative::BiCGStab<scalar_t>
        (spmv, MFsolve, x.rows(), x.data(), bloc.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         use_initial_guess, opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::GMRES: { // see above
      assert(x.cols() == 1);
      iterative::GMRes<scalar_t>
        (spmv, [](scalar_t* x) {}, x.rows(), x.data(), bloc.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         opts_.gmres_restart(), opts_.GramSchmidt_type(),
         use_initial_guess, opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::BICGSTAB: {
      assert(x.cols() == 1);
      iterative::BiCGStab<scalar_t>
        (spmv, [](scalar_t* x) {}, x.rows(), x.data(), bloc.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         use_initial_guess, opts_.verbose() && is_root_);
    }
    }
    ReturnCode rc = ReturnCode::SUCCESS;
    if (iterative_solve && Krylov_its_ >= opts_.maxit()) {
      // The solver used its whole iteration budget, so it may or may not have
      // reached the tolerance. Only in that case is it worth one extra
      // matrix-vector product to find out which, and to say so to the caller.
      using real_t = typename RealType<scalar_t>::value_type;
      DenseM_t r(x.rows(), x.cols());
      matrix()->spmv(x, r);
      r.scale_and_add(scalar_t(-1.), bloc);
      auto rnrm = r.norm();
      auto bnrm = bloc.norm();
      if (rnrm > opts_.abs_tol() &&
          (bnrm == real_t(0.) || rnrm / bnrm > opts_.rel_tol())) {
        if (opts_.verbose() && is_root_)
          std::cout << "STRUMPACK: iterative solver did not converge in "
                    << Krylov_its_ << " iterations, ||Ax-b|| = " << rnrm
                    << ", ||b|| = " << bnrm << std::endl;
        rc = ReturnCode::NO_CONVERGENCE;
      }
    }

    transform_x(x, bloc);

    t.stop();
    this->perf_counters_stop("DIRECT/GMRES solve");
    this->print_solve_stats(t);
    return rc;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::delete_factors_internal() {
    tree_.reset(nullptr);
  }

  // explicit template instantiations
  template class SparseSolver<float,int>;
  template class SparseSolver<double,int>;
  template class SparseSolver<std::complex<float>,int>;
  template class SparseSolver<std::complex<double>,int>;

  template class SparseSolver<float,long int>;
  template class SparseSolver<double,long int>;
  template class SparseSolver<std::complex<float>,long int>;
  template class SparseSolver<std::complex<double>,long int>;

  template class SparseSolver<float,long long int>;
  template class SparseSolver<double,long long int>;
  template class SparseSolver<std::complex<float>,long long int>;
  template class SparseSolver<std::complex<double>,long long int>;

} //end namespace strumpack
