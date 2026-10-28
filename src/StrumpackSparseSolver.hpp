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
/**
 * \file SparseSolver.hpp
 * \brief Contains the definition of the sequential/multithreaded
 * sparse solver class.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_HPP
#define STRUMPACK_SPARSE_SOLVER_HPP

#include <algorithm>
#include <new>
#include <memory>
#include <vector>
#include <string>

#include "SparseSolverBase.hpp"

namespace strumpack { namespace gpu {
  template<typename scalar_t, typename integer_t> class DeviceValueUpdater;   // F-DEVUPDATE
} }

/**
 * All of STRUMPACK is contained in the strumpack namespace.
 */
struct CUstream_st;   // S2: cudaStream_t is CUstream_st* (no CUDA header needed here)

namespace strumpack {

  // forward declarations
  template<typename scalar_t,typename integer_t> class MatrixReordering;
  template<typename scalar_t,typename integer_t> class EliminationTree;
  class TaskTimer;

  /**
   * \class SparseSolver
   *
   * \brief SparseSolver is the main sequential or
   * multithreaded sparse solver class.
   *
   * This is the main interface to STRUMPACK's sparse solver. Use this
   * for a sequential or multithreaded sparse solver. For the fully
   * distributed solver, see SparseSolverMPIDist.
   *
   * \tparam scalar_t can be: float, double, std::complex<float> or
   * std::complex<double>.
   *
   * \tparam integer_t defaults to a regular int. If regular int
   * causes 32 bit integer overflows, you should switch to
   * integer_t=int64_t instead. This should be a __signed__ integer
   * type.
   *
   * \see SparseSolverMPIDist
   */
  template<typename scalar_t,typename integer_t=int>
  class SparseSolver :
    public SparseSolverBase<scalar_t,integer_t> {

    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Tree_t = EliminationTree<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:

    /**
     * Constructor of the SparseSolver class, taking command line
     * arguments.
     *
     * \param argc number of arguments, i.e, number of elements in
     * the argv array
     * \param argv command line arguments. Add -h or --help to have a
     * description printed
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process, only the root will print certain messages to cout
     */
    SparseSolver(int argc, char* argv[], bool verbose=true, bool root=true);

    /**
     * Constructor of the SparseSolver class.
     *
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process. Only the root will print certain messages
     * \see set_from_options
     */
    SparseSolver(bool verbose=true, bool root=true);

    /**
     * (Virtual) destructor of the SparseSolver class.
     */
    ~SparseSolver();

    /**
     * Associate a (sequential) CSRMatrix with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function.
     *
     * \param A A CSRMatrix<scalar_t,integer_t> object, will
     * internally be duplicated
     *
     * \see set_csr_matrix
     */
    void set_matrix(const CSRMatrix<scalar_t,integer_t>& A);

    /**
     * Associate a (sequential) NxN CSR matrix with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function. See the manual for a description of the
     * CSR format. You can also use the CSRMatrix class.
     *
     * \param N number of rows and columns of the CSR input matrix.
     * \param row_ptr indices in col_ind and values for the start of
     * each row. Nonzeros for row r are in [row_ptr[r],row_ptr[r+1])
     * \param col_ind column indices of each nonzero
     * \param values nonzero values
     * \param symmetric_pattern denotes whether the sparsity
     * __pattern__ of the input matrix is symmetric, does not require
     * the matrix __values__ to be symmetric
     *
     * \see set_matrix
     */
    void set_csr_matrix(integer_t N,
                        const integer_t* row_ptr, const integer_t* col_ind,
                        const scalar_t* values, bool symmetric_pattern=false);

    /**
     * Associate the lower triangle from a (sequential) NxN CSR matrix
     * with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function. See the manual for a description of the
     * CSR format. You can also use the CSRMatrix class.
     *
     * \param A A CSRMatrix<scalar_t,integer_t> object, will
     * internally be duplicated
     *
     * \see set_matrix
     */
    void set_lower_triangle_matrix(const CSRMatrix<scalar_t,integer_t>& A);

    /**
     * This can only be used to UPDATE the nonzero values of the
     * matrix. So it should be called with exactly the same sparsity
     * pattern (row_ptr and col_ind) as used to set the initial matrix
     * (using set_matrix or set_csr_matrix). This routine can be
     * called after having performed a factorization of a different
     * matrix with the same sparsity pattern. In that case, when this
     * solver is used for another solve, with the updated matrix
     * values, the permutation vector previously computed will be
     * reused to permute the updated matrix values, instead of
     * recomputing the permutation. The numerical factorization will
     * automatically be redone.
     *
     * \param N Number of rows in the matrix.
     * \param row_ptr Row pointer array in the typical compressed
     * sparse row representation. This should be the same as used in
     * an earlier call to set_csr_matrix.
     * \param col_ind Column index array in the typical compressed
     * sparse row representation. This should be the same as used in
     * an earlier call to set_csr_matrix.
     * \param values Array with numerical nonzero values for the
     * matrix, corresponding to the row_ptr and col_ind compressed
     * sparse row representation.
     * \param symmetric_pattern Denotes whether the sparsity
     * __pattern__ of the input matrix is symmetric, does not require
     * the matrix __values__ to be symmetric
     *
     * \see set_csr_matrix, set_matrix
     */
    void update_matrix_values(integer_t N,
                              const integer_t* row_ptr,
                              const integer_t* col_ind,
                              const scalar_t* values,
                              bool symmetric_pattern=false);

    /**
     * This can only be used to UPDATE the nonzero values of the
     * matrix. So it should be called with exactly the same sparsity
     * pattern as used to set the initial matrix (using set_matrix or
     * set_csr_matrix). This routine can be called after having
     * performed a factorization of a different matrix with the same
     * sparsity pattern. In that case, when this solver is used for
     * another solve, with the updated matrix values, the permutation
     * vector previously computed will be reused to permute the
     * updated matrix values, instead of recomputing the
     * permutation. The numerical factorization will automatically be
     * redone.
     *
     * \param A Sparse matrix, should have the same sparsity pattern
     * as the matrix associated with this solver earlier.
     *
     * \see set_csr_matrix, set_matrix
     */
    void update_matrix_values(const CSRMatrix<scalar_t,integer_t>& A);

    /**
     * F-FP32-UPDATE (magma-hpc F track): value update from a matrix of another scalar type (the mixed-precision
     * solver's double input into the float inner solver). With the sparsity unchanged the cast is folded into the
     * cached gather (no intermediate cast_matrix copy); otherwise cast + the regular update path.
     */
    /**
     * S2 (magma-hpc F track, 2026-10-28): solve with DEVICE pointers. d_b (input, N entries) and d_x (output,
     * N entries) live in device memory; the whole solve, including the sparse reordering of b and of x, is
     * ordered on `stream` (a cudaStream_t; nullptr = the legacy default stream) and returns without any host
     * synchronisation or host staging. Independent solvers called with independent streams run concurrently.
     * Only the direct GPU solve without matching and equilibration (KrylovSolver::DIRECT, MatchingJob::NONE,
     * no equilibration, factors on the device) supports it: anything else returns ReturnCode::UNSUPPORTED and
     * does nothing (use solve()). The factorization is computed first when it is not up to date (that part
     * synchronises as factor() does).
     */
    ReturnCode solve_device(const scalar_t* d_b, scalar_t* d_x,
                            CUstream_st* stream = nullptr);

    template<typename src_t>
    void update_matrix_values_from(const CSRMatrix<src_t,integer_t>& A) {
      if (!(mat_ && A.size() == mat_->size() && A.nnz() <= mat_->nnz())) {
        this->print_wrong_sparsity_error();
        return;
      }
      if (update_values_cached_from<src_t>(A.size(), A.ptr(), A.ind(), A.val())) return;
      update_matrix_values(cast_matrix<src_t,integer_t,scalar_t>(A));
    }

  private:
    void setup_tree() override;
    void setup_reordering() override;
    int compute_reordering(const int* p, int base,
                           int nx, int ny, int nz,
                           int components, int width) override;
    void separator_reordering() override;

    SpMat_t* matrix() override { return mat_.get(); }
    Reord_t* reordering() override { return nd_.get(); }
    Tree_t* tree() override { return tree_.get(); }
    const SpMat_t* matrix() const override { return mat_.get(); }
    const Reord_t* reordering() const override { return nd_.get(); }
    const Tree_t* tree() const override { return tree_.get(); }

    void permute_matrix_values();
    // F-UPDATE (magma-hpc F track): value-only matrix update for a refactorization with the same sparsity.
    // After the first update_matrix_values with matching NONE and no equilibration, the net effect of
    // symmetrize_sparsity + permute on the values is a fixed gather (new permuted val[k] = input val[map[k]-1],
    // 0 for the explicit zeros symmetrization adds). The map is computed once with the same code run on an
    // index-valued matrix and reused while the input structure (N, nnz, ptr, ind) is unchanged.
    // STRUM_UPDATE_CACHE=0 disables it.
    bool update_values_cached(integer_t N, const integer_t* ptr, const integer_t* ind, const scalar_t* val);
    // same, with the cast src_t -> scalar_t folded into the gather (host) or done in the device gather kernel
    template<typename src_t>
    bool update_values_cached_from(integer_t N, const integer_t* ptr, const integer_t* ind, const src_t* val);
    // the cached map applies: same N / nnz / ptr as when it was built (ind too with STRUM_UPDATE_CHECK=1)
    bool update_cache_structure_ok(integer_t N, const integer_t* ptr, const integer_t* ind);
    // F-DEVUPDATE: the device-side update (sparse/fronts/DeviceValueUpdate.hpp); the matrix' host values are
    // then refreshed asynchronously and the factorization takes the values from the device
    void host_values_updated();   // host path wrote mat_->val(): the device copy is stale
    void device_update_invalidate();   // before mat_ is replaced
    void sync_host_matrix_values() override;   // lazy write-back before a host read of the values
#if defined(STRUMPACK_USE_CUDA)
    std::unique_ptr<gpu::DeviceValueUpdater<scalar_t,integer_t>> devupd_;
#endif

    void build_update_cache(integer_t N, const integer_t* ptr, const integer_t* ind, bool symm_sparsity);
    std::vector<integer_t> upd_map_, upd_ptr_, upd_ind_;
    integer_t upd_N_ = 0;
    bool upd_valid_ = false;

    ReturnCode solve_internal(const scalar_t* b, scalar_t* x,
                              bool use_initial_guess=false) override;
    ReturnCode solve_internal(const DenseM_t& b, DenseM_t& x,
                              bool use_initial_guess=false) override;

    void delete_factors_internal() override;

    void transform_x0(DenseM_t& x, DenseM_t& xtmp);
    void transform_b(const DenseM_t& b, DenseM_t& bloc);
    void transform_x(DenseM_t& x, DenseM_t& xtmp);

    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> mat_;
    std::unique_ptr<MatrixReordering<scalar_t,integer_t>> nd_;
    std::unique_ptr<EliminationTree<scalar_t,integer_t>> tree_;

    using SPBase_t = SparseSolverBase<scalar_t,integer_t>;
    using SPBase_t::opts_;
    using SPBase_t::is_root_;
    using SPBase_t::matching_;
    using SPBase_t::equil_;
    using SPBase_t::factored_;
    using SPBase_t::reordered_;
    using SPBase_t::Krylov_its_;
    using SPBase_t::solve_internal;
  };

  template<typename scalar_t,typename integer_t>
  using StrumpackSparseSolver = SparseSolver<scalar_t,integer_t>;

} //end namespace strumpack

#endif // STRUMPACK_SPARSE_SOLVER_HPP
