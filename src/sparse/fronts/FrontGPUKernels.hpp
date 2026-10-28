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
#ifndef FRONTAL_MATRIX_GPU_KERNELS_HPP
#define FRONTAL_MATRIX_GPU_KERNELS_HPP

#include "FrontGPUStructs.hpp"
#include "misc/Triplet.hpp"
#include "dense/GPUWrapper.hpp"

namespace strumpack {
  namespace gpu {

    template<typename T> void
    gather_front_elements(std::size_t n, const int* src, const unsigned* dst,
                          const T* vals, T* base);

    template<typename T> void
    assemble(unsigned int, AssembleData<T>*, AssembleData<T>*);

    template<typename T> void
    assemble_symmetric(unsigned int, AssembleData<T>*, AssembleData<T>*);

    template<typename T, int NT=32,
             typename real_t = typename RealType<T>::value_type>
    void factor_block_batch(unsigned int, FrontData<T>*, bool, real_t, int*);

    template<typename T, int NT=32,
             typename real_t = typename RealType<T>::value_type>
    void factor_symmetric_block_batch(unsigned int count, FrontData<T>* dat,
                                      bool replace, real_t thresh, int* dinfo);

    template<typename T,
             typename real_t = typename RealType<T>::value_type>
    void replace_pivots(int, T*, real_t, gpu::Stream* = nullptr);

    template<typename T,
             typename real_t = typename RealType<T>::value_type>
    void replace_pivots_vbatched(Handle& handle, int* dn, int max_n,
                                 T** dA, int* lddA, real_t thresh,
                                 unsigned int batchCount);

    template<typename T> void
    extend_add_rhs(int, int, unsigned int, AssembleData<T>*, AssembleData<T>*);

    template<typename T> void
    extract_rhs(int, int, unsigned int, AssembleData<T>*, AssembleData<T>*);

    // F-GRAPH (magma-hpc F track): the same kernels issued on an explicit
    // stream instead of the legacy default stream, so that a whole
    // factorization level / solve can be captured in a CUDA graph.
    template<typename T> void
    gather_front_elements(std::size_t n, const int* src, const unsigned* dst,
                          const T* vals, T* base, Stream& s);
    template<typename T> void
    assemble(unsigned int, AssembleData<T>*, AssembleData<T>*, Stream& s);
    template<typename T> void
    extend_add_rhs(int, int, unsigned int, AssembleData<T>*, AssembleData<T>*,
                   Stream& s);
    template<typename T> void
    extract_rhs(int, int, unsigned int, AssembleData<T>*, AssembleData<T>*,
                Stream& s);

  } // end namespace gpu
} // end namespace strumpack

#endif // FRONTAL_MATRIX_GPU_KERNELS_HPP
