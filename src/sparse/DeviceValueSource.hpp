/*
 * F-DEVUPDATE (magma-hpc F track, 2026-10-27): a sparse matrix can carry
 * a provider of its values on the GPU.
 *
 * When the value-only update (update_matrix_values with an unchanged
 * sparsity) ran on the device (see fronts/DeviceValueUpdate.hpp), the
 * host val() array may lag behind: it is refreshed by an asynchronous
 * write-back that completes while the factorization runs. Consumers
 * that work on the device (FrontMAGMA's factorization with the F-GATHER
 * structure cache) take the values from here instead of copying val().
 */
#ifndef STRUMPACK_DEVICE_VALUE_SOURCE_HPP
#define STRUMPACK_DEVICE_VALUE_SOURCE_HPP

#include <cstddef>

namespace strumpack {

  template<typename scalar_t> class DeviceValueSource {
  public:
    virtual ~DeviceValueSource() {}
    /**
     * Device pointer to the nnz current values (same order as the
     * matrix' val()), or nullptr when the device copy is not current
     * (the caller then copies val() itself). Blocks the host until the
     * values are ready, so they may be read on any stream afterwards,
     * and verifies the upload. *generation (optional) counts the
     * device-side updates.
     */
    virtual const scalar_t*
    device_values(std::size_t nnz, unsigned long long* generation=nullptr) = 0;
    /**
     * Called by the consumer once all its device work reading the
     * values has been issued (in legacy-default-stream order): the
     * provider may now refresh the host values (asynchronously).
     */
    virtual void factor_enqueued() = 0;
    /** Block until the matrix' host val() array is current. */
    virtual void sync_host_values() = 0;
    /**
     * F-GRAPH: like device_values(), but instead of blocking the host the
     * given CUDA stream (cudaStream_t passed as void*) is made to wait for
     * the values; the upload verification is deferred to
     * verify_pending_values(). Default: the blocking variant.
     */
    virtual const scalar_t*
    device_values_on_stream(std::size_t nnz, unsigned long long* generation,
                            void* cuda_stream) {
      (void)cuda_stream;
      return device_values(nnz, generation);
    }
    /**
     * Completes a deferred verification (blocks until the upload is done).
     * Returns false if the device values had to be re-uploaded, i.e. work
     * issued after device_values_on_stream() read wrong values and must be
     * re-issued.
     */
    virtual bool verify_pending_values() { return true; }
  };

} // end namespace strumpack

#endif // STRUMPACK_DEVICE_VALUE_SOURCE_HPP
