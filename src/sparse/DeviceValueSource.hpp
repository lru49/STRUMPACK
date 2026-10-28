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
     * Called by the consumer when it is done issuing its work on the
     * values (waits for the write-back of the host values).
     */
    virtual void release_device_values() = 0;
  };

} // end namespace strumpack

#endif // STRUMPACK_DEVICE_VALUE_SOURCE_HPP
