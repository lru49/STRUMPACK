/*
 * F-DEVUPDATE (magma-hpc F track, 2026-10-27): device-side value-only
 * matrix update for refactorizations with an unchanged sparsity.
 *
 * F-UPDATE applied the cached symmetrize+permute gather map on the host
 * (75 ms of host-only work for the JOREK proxy, GPU idle) and the
 * factorization then copied the 455 MB of permuted values to the device
 * (63 ms, no kernel overlap). Now the caller's RAW value array is copied
 * (asynchronously, in STRUM_DEVUPDATE_CHUNKS chunks, from page-locked
 * memory when the array can be registered) to a persistent device buffer,
 * the gather map (uploaded once per structure) is applied by one kernel
 * on the device, and the result stays resident as the factorization's
 * CSR value buffer. update_matrix_values returns as soon as the work is
 * enqueued; the factorization waits for it (and verifies a sample of the
 * uploaded values against the host array, which catches a stale host
 * registration: the F-VALCOPY v1 failure mode). The host values of the
 * matrix are refreshed by an asynchronous device-to-host write-back that
 * overlaps the factorization kernels, so every host consumer (iterative
 * refinement, spmv) still sees the new values.
 *
 * Pure data movement: bitwise identical to the host path.
 *
 * Knobs: STRUM_DEVUPDATE=0 restores the host update (default 1);
 * STRUM_DEVUPDATE_CHUNKS (8); STRUM_DEVUPDATE_PIN=0 never registers the
 * caller's array (staged copy through a pinned buffer, default 1);
 * STRUM_DEVUPDATE_WRITEBACK=0 leaves the host values stale (direct
 * solves only, default 1); STRUM_DEVUPDATE_VERBOSE=1 prints the times.
 *
 * Contract: the value array passed to update_matrix_values must stay
 * allocated and unmodified until the next factor() (the upload may still
 * be in flight when update_matrix_values returns).
 */
#ifndef STRUMPACK_DEVICE_VALUE_UPDATE_HPP
#define STRUMPACK_DEVICE_VALUE_UPDATE_HPP

#include <cstddef>
#include <memory>

#include "sparse/DeviceValueSource.hpp"

namespace strumpack {
  namespace gpu {

    template<typename scalar_t, typename integer_t>
    class DeviceValueUpdater : public DeviceValueSource<scalar_t> {
    public:
      DeviceValueUpdater();
      ~DeviceValueUpdater();

      /** STRUM_DEVUPDATE != 0 (default on). */
      static bool enabled();

      /**
       * (Re)build for a new structure: upload the gather map (nnz
       * entries, 1-based indices into the nnz_in raw values, 0 for the
       * explicit zeros added by the symmetrization) and remember the
       * matrix' own value array (nnz entries, write-back target; it is
       * page-locked here and must stay allocated until invalidate()).
       * Returns false when the device path cannot be used.
       */
      bool set_map(const integer_t* map, std::size_t nnz,
                   std::size_t nnz_in, scalar_t* host_vals);

      /**
       * Enqueue: raw (nnz_in values, caller's array) -> device, gather
       * (cast src_t -> scalar_t on the device for the mixed-precision
       * solver's double input). Returns at once. false = the device
       * path is not available (the caller must update on the host).
       */
      template<typename src_t>
      bool update_from(const src_t* raw, std::size_t nnz_in);
      bool update(const scalar_t* raw, std::size_t nnz_in);

      /** The host values were written by the host path: the device copy is stale. */
      void host_values_changed();

      /** Forget the map and every host registration (matrix replaced). */
      void invalidate();

      const scalar_t* device_values(std::size_t nnz,
                                    unsigned long long* generation) override;
      void factor_enqueued() override;
      void sync_host_values() override;

      struct Impl;
    private:
      std::unique_ptr<Impl> impl_;
    };

  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_DEVICE_VALUE_UPDATE_HPP
