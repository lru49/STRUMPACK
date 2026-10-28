// F-DEVUPDATE (magma-hpc F track, 2026-10-27): see DeviceValueUpdate.hpp
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <complex>

#include <cuda_runtime_api.h>

#include "DeviceValueUpdate.hpp"
#include "DeviceValueUpdateKernels.hpp"
#include "dense/CUDAWrapper.hpp"

namespace strumpack {
  namespace gpu {

    namespace {
      int env_int(const char* name, int def) {
        const char* e = std::getenv(name);
        return e ? std::atoi(e) : def;
      }
      // host registration of a caller-owned range, (re)validated before every use
      struct HostReg {
        const void* ptr = nullptr;
        std::size_t bytes = 0;
        void unregister() {
          if (ptr) {
            // the range may have been freed by its owner in the meantime: ignore the error
            cudaHostUnregister(const_cast<void*>(ptr));
            cudaGetLastError();
          }
          ptr = nullptr; bytes = 0;
        }
        // true if [p, p+bytes) is page-locked afterwards
        bool ensure(const void* p, std::size_t bytes_) {
          if (ptr && (ptr != p || bytes != bytes_)) unregister();
          cudaPointerAttributes a{};
          if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return false; }
          if (a.type == cudaMemoryTypeHost) {
            if (ptr == p) return true;              // our registration
            cudaPointerAttributes b{};              // page-locked by the caller: check the end of the range too
            const bool ok = cudaPointerGetAttributes(&b, static_cast<const char*>(p) + bytes_ - 1) == cudaSuccess &&
              b.type == cudaMemoryTypeHost;
            cudaGetLastError();
            return ok;
          }
          if (a.type != cudaMemoryTypeUnregistered) return false;   // device or managed memory
          if (cudaHostRegister(const_cast<void*>(p), bytes_, cudaHostRegisterDefault) != cudaSuccess) {
            cudaGetLastError();
            return false;
          }
          ptr = p; bytes = bytes_;
          return true;
        }
      };
    }

    template<typename scalar_t, typename integer_t>
    struct DeviceValueUpdater<scalar_t,integer_t>::Impl {
      static constexpr int NS = 4096;   // verification samples
      cudaStream_t stream = nullptr;    // non-blocking: nothing of the factorization waits on it implicitly
      cudaEvent_t e_order = nullptr, e_start = nullptr, e_copied = nullptr, e_gathered = nullptr,
        e_ready = nullptr, e_wb = nullptr;
      DeviceMemory<scalar_t> d_raw, d_out, d_samp;
      DeviceMemory<integer_t> d_map;
      HostMemory<scalar_t> h_stage, h_samp;
      std::size_t nnz = 0, nnz_in = 0;
      scalar_t* host_vals = nullptr;    // the matrix' own values (write-back target)
      HostReg reg_raw, reg_host;
      const scalar_t* raw_ptr = nullptr;
      unsigned long long gen = 0;
      bool map_ok = false, current = false, wb_pending = false, copied_once = false;
      bool pinned_last = false, staged_warned = false;
      std::chrono::steady_clock::time_point t_update;
      double host_ms = 0.;
      const int nchunks = std::max(1, env_int("STRUM_DEVUPDATE_CHUNKS", 8));
      const bool pin = env_int("STRUM_DEVUPDATE_PIN", 1) != 0;
      const bool writeback = env_int("STRUM_DEVUPDATE_WRITEBACK", 1) != 0;
      const bool verbose = env_int("STRUM_DEVUPDATE_VERBOSE", 0) != 0;

      Impl() {
        gpu_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        gpu_check(cudaEventCreateWithFlags(&e_order, cudaEventDisableTiming));
        gpu_check(cudaEventCreate(&e_start));
        gpu_check(cudaEventCreate(&e_copied));
        gpu_check(cudaEventCreate(&e_gathered));
        gpu_check(cudaEventCreateWithFlags(&e_ready, cudaEventDisableTiming));
        gpu_check(cudaEventCreateWithFlags(&e_wb, cudaEventDisableTiming));
        h_samp = HostMemory<scalar_t>(NS);
      }
      ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        reg_raw.unregister();
        reg_host.unregister();
        for (auto e : {e_order, e_start, e_copied, e_gathered, e_ready, e_wb}) if (e) cudaEventDestroy(e);
        if (stream) cudaStreamDestroy(stream);
        cudaGetLastError();
      }

      int nsamples() const { return int(std::min<std::size_t>(std::size_t(NS), nnz_in)); }

      // raw -> d_raw (chunked), gather, samples, write-back; all on `stream`
      void enqueue(const scalar_t* raw, bool try_pin) {
        const std::size_t bytes = nnz_in * sizeof(scalar_t);
        // order after everything queued on the legacy default stream (the previous factorization reads d_out
        // there) and, through stream order, after our own previous work
        gpu_check(cudaEventRecord(e_order, 0));
        gpu_check(cudaStreamWaitEvent(stream, e_order, 0));
        const bool pinned = try_pin && pin && reg_raw.ensure(raw, bytes);
        if (!pinned) {
          if (!staged_warned && verbose) {
            std::cout << "# F-DEVUPDATE: value array not page-locked, staged copy" << std::endl;
            staged_warned = true;
          }
          if (h_stage.size() < nnz_in) h_stage = HostMemory<scalar_t>(nnz_in);
          if (copied_once) gpu_check(cudaEventSynchronize(e_copied));   // previous use of the staging buffer
        }
        gpu_check(cudaEventRecord(e_start, stream));
        const std::size_t chunk = (nnz_in + nchunks - 1) / nchunks;
        for (int c=0; c<nchunks; c++) {
          const std::size_t lo = std::size_t(c)*chunk, hi = std::min(nnz_in, lo+chunk);
          if (lo >= hi) break;
          const scalar_t* src = raw + lo;
          if (!pinned) {
            scalar_t* stg = h_stage.data() + lo;
#pragma omp parallel for
            for (std::size_t k=0; k<hi-lo; k++) stg[k] = raw[lo+k];
            src = stg;
          }
          gpu_check(cudaMemcpyAsync(d_raw.template as<scalar_t>() + lo, src, (hi-lo)*sizeof(scalar_t),
                                    cudaMemcpyHostToDevice, stream));
        }
        gpu_check(cudaEventRecord(e_copied, stream));
        copied_once = true;
        devupd_gather<scalar_t,integer_t>(nnz, nnz_in, d_map, d_raw, d_out, stream);
        gpu_check(cudaEventRecord(e_gathered, stream));
        const int ns = nsamples();
        devupd_sample<scalar_t>(ns, nnz_in, d_raw, d_samp, stream);
        gpu_check(cudaMemcpyAsync(h_samp.data(), d_samp.template as<scalar_t>(), ns*sizeof(scalar_t),
                                  cudaMemcpyDeviceToHost, stream));
        gpu_check(cudaEventRecord(e_ready, stream));
        if (writeback) {
          gpu_check(cudaMemcpyAsync(host_vals, d_out.template as<scalar_t>(), nnz*sizeof(scalar_t),
                                    cudaMemcpyDeviceToHost, stream));
          wb_pending = true;
        }
        gpu_check(cudaEventRecord(e_wb, stream));
        raw_ptr = raw;
        pinned_last = pinned;
      }

      // the uploaded samples must match the host array (catches a stale registration / a modified array)
      bool verify() const {
        const int ns = nsamples();
        for (int i=0; i<ns; i++)
          if (std::memcmp(&h_samp.data()[i], raw_ptr + devupd_sample_index(i, ns, nnz_in), sizeof(scalar_t)))
            return false;
        return true;
      }
    };

    template<typename scalar_t, typename integer_t>
    DeviceValueUpdater<scalar_t,integer_t>::DeviceValueUpdater() : impl_(new Impl()) {}
    template<typename scalar_t, typename integer_t>
    DeviceValueUpdater<scalar_t,integer_t>::~DeviceValueUpdater() = default;

    template<typename scalar_t, typename integer_t> bool
    DeviceValueUpdater<scalar_t,integer_t>::enabled() {
      static const bool on = env_int("STRUM_DEVUPDATE", 1) != 0;
      return on;
    }

    template<typename scalar_t, typename integer_t> bool
    DeviceValueUpdater<scalar_t,integer_t>::set_map
    (const integer_t* map, std::size_t nnz, std::size_t nnz_in, scalar_t* host_vals) {
      auto& I = *impl_;
      invalidate();
      if (!nnz || !nnz_in || !host_vals) return false;
      if (I.writeback && !I.reg_host.ensure(host_vals, nnz*sizeof(scalar_t))) {
        if (I.verbose)
          std::cout << "# F-DEVUPDATE: cannot page-lock the matrix values for the write-back, host update" << std::endl;
        return false;
      }
      try {
        if (I.d_map.size() < nnz) I.d_map = DeviceMemory<integer_t>(nnz);
        if (I.d_out.size() < nnz) I.d_out = DeviceMemory<scalar_t>(nnz);
        if (I.d_raw.size() < nnz_in) I.d_raw = DeviceMemory<scalar_t>(nnz_in);
        if (!I.d_samp.size()) I.d_samp = DeviceMemory<scalar_t>(Impl::NS);
      } catch (const std::bad_alloc&) {
        I.d_map.release(); I.d_out.release(); I.d_raw.release();
        I.reg_host.unregister();
        if (I.verbose) std::cout << "# F-DEVUPDATE: out of device memory, host update" << std::endl;
        return false;
      }
      gpu_check(cudaMemcpy(I.d_map.template as<integer_t>(), map, nnz*sizeof(integer_t), cudaMemcpyHostToDevice));
      gpu_check(cudaMemset(I.d_raw.template as<scalar_t>(), 0, nnz_in*sizeof(scalar_t)));
      I.nnz = nnz; I.nnz_in = nnz_in; I.host_vals = host_vals;
      I.map_ok = true;
      return true;
    }

    template<typename scalar_t, typename integer_t> bool
    DeviceValueUpdater<scalar_t,integer_t>::update(const scalar_t* raw, std::size_t nnz_in) {
      auto& I = *impl_;
      if (!I.map_ok || nnz_in != I.nnz_in || !raw) return false;
      const auto t0 = std::chrono::steady_clock::now();
      I.enqueue(raw, true);   // stream order keeps it behind the previous write-back
      I.gen++;
      I.current = true;
      I.t_update = t0;
      I.host_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t0).count();
      return true;
    }

    template<typename scalar_t, typename integer_t> void
    DeviceValueUpdater<scalar_t,integer_t>::host_values_changed() {
      impl_->current = false;
    }

    template<typename scalar_t, typename integer_t> void
    DeviceValueUpdater<scalar_t,integer_t>::invalidate() {
      auto& I = *impl_;
      if (I.stream) gpu_check(cudaStreamSynchronize(I.stream));
      I.map_ok = I.current = I.wb_pending = false;
      I.reg_raw.unregister();
      I.reg_host.unregister();
      I.host_vals = nullptr; I.raw_ptr = nullptr;
      I.nnz = I.nnz_in = 0;
    }

    template<typename scalar_t, typename integer_t> const scalar_t*
    DeviceValueUpdater<scalar_t,integer_t>::device_values
    (std::size_t nnz, unsigned long long* generation) {
      auto& I = *impl_;
      if (!I.map_ok || !I.current || nnz != I.nnz) return nullptr;
      const auto t0 = std::chrono::steady_clock::now();
      gpu_check(cudaEventSynchronize(I.e_ready));
      const double wait_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (!I.verify()) {
        // stale page-locked registration (the caller replaced its array) or an array modified while the
        // upload was in flight: drop the registration and redo the upload through the staging buffer
        std::cerr << "# F-DEVUPDATE WARNING: uploaded values differ from the host array, "
          "re-uploading (stale host registration?)" << std::endl;
        I.reg_raw.unregister();
        I.enqueue(I.raw_ptr, false);
        gpu_check(cudaStreamSynchronize(I.stream));
        I.wb_pending = false;
        if (!I.verify())
          std::cerr << "# F-DEVUPDATE ERROR: device values could not be verified after re-upload; "
            "results may be wrong" << std::endl;
      }
      float copy_ms = 0.f, gather_ms = 0.f;
      cudaEventElapsedTime(&copy_ms, I.e_start, I.e_copied);
      cudaEventElapsedTime(&gather_ms, I.e_copied, I.e_gathered);
      cudaGetLastError();
      if (I.verbose) {
        const double since = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - I.t_update).count();
        const double gb = double(I.nnz_in) * sizeof(scalar_t) / 1e9;
        std::cout << "# F-DEVUPDATE gen " << I.gen << " host_update_ms " << I.host_ms
                  << " pinned " << (I.pinned_last ? 1 : 0)
                  << " copy_ms " << copy_ms << " (" << gb / (copy_ms * 1e-3) << " GB/s, " << gb << " GB, "
                  << I.nchunks << " chunks) gather_ms " << gather_ms
                  << " factor_wait_ms " << wait_ms << " update_to_factor_ms " << since << std::endl;
      }
      if (generation) *generation = I.gen;
      return I.d_out;
    }

    template<typename scalar_t, typename integer_t> void
    DeviceValueUpdater<scalar_t,integer_t>::release_device_values() {
      auto& I = *impl_;
      if (I.wb_pending) {
        gpu_check(cudaEventSynchronize(I.e_wb));
        I.wb_pending = false;
      }
    }

    template class DeviceValueUpdater<float,int>;
    template class DeviceValueUpdater<double,int>;
    template class DeviceValueUpdater<std::complex<float>,int>;
    template class DeviceValueUpdater<std::complex<double>,int>;
    template class DeviceValueUpdater<float,long>;
    template class DeviceValueUpdater<double,long>;
    template class DeviceValueUpdater<std::complex<float>,long>;
    template class DeviceValueUpdater<std::complex<double>,long>;
    template class DeviceValueUpdater<float,long long>;
    template class DeviceValueUpdater<double,long long>;
    template class DeviceValueUpdater<std::complex<float>,long long>;
    template class DeviceValueUpdater<std::complex<double>,long long>;

  } // end namespace gpu
} // end namespace strumpack
