// F-DEVUPDATE (magma-hpc F track, 2026-10-27): see DeviceValueUpdate.hpp
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
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
      static constexpr int NS = 4096;           // verification samples
      static constexpr std::size_t MAX_ELEM = 16; // largest raw element (complex<double>)
      cudaStream_t stream = nullptr;            // non-blocking: nothing of the factorization waits on it implicitly
      cudaEvent_t e_order = nullptr, e_start = nullptr, e_copied = nullptr, e_gathered = nullptr,
        e_ready = nullptr, e_wb = nullptr;
      DeviceMemory<char> d_raw, d_samp;         // raw values (src_t), samples (src_t)
      DeviceMemory<scalar_t> d_out;             // the permuted values, consumed by the factorization
      DeviceMemory<integer_t> d_map;
      HostMemory<char> h_stage, h_samp;
      std::size_t nnz = 0, nnz_in = 0, raw_elem = 0;
      scalar_t* host_vals = nullptr;            // the matrix' own values (write-back target)
      HostReg reg_raw, reg_host;
      const void* raw_ptr = nullptr;
      std::function<void(bool)> redo;           // re-enqueue of the last update (repair path)
      unsigned long long gen = 0;
      bool map_ok = false, current = false, wb_pending = false, copied_once = false;
      bool host_stale = false, never_warned = false;   // host_vals lag behind d_out
      bool pinned_last = false, staged_warned = false;
      std::chrono::steady_clock::time_point t_update;
      double host_ms = 0.;
      const int nchunks = std::max(1, env_int("STRUM_DEVUPDATE_CHUNKS", 8));
      const bool pin = env_int("STRUM_DEVUPDATE_PIN", 1) != 0;
      // STRUM_DEVUPDATE_WRITEBACK: unset = LAZY (the host values are refreshed only when a host consumer asks,
      // never in the direct-solve flow), 1 = eager (D2H right after the factorization is issued), 0 = never
      const int writeback = env_int("STRUM_DEVUPDATE_WRITEBACK", 2);
      const bool verbose = env_int("STRUM_DEVUPDATE_VERBOSE", 0) != 0;

      Impl() {
        gpu_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        gpu_check(cudaEventCreateWithFlags(&e_order, cudaEventDisableTiming));
        gpu_check(cudaEventCreate(&e_start));
        gpu_check(cudaEventCreate(&e_copied));
        gpu_check(cudaEventCreate(&e_gathered));
        gpu_check(cudaEventCreateWithFlags(&e_ready, cudaEventDisableTiming));
        gpu_check(cudaEventCreateWithFlags(&e_wb, cudaEventDisableTiming));
        h_samp = HostMemory<char>(std::size_t(NS) * MAX_ELEM);
        d_samp = DeviceMemory<char>(std::size_t(NS) * MAX_ELEM);
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

      // raw -> d_raw (chunked), gather(+cast), samples; all on `stream`
      template<typename src_t> void enqueue(const src_t* raw, bool try_pin) {
        const std::size_t bytes = nnz_in * sizeof(src_t);
        // order after everything queued on the legacy default stream (the previous factorization reads d_out
        // there) and, through stream order, after our own previous work (incl. the write-back reading d_out)
        gpu_check(cudaEventRecord(e_order, 0));
        gpu_check(cudaStreamWaitEvent(stream, e_order, 0));
        const bool pinned = try_pin && pin && reg_raw.ensure(raw, bytes);
        if (!pinned) {
          if (!staged_warned && verbose) {
            std::cout << "# F-DEVUPDATE: value array not page-locked, staged copy" << std::endl;
            staged_warned = true;
          }
          if (h_stage.size() < bytes) h_stage = HostMemory<char>(bytes);
          if (copied_once) gpu_check(cudaEventSynchronize(e_copied));   // previous use of the staging buffer
        }
        src_t* d_raw_t = reinterpret_cast<src_t*>(d_raw.template as<char>());
        gpu_check(cudaEventRecord(e_start, stream));
        const std::size_t chunk = (nnz_in + nchunks - 1) / nchunks;
        for (int c=0; c<nchunks; c++) {
          const std::size_t lo = std::size_t(c)*chunk, hi = std::min(nnz_in, lo+chunk);
          if (lo >= hi) break;
          const src_t* src = raw + lo;
          if (!pinned) {
            src_t* stg = reinterpret_cast<src_t*>(h_stage.data()) + lo;
#pragma omp parallel for
            for (std::size_t k=0; k<hi-lo; k++) stg[k] = raw[lo+k];
            src = stg;
          }
          gpu_check(cudaMemcpyAsync(d_raw_t + lo, src, (hi-lo)*sizeof(src_t), cudaMemcpyHostToDevice, stream));
        }
        gpu_check(cudaEventRecord(e_copied, stream));
        copied_once = true;
        devupd_gather<scalar_t,src_t,integer_t>(nnz, nnz_in, d_map, d_raw_t, d_out, stream);
        gpu_check(cudaEventRecord(e_gathered, stream));
        const int ns = nsamples();
        devupd_sample<src_t>(ns, nnz_in, d_raw_t, reinterpret_cast<src_t*>(d_samp.template as<char>()), stream);
        gpu_check(cudaMemcpyAsync(h_samp.data(), d_samp.template as<char>(), ns*sizeof(src_t),
                                  cudaMemcpyDeviceToHost, stream));
        gpu_check(cudaEventRecord(e_ready, stream));
        raw_ptr = raw; raw_elem = sizeof(src_t);
        pinned_last = pinned;
      }

      // the uploaded samples must match the host array (catches a stale registration / a modified array)
      bool verify() const {
        const int ns = nsamples();
        const char* raw = static_cast<const char*>(raw_ptr);
        for (int i=0; i<ns; i++)
          if (std::memcmp(h_samp.data() + std::size_t(i)*raw_elem,
                          raw + devupd_sample_index(i, ns, nnz_in)*raw_elem, raw_elem))
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
      if (I.writeback != 0 && !I.reg_host.ensure(host_vals, nnz*sizeof(scalar_t))) {
        if (I.verbose)
          std::cout << "# F-DEVUPDATE: cannot page-lock the matrix values for the write-back, host update" << std::endl;
        return false;
      }
      try {
        if (I.d_map.size() < nnz) I.d_map = DeviceMemory<integer_t>(nnz);
        if (I.d_out.size() < nnz) I.d_out = DeviceMemory<scalar_t>(nnz);
      } catch (const std::bad_alloc&) {
        I.d_map.release(); I.d_out.release(); I.d_raw.release();
        I.reg_host.unregister();
        if (I.verbose) std::cout << "# F-DEVUPDATE: out of device memory, host update" << std::endl;
        return false;
      }
      gpu_check(cudaMemcpy(I.d_map.template as<integer_t>(), map, nnz*sizeof(integer_t), cudaMemcpyHostToDevice));
      I.nnz = nnz; I.nnz_in = nnz_in; I.host_vals = host_vals;
      I.map_ok = true;
      return true;
    }

    template<typename scalar_t, typename integer_t> template<typename src_t> bool
    DeviceValueUpdater<scalar_t,integer_t>::update_from(const src_t* raw, std::size_t nnz_in) {
      auto& I = *impl_;
      if (!I.map_ok || nnz_in != I.nnz_in || !raw) return false;
      const auto t0 = std::chrono::steady_clock::now();
      const std::size_t bytes = nnz_in * sizeof(src_t);
      if (I.d_raw.size() < bytes) {   // raw buffer sized for the source type (first update / larger type)
        try { I.d_raw = DeviceMemory<char>(bytes); }
        catch (const std::bad_alloc&) {
          if (I.verbose) std::cout << "# F-DEVUPDATE: out of device memory for the raw values, host update" << std::endl;
          return false;
        }
        gpu_check(cudaMemset(I.d_raw.template as<char>(), 0, bytes));
      }
      I.template enqueue<src_t>(raw, true);
      I.redo = [&I, raw](bool try_pin) { I.template enqueue<src_t>(raw, try_pin); };
      I.gen++;
      I.current = true;
      I.host_stale = true;
      I.t_update = t0;
      I.host_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t0).count();
      return true;
    }

    template<typename scalar_t, typename integer_t> bool
    DeviceValueUpdater<scalar_t,integer_t>::update(const scalar_t* raw, std::size_t nnz_in) {
      return update_from<scalar_t>(raw, nnz_in);
    }

    template<typename scalar_t, typename integer_t> void
    DeviceValueUpdater<scalar_t,integer_t>::host_values_changed() {
      impl_->current = false;
      impl_->host_stale = false;
    }

    template<typename scalar_t, typename integer_t> void
    DeviceValueUpdater<scalar_t,integer_t>::invalidate() {
      auto& I = *impl_;
      if (I.stream) gpu_check(cudaStreamSynchronize(I.stream));
      I.map_ok = I.current = I.wb_pending = I.host_stale = false;
      I.reg_raw.unregister();
      I.reg_host.unregister();
      I.host_vals = nullptr; I.raw_ptr = nullptr; I.redo = nullptr;
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
        I.redo(false);
        gpu_check(cudaStreamSynchronize(I.stream));
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
        const double gb = double(I.nnz_in) * I.raw_elem / 1e9;
        std::cout << "# F-DEVUPDATE gen " << I.gen << " host_update_ms " << I.host_ms
                  << " pinned " << (I.pinned_last ? 1 : 0) << " raw_elem_bytes " << I.raw_elem
                  << " copy_ms " << copy_ms << " (" << gb / (copy_ms * 1e-3) << " GB/s, " << gb << " GB, "
                  << I.nchunks << " chunks) gather_ms " << gather_ms
                  << " factor_wait_ms " << wait_ms << " update_to_factor_ms " << since << std::endl;
      }
      if (generation) *generation = I.gen;
      return I.d_out;
    }

    template<typename scalar_t, typename integer_t> void
    DeviceValueUpdater<scalar_t,integer_t>::factor_enqueued() {
      auto& I = *impl_;
      if (I.writeback != 1 || !I.host_stale || !I.current || !I.host_vals) return;
      // eager mode: after the factorization work (legacy default stream order), refresh the host values; this
      // overlaps the factorization kernels and is waited for only by host consumers (sync_host_values)
      gpu_check(cudaEventRecord(I.e_order, 0));
      gpu_check(cudaStreamWaitEvent(I.stream, I.e_order, 0));
      gpu_check(cudaMemcpyAsync(I.host_vals, I.d_out.template as<scalar_t>(), I.nnz*sizeof(scalar_t),
                                cudaMemcpyDeviceToHost, I.stream));
      gpu_check(cudaEventRecord(I.e_wb, I.stream));
      I.wb_pending = true;
    }

    template<typename scalar_t, typename integer_t> void
    DeviceValueUpdater<scalar_t,integer_t>::sync_host_values() {
      auto& I = *impl_;
      if (I.wb_pending) {   // eager copy in flight
        gpu_check(cudaEventSynchronize(I.e_wb));
        I.wb_pending = false;
        I.host_stale = false;
        return;
      }
      if (!I.host_stale || !I.current || !I.host_vals) return;
      if (I.writeback == 0) {
        if (!I.never_warned)
          std::cerr << "# F-DEVUPDATE WARNING: a host consumer reads the matrix values but "
            "STRUM_DEVUPDATE_WRITEBACK=0 forbids the write-back; the host values are stale" << std::endl;
        I.never_warned = true;
        return;
      }
      // lazy mode: copy the permuted values back now (after the gather and whatever was issued on the legacy stream)
      const auto t0 = std::chrono::steady_clock::now();
      gpu_check(cudaEventRecord(I.e_order, 0));
      gpu_check(cudaStreamWaitEvent(I.stream, I.e_order, 0));
      gpu_check(cudaMemcpyAsync(I.host_vals, I.d_out.template as<scalar_t>(), I.nnz*sizeof(scalar_t),
                                cudaMemcpyDeviceToHost, I.stream));
      gpu_check(cudaStreamSynchronize(I.stream));
      I.host_stale = false;
      if (I.verbose)
        std::cout << "# F-DEVUPDATE lazy write-back of the host values: "
                  << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t0).count()
                  << " ms (" << I.nnz*sizeof(scalar_t)/1e6 << " MB)" << std::endl;
    }

#define DEVUPD_INST(S, I)                                                \
    template class DeviceValueUpdater<S,I>;
#define DEVUPD_INST_FROM(S, F, I)                                       \
    template bool DeviceValueUpdater<S,I>::update_from<F>(const F*, std::size_t);
#define DEVUPD_INST_ALL(I)                                               \
    DEVUPD_INST(float, I) DEVUPD_INST(double, I)                         \
    DEVUPD_INST(std::complex<float>, I) DEVUPD_INST(std::complex<double>, I) \
    DEVUPD_INST_FROM(float, float, I) DEVUPD_INST_FROM(double, double, I) \
    DEVUPD_INST_FROM(std::complex<float>, std::complex<float>, I)        \
    DEVUPD_INST_FROM(std::complex<double>, std::complex<double>, I)      \
    DEVUPD_INST_FROM(float, double, I)                                   \
    DEVUPD_INST_FROM(std::complex<float>, std::complex<double>, I)
    DEVUPD_INST_ALL(int)
    DEVUPD_INST_ALL(long)
    DEVUPD_INST_ALL(long long)

  } // end namespace gpu
} // end namespace strumpack
