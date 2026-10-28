// F-DEVUPDATE: kernel launchers (DeviceValueUpdate.cu) used by DeviceValueUpdate.cpp
#ifndef STRUMPACK_DEVICE_VALUE_UPDATE_KERNELS_HPP
#define STRUMPACK_DEVICE_VALUE_UPDATE_KERNELS_HPP

#include <cstddef>
#include <cuda_runtime_api.h>

namespace strumpack {
  namespace gpu {

    // position of sample i of ns in an array of nnz_in values (same formula on host and device)
    __host__ __device__ inline std::size_t
    devupd_sample_index(int i, int ns, std::size_t nnz_in) {
      const std::size_t stride = nnz_in / std::size_t(ns);
      std::size_t p = std::size_t(i) * stride + stride / 2;
      return p < nnz_in ? p : nnz_in - 1;
    }

    // out (T, nnz) = cast of raw (S, nnz_in) gathered through map; T == S or (float, double) / complex alike
    template<typename T, typename S, typename I> void
    devupd_gather(std::size_t n, std::size_t nnz_in, const I* map, const S* raw, T* out, cudaStream_t s);

    template<typename S> void
    devupd_sample(int ns, std::size_t nnz_in, const S* raw, S* samp, cudaStream_t s);

  } // end namespace gpu
} // end namespace strumpack

#endif
