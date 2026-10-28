// F-DEVUPDATE kernels (see DeviceValueUpdate.hpp): gather with the cached
// symmetrize+permute map, and a strided sample of the uploaded raw values.
#include <cuda_runtime.h>
#include <complex>
#include <cstdlib>

#include "dense/CUDAWrapper.hpp"
#include "DeviceValueUpdateKernels.hpp"

namespace strumpack {
  namespace gpu {

    // out[k] = map[k] > 0 ? raw[map[k]-1] : 0 (the explicit zeros of the symmetrization)
    template<typename T, typename I> __global__ void
    devupd_gather_kernel(std::size_t n, std::size_t nnz_in, const I* __restrict__ map,
                         const T* __restrict__ raw, T* __restrict__ out) {
      const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
      for (std::size_t k = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < n; k += stride) {
        I m = map[k];
#if defined(STRUM_PLANT_DEVUPD_OFF1)
        // PLANT devupd-off1: off-by-one in the 1-based map for the last of 8 output chunks
        if (k >= n - n / 8 && m > 0 && std::size_t(m) < nnz_in) m = m + 1;
#endif
        T z = {};
        out[k] = m > 0 ? raw[m-1] : z;
      }
    }

    template<typename T> __global__ void
    devupd_sample_kernel(int ns, std::size_t nnz_in, const T* __restrict__ raw, T* __restrict__ samp) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < ns) samp[i] = raw[devupd_sample_index(i, ns, nnz_in)];
    }

    template<typename T, typename I> void
    devupd_gather(std::size_t n, std::size_t nnz_in, const I* map, const T* raw, T* out, cudaStream_t s) {
      if (!n) return;
      using T_ = typename cuda_type<T>::value_type;
      const unsigned nt = 256;
      const std::size_t nb = (n + nt - 1) / nt;
      const unsigned bl = unsigned(std::min<std::size_t>(nb, 8192));
      devupd_gather_kernel<T_,I><<<bl, nt, 0, s>>>
        (n, nnz_in, map, reinterpret_cast<const T_*>(raw), reinterpret_cast<T_*>(out));
      gpu_check(cudaPeekAtLastError());
    }

    template<typename T> void
    devupd_sample(int ns, std::size_t nnz_in, const T* raw, T* samp, cudaStream_t s) {
      if (ns <= 0) return;
      using T_ = typename cuda_type<T>::value_type;
      devupd_sample_kernel<T_><<<(ns + 255) / 256, 256, 0, s>>>
        (ns, nnz_in, reinterpret_cast<const T_*>(raw), reinterpret_cast<T_*>(samp));
      gpu_check(cudaPeekAtLastError());
    }

    template void devupd_gather(std::size_t, std::size_t, const int*, const float*, float*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const int*, const double*, double*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const int*, const std::complex<float>*, std::complex<float>*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const int*, const std::complex<double>*, std::complex<double>*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const long*, const float*, float*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const long*, const double*, double*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const long*, const std::complex<float>*, std::complex<float>*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const long*, const std::complex<double>*, std::complex<double>*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const long long*, const float*, float*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const long long*, const double*, double*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const long long*, const std::complex<float>*, std::complex<float>*, cudaStream_t);
    template void devupd_gather(std::size_t, std::size_t, const long long*, const std::complex<double>*, std::complex<double>*, cudaStream_t);
    template void devupd_sample(int, std::size_t, const float*, float*, cudaStream_t);
    template void devupd_sample(int, std::size_t, const double*, double*, cudaStream_t);
    template void devupd_sample(int, std::size_t, const std::complex<float>*, std::complex<float>*, cudaStream_t);
    template void devupd_sample(int, std::size_t, const std::complex<double>*, std::complex<double>*, cudaStream_t);

  } // end namespace gpu
} // end namespace strumpack
