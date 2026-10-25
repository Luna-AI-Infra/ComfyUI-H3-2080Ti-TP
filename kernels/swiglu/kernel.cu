// E45: preserve separate SiLU rounding for the main FP16 path;
// produce the FP32 LoRA activation directly from the original input.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

template<typename T> __device__ float read_value(T v) { return (float)v; }
template<typename T> __device__ T cast_value(float v) { return (T)v; }
template<> __device__ half cast_value<half>(float v) { return __float2half_rn(v); }

template<typename T, bool Main, bool Lora>
__global__ void swiglu_kernel(const T* __restrict__ x, T* __restrict__ y,
                              float* __restrict__ lora, int64_t count, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int64_t row = i / k, col = i - row * k;
    float gate = read_value(x[row * (2LL*k) + col]);
    float up = read_value(x[row * (2LL*k) + k + col]);
    float silu = gate / (1.0f + expf(-gate));
    if (Main) {
        T rounded = cast_value<T>(silu);
        y[i] = cast_value<T>(read_value(rounded) * up);
    }
    if (Lora) lora[i] = silu * up;
}

template<typename T> int dispatch(const void* x, void* y, void* lora,
                                 int64_t count, int k, int flags, int threads,
                                 cudaStream_t stream) {
    int blocks = (count + threads - 1) / threads;
    if (flags == 3) swiglu_kernel<T,true,true><<<blocks,threads,0,stream>>>((const T*)x,(T*)y,(float*)lora,count,k);
    else if (flags == 1) swiglu_kernel<T,true,false><<<blocks,threads,0,stream>>>((const T*)x,(T*)y,nullptr,count,k);
    else if (flags == 2) swiglu_kernel<T,false,true><<<blocks,threads,0,stream>>>((const T*)x,nullptr,(float*)lora,count,k);
    else return (int)cudaErrorInvalidValue;
    return (int)cudaGetLastError();
}

extern "C" int launch_swiglu(const void* x, void* y, void* lora, int m, int k,
                              int dtype, int flags, int threads, void* stream) {
    if (m <= 0 || k <= 0 || (threads != 128 && threads != 256 && threads != 512)) return (int)cudaErrorInvalidValue;
    if (dtype == 0) return dispatch<half>(x,y,lora,(int64_t)m*k,k,flags,threads,(cudaStream_t)stream);
    if (dtype == 1) return dispatch<float>(x,y,lora,(int64_t)m*k,k,flags,threads,(cudaStream_t)stream);
    return (int)cudaErrorInvalidValue;
}
