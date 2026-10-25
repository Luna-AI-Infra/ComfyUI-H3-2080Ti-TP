// E49: exact FP16 SwiGLU, vectorized paired gate/value loads and output stores.
// Retain the original rounding after SiLU and after multiplication.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
template<int V> struct Pack;
template<> struct Pack<2>{using Bits=uint32_t;};
template<> struct Pack<4>{using Bits=uint2;};
template<> struct Pack<8>{using Bits=uint4;};
template<int V> union Vector {typename Pack<V>::Bits bits;half h[V];};
template<int V>
__global__ void vae_swiglu_vec(const half* __restrict__ x,half* __restrict__ y,int count,int k){
 int packet=blockIdx.x*blockDim.x+threadIdx.x;
 int i=packet*V;if(i>=count)return;
 int row=i/k,col=i-row*k;
 Vector<V> g,v,o;
 g.bits=*reinterpret_cast<const typename Pack<V>::Bits*>(x+row*(2*k)+col);
 v.bits=*reinterpret_cast<const typename Pack<V>::Bits*>(x+row*(2*k)+k+col);
#pragma unroll
 for(int j=0;j<V;++j){float gate=__half2float(g.h[j]);half silu=__float2half_rn(gate/(1.0f+expf(-gate)));o.h[j]=__hmul(silu,v.h[j]);}
 *reinterpret_cast<typename Pack<V>::Bits*>(y+i)=o.bits;
}
extern "C" int launch_vae_swiglu(void*x,void*y,int m,int k,int vec,int threads,void*stream){
 if(m<=0||k<=0||k%vec||((int64_t)m*k>2147483647)|| (threads!=128 && threads!=256))return (int)cudaErrorInvalidValue;
 int count=m*k,blocks=(count/vec+threads-1)/threads;cudaStream_t s=(cudaStream_t)stream;
 if(vec==2)vae_swiglu_vec<2><<<blocks,threads,0,s>>>((half*)x,(half*)y,count,k);
 else if(vec==4)vae_swiglu_vec<4><<<blocks,threads,0,s>>>((half*)x,(half*)y,count,k);
 else if(vec==8)vae_swiglu_vec<8><<<blocks,threads,0,s>>>((half*)x,(half*)y,count,k);
 else return (int)cudaErrorInvalidValue;
 return (int)cudaGetLastError();
}
