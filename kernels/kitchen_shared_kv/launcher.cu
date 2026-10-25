// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include "qk_int_sv_i8_packed_fine_cuda.cuh"
#include "qk_occ_alias.cuh"
#include "qk_occ_micro64.cuh"
extern "C" int launch(int mode, void* q, void* k, void* v, void* o,
    void* qs, void* ks, void* vs, void* lut, void* counts, const int* a, void* stream) {
  // a: B,H,Nq,Nk,groups, q(b,s,h),k(b,s,h),v(b,h,d),o(b,s,h),qs(b,h),lut_stride
  auto original=qk_int_sv_i8_packed_fine_attn_kernel<32,half,false,1>;
  auto alias=kitchen_occ_alias_kernel<32,half,false,2>;
  auto micro=kitchen_occ_micro64_kernel<32,half,false,2>;
  auto fn=mode==0?original:(mode==1?alias:micro);
  const int smem=mode==0?40960:24576;
  auto error=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  if(error!=cudaSuccess)return int(error);
  fn<<<dim3((a[2]+63)/64,a[1],a[0]),dim3(32,4),smem,(cudaStream_t)stream>>>(
    (int8_t*)q,(int8_t*)k,(int8_t*)v,(half*)o,nullptr,(float*)qs,(float*)ks,(float*)vs,
    a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],a[18],
    0.08838834764831845f,(int32_t*)lut,(int32_t*)counts,a[19]);
  return int(cudaGetLastError());
}
