// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include "kernel_m0.cuh"
#include "kernel_m1.cuh"
#include "kernel_m2.cuh"
#include "kernel_m3.cuh"
#include "kernel_m4.cuh"
extern "C" int launch(int mode, void* q, void* k, void* v, void* o,
    void* qs, void* ks, void* vs, void* lut, void* counts, const int* a, void* stream) {
  // a: B,H,Nq,Nk,groups, q(b,s,h),k(b,s,h),v(b,h,d),o(b,s,h),qs(b,h),lut_stride
  auto f0=kitchen_e48_m0<32,half,false,2>;
  auto f1=kitchen_e48_m1<32,half,false,2>;
  auto f2=kitchen_e48_m2<32,half,false,2>;
  auto f3=kitchen_e48_m3<32,half,false,2>;
  auto f4=kitchen_e48_m4<32,half,false,2>;
  auto fn=mode==0?f0:mode==1?f1:mode==2?f2:mode==3?f3:f4;
  const int smem=24576;
  auto error=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  if(error!=cudaSuccess)return int(error);
  fn<<<dim3((a[2]+63)/64,a[1],a[0]),dim3(32,4),smem,(cudaStream_t)stream>>>(
    (int8_t*)q,(int8_t*)k,(int8_t*)v,(half*)o,nullptr,(float*)qs,(float*)ks,(float*)vs,
    a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],a[18],
    0.08838834764831845f,(int32_t*)lut,(int32_t*)counts,a[19]);
  return int(cudaGetLastError());
}

// Pack sorted selected tiles and dense context directly into token offsets.
// Padding repeats last live tile; attention masks its extra final subtiles.
__global__ void pack_offsets_kernel(const int32_t* selected,int32_t* offsets,int32_t* counts,
 int qtiles,int kvtiles,int qstart,int context,int retained,int stride) {
 int row=blockIdx.x,q=row%qtiles,bh=row/qtiles;
 bool sparse=q>=qstart && retained<kvtiles-context;
 int count=sparse?context+retained:kvtiles;
 if(threadIdx.x==0)counts[row]=count;
 for(int j=threadIdx.x;j<stride;j+=blockDim.x){
  int at=min(j,count-1),tile=at;
  if(sparse && at>=context)tile=selected[(bh*(qtiles-qstart)+(q-qstart))*retained+at-context]+context;
  offsets[row*stride+j]=tile*32;
 }
}
extern "C" int pack_offsets(void* selected,void* offsets,void* counts,int bh,int qtiles,int kvtiles,
 int qstart,int context,int retained,int stride,void* stream){
 pack_offsets_kernel<<<bh*qtiles,128,0,(cudaStream_t)stream>>>((int32_t*)selected,(int32_t*)offsets,(int32_t*)counts,qtiles,kvtiles,qstart,context,retained,stride);
 return int(cudaGetLastError());
}
