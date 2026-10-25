// SM75 exact INT8->INT32 GEMM. CUTLASS BSD-3-Clause templates.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
using RM=cutlass::layout::RowMajor;using CM=cutlass::layout::ColumnMajor;
using Ep=cutlass::epilogue::thread::LinearCombination<int32_t,4,int32_t,int32_t>;
template<int BM,int BN,int BK,int WM,int WN,int SW=1> struct Ops {
 using Shape=cutlass::gemm::GemmShape<BM,BN,BK>;using Warp=cutlass::gemm::GemmShape<WM,WN,BK>;using Inst=cutlass::gemm::GemmShape<8,8,16>;
 using Base=cutlass::gemm::device::Gemm<int8_t,RM,int8_t,CM,int32_t,RM,int32_t,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,Shape,Warp,Inst,Ep,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SW>,2,16,16>;
 using Split=cutlass::gemm::device::GemmSplitKParallel<int8_t,RM,int8_t,CM,int32_t,RM,int32_t,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,Shape,Warp,Inst,Ep,cutlass::epilogue::thread::Convert<int32_t,4,int32_t>,cutlass::reduction::thread::ReduceAdd<int32_t,int32_t,4>,cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,2,16,16>;
 static long long bytes(int m,int n,int k,int s){typename Split::Arguments args({m,n,k},{nullptr,k},{nullptr,k},{nullptr,n},{nullptr,n},{1,0},s);return s==1?0:Split::get_workspace_size(args);}
 static int launch(int m,int n,int k,int s,void*a,void*b,void*c,void*ws,void*stream){
  if(s==1){Base op;typename Base::Arguments args({m,n,k},{(int8_t*)a,k},{(int8_t*)b,k},{(int32_t*)c,n},{(int32_t*)c,n},{1,0});return int(op(args,nullptr,(cudaStream_t)stream));}
  Split op;typename Split::Arguments args({m,n,k},{(int8_t*)a,k},{(int8_t*)b,k},{(int32_t*)c,n},{(int32_t*)c,n},{1,0},s);return int(op(args,ws,(cudaStream_t)stream));
 }
};
#define CONFIGS(F) \
 F(0,128,64,64,64,64,1) \
 F(1,128,128,64,64,64,1) \
 F(2,256,128,64,64,64,1) \
 F(3,128,256,64,64,64,1) \
 F(4,64,128,64,32,64,1) \
 F(5,128,128,32,64,64,1) \
 F(6,256,128,32,64,64,1) \
 F(7,128,256,32,64,64,1) \
 F(8,128,64,128,64,64,1) \
 F(9,128,128,128,64,64,1) \
 F(10,128,128,64,32,64,1) \
 F(11,256,128,64,64,64,4) \
 F(12,128,256,64,64,64,4)
extern "C" long long workspace_bytes(int id,int m,int n,int k,int s){
#define B(I,M,N,K,WM,WN,SW) if(id==I)return Ops<M,N,K,WM,WN,SW>::bytes(m,n,k,s);
 CONFIGS(B)
#undef B
 return -1;
}
extern "C" int launch(int id,int m,int n,int k,int s,void*a,void*b,void*c,void*ws,void*stream){
#define L(I,M,N,K,WM,WN,SW) if(id==I)return Ops<M,N,K,WM,WN,SW>::launch(m,n,k,s,a,b,c,ws,stream);
 CONFIGS(L)
#undef L
 return -1;
}
