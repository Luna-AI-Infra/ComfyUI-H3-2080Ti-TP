#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
struct Plan{cublasLtHandle_t h;cublasLtMatmulDesc_t op;cublasLtMatrixLayout_t A,B,C;cublasLtMatmulAlgo_t algo;};
extern "C" void* create(int m,int n,int k,void*bias){
 auto*p=new Plan{};if(cublasLtCreate(&p->h))return nullptr;
 cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_16F,CUDA_R_16F);
 cublasOperation_t tr=CUBLAS_OP_T;cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&tr,sizeof(tr));
 if(bias){cublasLtEpilogue_t ep=CUBLASLT_EPILOGUE_BIAS;cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep));cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias));}
 cublasLtMatrixLayoutCreate(&p->A,CUDA_R_16F,k,n,k);cublasLtMatrixLayoutCreate(&p->B,CUDA_R_16F,k,m,k);cublasLtMatrixLayoutCreate(&p->C,CUDA_R_16F,n,m,n);
 cublasLtMatmulPreference_t pref;cublasLtMatmulPreferenceCreate(&pref);size_t ws=128ULL<<20;cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws));
 cublasLtMatmulHeuristicResult_t result;int count=0;auto status=cublasLtMatmulAlgoGetHeuristic(p->h,p->op,p->A,p->B,p->C,p->C,pref,1,&result,&count);cublasLtMatmulPreferenceDestroy(pref);
 if(status||!count||result.state)return nullptr;p->algo=result.algo;return p;
}
extern "C" int run(void*v,void*x,void*w,void*y,void*ws,void*stream){auto*p=(Plan*)v;__half alpha=__float2half(1.f),beta=__float2half(0.f);return cublasLtMatmul(p->h,p->op,&alpha,w,p->A,x,p->B,&beta,y,p->C,y,p->C,&p->algo,ws,128ULL<<20,(cudaStream_t)stream);}
extern "C" void destroy(void*v){auto*p=(Plan*)v;cublasLtMatrixLayoutDestroy(p->A);cublasLtMatrixLayoutDestroy(p->B);cublasLtMatrixLayoutDestroy(p->C);cublasLtMatmulDescDestroy(p->op);cublasLtDestroy(p->h);delete p;}
