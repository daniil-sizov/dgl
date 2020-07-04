/*** 
The code under BASE is the original DGL code, by default BASE is 0.
PCL optimized code is under #else .. #endif portion.
Authors: Md. Vasimuddin (PCL, Intel), Sanchit Misra (PCL, Intel), Sasikant Avancha (PCL, Intel),
Ramanarayan Mohanty (PCL, Intel)

Notes: 
***/

/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_sum.cc
 * \brief CPU kernels for binary reduce sum
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"
#include "macro.h"

#if BASE
namespace dgl {
namespace kernel {
 
#define REDUCER ReduceSum
#define XPU kDLCPU

#define IDX int32_t
EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE);
#undef IDX

#define IDX int64_t
EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE);
#undef IDX

}  // namespace kernel
}  // namespace dgl


#else
/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_sum.cc
 * \brief CPU kernels for binary reduce sum
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"
#include "../utils.h"
#include "../csr_interface.h"
#include <functional>
#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <sys/syscall.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <math.h>
#define gettid() ((int)syscall(SYS_gettid))

extern int64_t acc_cycs;
// kmp_affinity_mask_t mask;

using minigun::advance::RuntimeConfig;

#include <mkl_spblas.h>
#include <mkl.h>

namespace dgl {
namespace kernel {
namespace cpu {


template <typename DType>
void sparse_mm3(const RuntimeConfig &rtcfg,
                const aten::CSRMatrix& csr,
                DType* B, DType* C,
                int x_length)
{
#define K_BLOCK_SIZE 1024 //4096
#define K_BLOCK_MASK 1023 //4095
#define N_BLOCK_SIZE 640
#define SORT 1
	const int M = csr.num_rows;
	const int N = x_length;
	const int K = csr.num_cols;
	int nthreads = omp_get_max_threads();

	int perThreadQuota = (M + nthreads - 1) / nthreads;
#pragma omp parallel num_threads(nthreads)
    {
        int *my_cur_col_id = (int *)_mm_malloc(2 * perThreadQuota * sizeof(int), 64);
         
        int tid = omp_get_thread_num();		
		
        int M_start = tid * perThreadQuota; 
        int M_end = (tid + 1) * perThreadQuota; 
        if(M_end > M) M_end = M;
        //int16_t count[K_BLOCK_SIZE];
        int16_t *count = (int16_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int16_t), 64);
        // int32_t *count = (int32_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int32_t), 64);
        for(int i = M_start; i < M_end; i++)
        {
            my_cur_col_id[(i - M_start) * 2] = static_cast<int32_t*>(csr.indptr->data)[i];
            my_cur_col_id[(i - M_start) * 2 + 1] = static_cast<int32_t*>(csr.indptr->data)[i + 1];
        }

        int max_nnz = 0;
        for(int a_col = 0; a_col < K; a_col += K_BLOCK_SIZE)
        {
            int nnz = 0;
            //printf("%d] a_col = %d, nnz = %ld\n", tid, a_col, nnz);
            for(int i = M_start; i < M_end; i++)
            {
                const int row_start = my_cur_col_id[(i - M_start) * 2];
                const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                for(int eid = row_start; eid < row_end; eid++)
                {
                    const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                    if(dst >= a_col + K_BLOCK_SIZE)
                    {
                        my_cur_col_id[(i - M_start) * 2] = eid;
                        break;
                    }
                    nnz++;
                }
            }
            if(nnz > max_nnz) max_nnz = nnz;
            //printf("%d] a_col = %d, nnz = %ld\n", tid, a_col, nnz);
        }
        //printf("%d] max_nnz = %ld\n", tid, max_nnz);
        int max_nnz_aligned = ((max_nnz + 15) / 16) * 16;
        //int my_nz_cells[max_nnz_aligned];
        //int my_nz_cells_sorted[max_nnz_aligned];
        int *my_nz_cells = (int *)_mm_malloc(2 * max_nnz_aligned * sizeof(int), 64);
        int *my_nz_cells_sorted = (int *)_mm_malloc(2 * max_nnz_aligned * sizeof(int), 64);

        for(int N_block_start = 0; N_block_start < N; N_block_start += N_BLOCK_SIZE)
        {
            int N_block_end = N_block_start + N_BLOCK_SIZE;
            if(N_block_end > N) N_block_end = N;
            for(int i = M_start; i < M_end; i++)
            {
                my_cur_col_id[(i - M_start) * 2] = static_cast<int32_t*>(csr.indptr->data)[i];
                my_cur_col_id[(i - M_start) * 2 + 1] = static_cast<int32_t*>(csr.indptr->data)[i + 1];
            }

            for(int a_col = 0; a_col < K; a_col += K_BLOCK_SIZE)
            {
                int nnz = 0;
                for(int i = M_start; i < M_end; i++)
                {
                    const int row_start = my_cur_col_id[(i - M_start) * 2];
                    const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                    for(int eid = row_start; eid < row_end; eid++)
                    {
                        const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                        if(dst >= a_col + K_BLOCK_SIZE)
                        {
                            break;
                        }
                        my_nz_cells[nnz * 2] = i;
                        my_nz_cells[nnz * 2 + 1] = dst;
                        my_cur_col_id[(i - M_start) * 2] = eid + 1;
                        nnz++;
                    }
                }
#if SORT
                //printf("nnz = %d\n", nnz);
                // Radix sort nz_cells according to dst
                memset(count, 0, K_BLOCK_SIZE * sizeof(int16_t));
                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 2 + 1] & K_BLOCK_MASK;
                    count[v]++;
                }
                int cumulative_sum = 0;
                for(int t = 0; t < K_BLOCK_SIZE; t++)
                {
                    int c = count[t];
                    count[t] = cumulative_sum;
                    cumulative_sum += c;
                }

                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 2 + 1] & K_BLOCK_MASK;
                    int ind = count[v];
                    my_nz_cells_sorted[ind * 2] = my_nz_cells[t * 2];
                    my_nz_cells_sorted[ind * 2 + 1] = my_nz_cells[t * 2 + 1];
                    count[v]++;
                }
#endif

                for(int t = 0; t < nnz; t++)
                {
#if SORT
                    int i = my_nz_cells_sorted[t * 2];
                    int dst = my_nz_cells_sorted[t * 2 + 1];
#else
                    int i = my_nz_cells[t * 2];
                    int dst = my_nz_cells[t * 2 + 1];
#endif

#if 1
                    int j;
#pragma unroll(16)
                    for(j = N_block_start; j < N_block_end - 16; j += 16)
                    {
                        __m512 c512 = _mm512_loadu_ps(&C[i * N + j]);
                        __m512 b512 = _mm512_loadu_ps(&B[dst * N + j]);
                        c512 = _mm512_add_ps(b512, c512);
                        _mm512_storeu_ps(&C[i * N + j], c512);
                    }
                    for(; j < N_block_end; j++)
                    {
                        C[i * N + j] += B[dst * N + j];
                    }
#else
#pragma omp simd
                    for(int j = N_block_start; j < N_block_end; j++)
                    {
                        C[i * N + j] += B[dst * N + j];
                    }
#endif
                }
            }
        }
        _mm_free(count);
        _mm_free(my_cur_col_id);
        _mm_free(my_nz_cells);
        _mm_free(my_nz_cells_sorted);
    }
    
    #undef K_BLOCK_SIZE
    #undef K_BLOCK_MASK
    #undef N_BLOCK_SIZE
    #undef SORT    
}

__m512 maxv512(__m512 a, __m512 b) {
    return (_mm512_max_ps(a, b));
}
float maxs(float a, float b) {
    return (a>b?a:b);
}
__m512 addv512(__m512 a, __m512 b) {
    return (_mm512_add_ps(a, b));
}
float adds(float a, float b) {
    return (a+b);
}

template <typename DType>
void sparse_mm3_gen(const RuntimeConfig &rtcfg,
                    const aten::CSRMatrix& csr,
                    DType* B, DType* C,
                    int x_length,
                    __m512 (*f_vec)(__m512, __m512),
                    float (*f)(float, float))
{
    #define K_BLOCK_SIZE 1024 //4096
    #define K_BLOCK_MASK 1023 //4095
    #define N_BLOCK_SIZE 640
    #define SORT 1
    
	const int M = csr.num_rows;
	const int N = x_length;
	const int K = csr.num_cols;
	int nthreads = omp_get_max_threads();

	int perThreadQuota = (M + nthreads - 1) / nthreads;
#pragma omp parallel num_threads(nthreads)
    {
        int *my_cur_col_id = (int *)_mm_malloc(2 * perThreadQuota * sizeof(int), 64);
         
        int tid = omp_get_thread_num();		
		
        int M_start = tid * perThreadQuota; 
        int M_end = (tid + 1) * perThreadQuota; 
        if(M_end > M) M_end = M;
        //int16_t count[K_BLOCK_SIZE];
        // int16_t *count = (int16_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int16_t), 64);
        int32_t *count = (int32_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int32_t), 64);
        for(int i = M_start; i < M_end; i++)
        {
            my_cur_col_id[(i - M_start) * 2] = static_cast<int32_t*>(csr.indptr->data)[i];
            my_cur_col_id[(i - M_start) * 2 + 1] = static_cast<int32_t*>(csr.indptr->data)[i + 1];
        }

        int max_nnz = 0;
        for(int a_col = 0; a_col < K; a_col += K_BLOCK_SIZE)
        {
            int nnz = 0;
            //printf("%d] a_col = %d, nnz = %ld\n", tid, a_col, nnz);
            for(int i = M_start; i < M_end; i++)
            {
                const int row_start = my_cur_col_id[(i - M_start) * 2];
                const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                for(int eid = row_start; eid < row_end; eid++)
                {
                    const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                    if(dst >= a_col + K_BLOCK_SIZE)
                    {
                        my_cur_col_id[(i - M_start) * 2] = eid;
                        break;
                    }
                    nnz++;
                }
            }
            if(nnz > max_nnz) max_nnz = nnz;
            //printf("%d] a_col = %d, nnz = %ld\n", tid, a_col, nnz);
        }

        //printf("%d] max_nnz = %ld\n", tid, max_nnz);
        int max_nnz_aligned = ((max_nnz + 15) / 16) * 16;
        //int my_nz_cells[max_nnz_aligned];
        //int my_nz_cells_sorted[max_nnz_aligned];
        int *my_nz_cells = (int *)_mm_malloc(2 * max_nnz_aligned * sizeof(int), 64);
        int *my_nz_cells_sorted = (int *)_mm_malloc(2 * max_nnz_aligned * sizeof(int), 64);

        for(int N_block_start = 0; N_block_start < N; N_block_start += N_BLOCK_SIZE)
        {
            int N_block_end = N_block_start + N_BLOCK_SIZE;
            if(N_block_end > N) N_block_end = N;
            for(int i = M_start; i < M_end; i++)
            {
                my_cur_col_id[(i - M_start) * 2] = static_cast<int32_t*>(csr.indptr->data)[i];
                my_cur_col_id[(i - M_start) * 2 + 1] = static_cast<int32_t*>(csr.indptr->data)[i + 1];
            }

            for(int a_col = 0; a_col < K; a_col += K_BLOCK_SIZE)
            {
                int nnz = 0;
                for(int i = M_start; i < M_end; i++)
                {
                    const int row_start = my_cur_col_id[(i - M_start) * 2];
                    const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                    for(int eid = row_start; eid < row_end; eid++)
                    {
                        const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                        if(dst >= a_col + K_BLOCK_SIZE)
                        {
                            break;
                        }
                        my_nz_cells[nnz * 2] = i;
                        my_nz_cells[nnz * 2 + 1] = dst;
                        my_cur_col_id[(i - M_start) * 2] = eid + 1;
                        nnz++;
                    }
                }
#if SORT
                // printf("nnz = %d\n", nnz);
                // Radix sort nz_cells according to dst
                // memset(count, 0, K_BLOCK_SIZE * sizeof(int16_t));
                memset(count, 0, K_BLOCK_SIZE * sizeof(int32_t));
                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 2 + 1] & K_BLOCK_MASK;
                    count[v]++;
                }
                int cumulative_sum = 0;
                for(int t = 0; t < K_BLOCK_SIZE; t++)
                {
                    int c = count[t];
                    count[t] = cumulative_sum;
                    cumulative_sum += c;
                }

                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 2 + 1] & K_BLOCK_MASK;
                    int ind = count[v];
                    my_nz_cells_sorted[ind * 2] = my_nz_cells[t * 2];
                    my_nz_cells_sorted[ind * 2 + 1] = my_nz_cells[t * 2 + 1];
                    count[v]++;
                }
#endif

                for(int t = 0; t < nnz; t++)
                {
#if SORT
                    int i = my_nz_cells_sorted[t * 2];
                    int dst = my_nz_cells_sorted[t * 2 + 1];
#else
                    int i = my_nz_cells[t * 2];
                    int dst = my_nz_cells[t * 2 + 1];
#endif

#if 1
                    int j;
#pragma unroll(16)
                    for(j = N_block_start; j < N_block_end - 16; j += 16)
                    {
                        __m512 c512 = _mm512_loadu_ps(&C[i * N + j]);
                        __m512 b512 = _mm512_loadu_ps(&B[dst * N + j]);
                        // c512 = _mm512_add_ps(b512, c512);
                        c512 = f_vec(c512, b512);
                        _mm512_storeu_ps(&C[i * N + j], c512);
                    }
                    for(; j < N_block_end; j++)
                    {
                        // C[i * N + j] += B[dst * N + j];
                        C[i * N + j] = f(C[i * N + j], B[dst * N + j]);
                    }
#else
#pragma omp simd
                    for(int j = N_block_start; j < N_block_end; j++)
                    {
                        // C[i * N + j] += B[dst * N + j];
                        C[i * N + j] = f(C[i * N + j], B[dst * N + j]);
                    }
#endif
                }
            }
        }
        _mm_free(count);
        _mm_free(my_cur_col_id);
        _mm_free(my_nz_cells);
        _mm_free(my_nz_cells_sorted);
    }
    
    #undef K_BLOCK_SIZE
    #undef K_BLOCK_MASK
    #undef N_BLOCK_SIZE
    #undef SORT    
}

template <typename DType>
void sparse_mkl(const RuntimeConfig &rtcfg,
                const aten::CSRMatrix& csr,
                DType* B, DType* C,
                int x_length)
{
	const int M = csr.num_rows;
	const int N = x_length;
	const int K = csr.num_cols;
	const int nnz = csr.indices->shape[0];
	const DType alpha = 1.0;
	const DType beta = 0.0;
	
	DType* valptr = (DType*) malloc(nnz * sizeof(DType));
	utils::Fill<kDLCPU>(rtcfg.ctx, valptr, nnz, static_cast<DType>(1.));
	
	matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	
	sparse_matrix_t csr_mat_t; // = &csr_mat;

	mkl_sparse_s_create_csr(&csr_mat_t, SPARSE_INDEX_BASE_ZERO, M, N,
							static_cast<int32_t*>(csr.indptr->data),
							static_cast<int32_t*>(csr.indptr->data) + 1,
							static_cast<int32_t*>(csr.indices->data),
							valptr);
	
	
	mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
					alpha,
					csr_mat_t,  /* Sparse matrix */
					descr,                      
					SPARSE_LAYOUT_ROW_MAJOR,  /* Storage scheme for dense matrix*/
					B, N, N,
					beta,
					C,
					N);
	free(valptr);	
}


/*********************** Fallback CallBinaryReduce/cast code *****************/
// forward
// template <typename DType>
template <int XPU, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void FallbackCallBinaryReduce(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, DType>* gdata) {
    // constexpr int XPU = kDLCPU;
    // typedef int32_t Idx;
    // typedef SelectSrc LeftSelector;
    // typedef SelectNone RightSelector;
    // typedef BinaryUseLhs<DType> BinaryOp;
    // typedef ReduceSum<kDLCPU, DType> Reducer;
    typedef cpu::FunctorsTempl<Idx, DType, LeftSelector,
                               RightSelector, BinaryOp, Reducer>
        Functors;
    typedef cpu::BinaryReduce<Idx, DType, Functors> UDF;
    #if DTRACE
    std::cout << "Trace: FallbackCallBinaryReduce\n";
    #endif
    
    // csr
    auto outcsr = graph.GetOutCSRMatrix();
    minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
    // If the user-given mapping is none and the target is edge data, we need to
    // replace the mapping by the edge ids in the csr graph so that the edge
    // data is correctly read/written.
    if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
        gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
    }
    if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
        gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
    }
    if (OutSelector<Reducer>::Type::target == binary_op::kEdge
        && gdata->out_mapping == nullptr) {
        gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
    }
    // TODO(minjie): allocator
    minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig, GData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

// template <typename DType>
template <int XPU, int Mode, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void FallbackCallBackwardBinaryReduce(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, DType>* gdata) {

    #if DTRACE
    std::cout << "Trace: FallbackCallBackwardBinaryReduce\n";
    #endif
    // constexpr int XPU = kDLCPU;
    // constexpr int Mode = binary_op::kGradLhs;
    // typedef int32_t Idx;
    // typedef SelectSrc LeftSelector;
    // typedef SelectNone RightSelector;
    // typedef BinaryUseLhs<DType> BinaryOp;
    // typedef ReduceSum<kDLCPU, DType> Reducer;
    
    // For backward computation, we use reverse csr and switch dst and src.
    // This benefits the most common src_op_edge or copy_src case, because the
    // gradients of src are now aggregated into destination buffer to reduce
    // competition of atomic add.
    auto incsr = graph.GetInCSRMatrix();
    minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
    typedef cpu::BackwardFunctorsTempl<Idx, DType,
                                       typename SwitchSrcDst<LeftSelector>::Type,
                                       typename SwitchSrcDst<RightSelector>::Type,
                                       BinaryOp, Reducer> Functors;
    typedef cpu::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
    // If the user-given mapping is none and the target is edge data, we need to
    // replace the mapping by the edge ids in the csr graph so that the edge
    // data is correctly read/written.
    if (LeftSelector::target == binary_op::kEdge
        && gdata->lhs_mapping == nullptr) {
        gdata->lhs_mapping = static_cast<Idx*>(incsr.data->data);
    }
    if (RightSelector::target == binary_op::kEdge
        && gdata->rhs_mapping == nullptr) {
        gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
    }
    if (OutSelector<Reducer>::Type::target == binary_op::kEdge
        && gdata->out_mapping == nullptr) {
        gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
    }
    // TODO(minjie): allocator
    minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig, BackwardGData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}


// Template implementation of BinaryReduce broadcasting operator.
template <int XPU, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void FallbackCallBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BcastGData<NDim, Idx, DType>* gdata) {

    #if DTRACE
    std::cout << "Trace: FallbackCallBinaryReduceBcast\n";
    #endif
    
    typedef cpu::FunctorsTempl<Idx, DType, LeftSelector,
                               RightSelector, BinaryOp, Reducer>
        Functors;
    if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        typedef cpu::BinaryReduceBcast<NDim, Idx, DType, Functors> UDF;  
        auto outcsr = graph.GetOutCSRMatrix(); // csr
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
        // If the user-given mapping is none and the target is edge data, we need to
        // replace the mapping by the edge ids in the csr graph so that the edge
        // data is correctly read/written.
        if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
            gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        if (OutSelector<Reducer>::Type::target == binary_op::kEdge
            && gdata->out_mapping == nullptr) {
            gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        // TODO(minjie): allocator
        minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig,
                                  BcastGData<NDim, Idx, DType>, UDF>(
                                      rtcfg, csr, gdata, minigun::IntArray1D<Idx>());    
    }
    else {
        #if BASE
        typedef cpu::BinaryReduceBcast<NDim, Idx, DType, Functors> UDF;  
        auto outcsr = graph.GetOutCSRMatrix(); // csr
        #else
        // Intel code
        typedef cpu::BinaryReduceBcastIntel<NDim, Idx, DType, Functors> UDF;  
        auto outcsr = graph.GetInCSRMatrix(); // csr  
        #endif
    
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
        // If the user-given mapping is none and the target is edge data, we need to
        // replace the mapping by the edge ids in the csr graph so that the edge
        // data is correctly read/written.
        if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
            gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        if (OutSelector<Reducer>::Type::target == binary_op::kEdge
            && gdata->out_mapping == nullptr) {
            gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        // TODO(minjie): allocator
        minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig,
                                  BcastGData<NDim, Idx, DType>, UDF>(
                                      rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
    }
}

// Template implementation of BackwardBinaryReduce with broadcasting operator.
template <int XPU, int Mode, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void FallbackCallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<NDim, Idx, DType>* gdata) {
    // For backward computation, we use reverse csr and switch dst and src.
    // This benefits the most common src_op_edge or copy_src case, because the
    // gradients of src are now aggregated into destination buffer to reduce
    // competition of atomic add.
    typedef cpu::BackwardFunctorsTempl<Idx, DType,
                                       typename SwitchSrcDst<LeftSelector>::Type,
                                       typename SwitchSrcDst<RightSelector>::Type,
                                       BinaryOp, Reducer> Functors;
    
    if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "Trace: FallbackCallBackwardBinaryReduceBcast BASE\n";
        #endif
        
        typedef cpu::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
        auto incsr = graph.GetInCSRMatrix();
        // typedef cpu::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
        // auto incsr = graph.GetInCSRMatrix();
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
  
        // If the user-given mapping is none and the target is edge data, we need to
        // replace the mapping by the edge ids in the csr graph so that the edge
        // data is correctly read/written.
        if (LeftSelector::target == binary_op::kEdge
            && gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<Idx*>(incsr.data->data);
        }
        if (RightSelector::target == binary_op::kEdge
            && gdata->rhs_mapping == nullptr) {
            gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
        }
        if (OutSelector<Reducer>::Type::target == binary_op::kEdge
            && gdata->out_mapping == nullptr) {
            gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
        }
        // TODO(minjie): allocator
        minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig,
                                  BackwardBcastGData<NDim, Idx, DType>, UDF>(
                                      rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
    }
    else
    {
        #if BASE
        std::cout << "CallBackwardBinaryReduceBcast BASE..\n";
        typedef cpu::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
        auto incsr = graph.GetInCSRMatrix();
        #else
        std::cout << "CallBackwardBinaryReduceBcast Intel..\n";
        // Intel code
        typedef cpu::BackwardBinaryReduceBcastIntel<Mode, NDim, Idx, DType, Functors> UDF;
        auto incsr = graph.GetOutCSRMatrix();
        #endif
  
        // typedef cpu::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
        // auto incsr = graph.GetInCSRMatrix();
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
  
        // If the user-given mapping is none and the target is edge data, we need to
        // replace the mapping by the edge ids in the csr graph so that the edge
        // data is correctly read/written.
        if (LeftSelector::target == binary_op::kEdge
            && gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<Idx*>(incsr.data->data);
        }
        if (RightSelector::target == binary_op::kEdge
            && gdata->rhs_mapping == nullptr) {
            gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
        }
        if (OutSelector<Reducer>::Type::target == binary_op::kEdge
            && gdata->out_mapping == nullptr) {
            gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
        }
        // TODO(minjie): allocator
        minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig,
                                  BackwardBcastGData<NDim, Idx, DType>, UDF>(
                                      rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
    }
}
}  // namespace cpu

/*************************** copy-reduce: binary_reduce<float/double> ***************************/
// CR: u_copy_sum_v
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectNone,
                      BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, float>* gdata) {
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "If CallBinaryReduce_Float_Sum In\n";
        #endif
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectNone,
                                      BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
        #if DTRACE
        std::cout << "Else CallBinaryReduce_Float_Sum In\n";
        #endif
        
		// cusparse use rev csr for csrmm
		// uint64_t tim =__rdtsc();
		
		auto csr = graph.GetInCSRMatrix();
        // cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        // gdata->x_length);
        cpu::sparse_mkl(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);
	}
}

template <>
void CallBinaryReduce<kDLCPU, int32_t, double, SelectSrc, SelectNone,
                      BinaryUseLhs<double>, ReduceSum<kDLCPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, double>* gdata) {

    #if DTRACE
    std::cout<< "In CallBinaryReduce_Double_Sum\n";
	#endif
    
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, double, SelectSrc, SelectNone,
                                      BinaryUseLhs<double>, ReduceSum<kDLCPU, double>>
            (rtcfg, graph, gdata);
	} else {
		// cusparse use rev csr for csrmm
		auto csr = graph.GetInCSRMatrix();
        cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                              gdata->x_length);                
	}
}

// CR: backward
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {
    
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "IF CallBackwardBinaryReduce FLOAT SUM\n";
        #endif
        
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                              SelectSrc, SelectNone,
                                              BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
        #if DTRACE
        std::cout << "Else CallBackwardBinaryReduce FLOAT SUM\n";
        #endif
		auto csr = graph.GetOutCSRMatrix();        
        // cpu::sparse_mm3(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
        //             gdata->x_length);
        cpu::sparse_mkl(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
                        gdata->x_length);        
	}
}

template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, double,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<double>, ReduceSum<kDLCPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, double>* gdata) {
    if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, double,
                                              SelectSrc, SelectNone,
                                              BinaryUseLhs<double>, ReduceSum<kDLCPU, double>>
            (rtcfg, graph, gdata);
    } else {
        auto csr = graph.GetOutCSRMatrix();
        cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);
    }
}

template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectNone,
                      BinaryDot<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, float>* gdata) {
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectNone,
                                      BinaryDot<float>, ReduceSum<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
		// cusparse use rev csr for csrmm		
		auto csr = graph.GetInCSRMatrix();
        cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);
	}
}

/*************************** binary-reduce-bcast: binary_reduce_bcast<float> ***************************/
#define NDIM4 2
int prod(int *a, int n) {
    int val = 1;
    for (auto i=0; i<n; i++) val *= a[i];
    return val;
}
int64_t prod(int64_t *a, int n) {
    int val = 1;
    for (auto i=0; i<n; i++) val *= a[i];
    return val;
}

void broadcast(int index_out, int index_lhs, int lframe, int out_dim, int64_t *out_shape,
               int64_t *lhs_shape, float *lhs_data, float *out_data)
{
    
    if (out_dim == 1)
    {
        if (lhs_shape[1] != out_shape[1])
        {
            // int lhs_off = lhs_shape[0] * lhs_shape[1] * index;
            // int out_off = out_shape[0] * out_shape[1] * index;
            int lhs_off = index_lhs;
            int out_off = index_out;
            
            float *lhs_base_ = lhs_data + lhs_off;
            float *out_base_ = out_data + out_off;
            
            for (auto i=0; i<lhs_shape[0]; i++)
            {
                float *lhs_base = lhs_base_ + lhs_shape[1] * i;
                float *out_base = out_base_ + out_shape[1] * i;
                int num_val = lhs_shape[1];
                float value = lhs_base[num_val - 1];
                memcpy(out_base, lhs_base, num_val * sizeof(float));
                // memset(out_base + num_val, value, out_shape[1] - num_val);
                std::fill_n(out_base + num_val, out_shape[1] - num_val, value);
            }
        }
        if (lhs_shape[0] != out_shape[0])
        {
            int lhs_off = index_out + (lhs_shape[0] - 1)* out_shape[1];
            int out_off = index_out + lhs_shape[0] * out_shape[1];
            float *lhs_base = out_data + lhs_off;
            float *out_base_ = out_data + out_off;
            int num_val = out_shape[1];
            
            for (auto i=0; i< out_shape[0] - lhs_shape[0]; i++)
            {
                float *out_base = out_base_ + out_shape[1] * i;
                memcpy(out_base, lhs_base, num_val * sizeof(float));
            }
        }
        
        return;
    }
    else if (lhs_shape[out_dim] == 1) {
        int lhs_off = index_out - lframe;
        int out_off = index_out;
        
        float *lhs_base_ = out_data + lhs_off;
        float *out_base_ = out_data + out_off;
        memcpy(out_base_, lhs_base_, lframe * sizeof(float));                
    }

    for (auto i=0; i<out_shape[out_dim]; i++)
    {
        int index_out_ = index_out + prod(out_shape, out_dim) * i;
        int index_lhs_ = index_lhs + prod(lhs_shape, out_dim) * i; 
        broadcast(index_out_, index_lhs_, prod(out_shape, out_dim), out_dim - 1,
                  out_shape, lhs_shape, lhs_data, out_data);
        
        if ((i - lhs_shape[out_dim]) >= 0) {
            int lframe = prod(out_shape, out_dim);
            int lhs_off = index_out_ - lframe;
            int out_off = index_out_;
            
            float *lhs_base_ = out_data + lhs_off;
            float *out_base_ = out_data + out_off;
            memcpy(out_base_, lhs_base_, lframe * sizeof(float));                
        }
    }
}


/*
  Designed assuming csr-in matrix: broadcasting u and e in the csr-in model
  No edge mapping is required as we are covering all the edges here.
*/
template<typename T>
void bcast_dgl(const aten::CSRMatrix& csr, T* gdata,
               float* out_data_r, float* out_data_l,
               bool lhsf, bool rhsf)
{
	const int M = csr.num_rows;
	// const int N = gdata->out_len;
	// const int K = csr.num_cols;
    const int64_t len = gdata->data_len;
    assert(len == 1);
    assert(gdata->lhs_len != gdata->rhs_len);

    #pragma omp parallel for
    for (int src=0; src<M; src++)
    {
        int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
        int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];

        for (int eid=row_start; eid<row_end; eid++) {
            const int dst = static_cast<int32_t*>(csr.indices->data)[eid];

            const int64_t len = gdata->data_len;
            if (gdata->out_len > gdata->rhs_len && rhsf)
            {
                assert(gdata->lhs_len == gdata->out_len);
                int eid_ = eid;
                //if (mflag)
                //    eid_ = mapping[eid];
                
                // float* lhs_data = gdata->lhs_data + dst * gdata->lhs_len * len; // data with len size
                float* rhs_data = gdata->rhs_data + eid_ * gdata->rhs_len * len;
                // float* out_data = out_data_r + src * gdata->out_len *len;
                float* out_data = out_data_r + eid_ * gdata->out_len *len;
                // std::cout << "eid: " <<eid << std::endl;

                int ndim = gdata->ndim;
                if (ndim == 1) {
                    int rhs_off = gdata->rhs_shape[0];
                    int out_off = gdata->out_shape[0];
                    float value = rhs_data[rhs_off - 1];
                    memcpy(out_data, rhs_data, rhs_off *sizeof(float));
                    // memset(out_data + rhs_off, value, out_off - rhs_off);
                    std::fill_n(out_data + rhs_off, out_off - rhs_off, value);
                }
                else
                {
                    int index_out = 0, index_rhs = 0, lframe = -1;
                    broadcast(index_out, index_rhs, lframe, ndim - 1, gdata->out_shape,
                              gdata->rhs_shape, rhs_data, out_data);                    
                }
            }
            // else {}
            
            if (gdata->out_len > gdata->lhs_len && lhsf)
            {
                assert(gdata->rhs_len == gdata->out_len);
                float* lhs_data = gdata->lhs_data + dst * gdata->lhs_len * len;  // data with len size
                // float* rhs_data = gdata->rhs_data + eid * gdata->rhs_len * len;
                float* out_data = out_data_l + dst * gdata->out_len *len; 

                int ndim = gdata->ndim;
                if (ndim == 1) {
                    int lhs_off = gdata->lhs_shape[0];
                    int out_off = gdata->out_shape[0];
                    float value = lhs_data[lhs_off - 1];
                    memcpy(out_data, lhs_data, lhs_off *sizeof(float));
                    // memset(out_data + rhs_off, value, out_off - rhs_off);
                    std::fill_n(out_data + lhs_off, out_off - lhs_off, value);
                }
                else
                {
                    int index_out = 0, index_lhs = 0, lframe = -1;
                    broadcast(index_out, index_lhs, lframe, ndim - 1, gdata->out_shape,
                              gdata->lhs_shape, lhs_data, out_data);                    
                }                
                
            }
            // else {}            
        }
    } // csr loop
}

// BR: u_mul_e_sum_v
template <>
void CallBinaryReduceBcast<kDLCPU, NDIM4, int, float, SelectSrc, SelectEdge,
                           BinaryMul<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BcastGData<NDIM4, int, float>* gdata) {
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        cpu::FallbackCallBinaryReduceBcast<kDLCPU, NDIM4, int, float, SelectSrc, SelectEdge,
                                           BinaryMul<float>, ReduceSum<kDLCPU, float>>            
            (rtcfg, graph, gdata);
        #if DTRACE
        std::cout << "If CallBinaryReduceBcast u,e, MUL, SUM\n";
        #endif
	}
    else {
		// cusparse use rev csr for csrmm
		auto csr = graph.GetInCSRMatrix();
        int32_t num_edges = static_cast<int32_t*>(csr.indptr->data)[csr.num_rows];
        int32_t out_len = gdata->out_len;
        
        // step 1: bcast
        float *out_data_r, *out_data_l, *xhs_data = NULL;
        int64_t nlength = out_len * csr.num_cols;
        int64_t rlength = out_len * num_edges;

        bool loc[2] = {0};
        if (gdata->out_len > gdata->rhs_len) {
            out_data_r= (float*)malloc(rlength * sizeof(float));
            xhs_data = out_data_r;
            loc[0] = 1;
        }
        else
            out_data_r = gdata->rhs_data;

        if (gdata->out_len > gdata->lhs_len) {
            // std::cout << "lhs edge..\n";
            out_data_l= (float*) malloc(nlength * sizeof(float));
            // xhs_data = out_data_l;
            loc[1] = 1;
        }
        else
            out_data_l = gdata->lhs_data;

        if (gdata->rhs_mapping == nullptr) {
            gdata->rhs_mapping = static_cast<int32_t*>(csr.data->data);
        }

        // bcast_dgl<BcastGData<NDIM4, int32_t, float>>(csr, gdata, out_data_r, out_data_l,
        //                                          gdata->rhs_mapping, mapping);
        bcast_dgl<BcastGData<NDIM4, int32_t, float>>(csr, gdata, out_data_r, out_data_l,
                                                     true, true);
                
        if (!loc[0]) 
            xhs_data = (float*) malloc(rlength * sizeof(float));
        
        // step 2: Multiply
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        int32_t l_restore = static_cast<int32_t*>(csr.indptr->data)[M];
        int32_t *restore = (int32_t*)malloc(l_restore * sizeof(int32_t));
        int K = csr.num_cols;
        int maxval = 0;
        // bug alert: maxval is getting shared here.
        #pragma omp parallel for reduction(max: maxval)
        for (int src=0; src<M; src++)  // level1, v - level1
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];            
            for (int eid=row_start; eid<row_end; eid++)
            {
                const int dst = static_cast<int32_t*>(csr.indices->data)[eid];  // u, level2, u_mul_e => lhs: u and rhs: e
                restore[eid] = static_cast<int32_t*>(csr.indices->data)[eid];
                static_cast<int32_t*>(csr.indices->data)[eid] = gdata->rhs_mapping[eid];
                maxval = maxval < gdata->rhs_mapping[eid]? gdata->rhs_mapping[eid]: maxval;

                int eid_ = gdata->rhs_mapping[eid];
                // printf("src: %d, dst: %d, eid: %d, eid_: %d\n", src, dst, eid, eid_);
                float* out_data = xhs_data   + eid_ * out_len *len;
                float* rhs_data = out_data_r + eid_ * out_len * len;
                float* lhs_data = out_data_l + dst * out_len * len;
                #pragma omp simd
                for (auto l=0; l<out_len; l++) {
                    out_data[l] = lhs_data[l] * rhs_data[l];
                }
            }
        }

        csr.num_cols = maxval + 10;
        
        // step 3: Sparse mm
        cpu::sparse_mkl(rtcfg, csr, xhs_data, gdata->out_data, gdata->out_len);

        // restore
        csr.num_cols = K;
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
            
            for (int eid=row_start; eid<row_end; eid++)
            {
                static_cast<int32_t*>(csr.indices->data)[eid] = restore[eid];
            }
        }

        // free memory
        free(restore);
        if (loc[0])
            free(out_data_r);
        else
            free(xhs_data);
        
        if (loc[1])
            free(out_data_l);

        return;
	}
}

// BackwardBcast: GradLhs
template <>
void CallBackwardBinaryReduceBcast<kDLCPU, binary_op::kGradLhs, NDIM4, int, float,
                                   SelectSrc, SelectEdge,
                                   BinaryMul<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<NDIM4, int, float>* gdata) {
    
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        cpu::FallbackCallBackwardBinaryReduceBcast<kDLCPU, binary_op::kGradLhs, NDIM4, int, float,
                                                   SelectSrc, SelectEdge,
                                                   BinaryMul<float>, ReduceSum<kDLCPU, float>>
            (rtcfg, graph, gdata);
        #if DTRACE
        std::cout << "If (Lhs) CallBackwardBinaryReduceBcast u,e, MUL, SUM\n";
        #endif
	}
    else {
		auto csr = graph.GetOutCSRMatrix();
        int32_t num_edges = static_cast<int32_t*>(csr.indptr->data)[csr.num_rows];
        int32_t out_len = gdata->out_len;

        // step 1: bcast
        float *out_data_r, *out_data_l, *xhs_data = NULL;
        int64_t nlength = gdata->out_len * csr.num_cols;
        int64_t rlength = out_len * num_edges;
        
        bool loc[2] = {0};
        if (gdata->out_len > gdata->rhs_len) {
            out_data_r= (float*)_mm_malloc(rlength * sizeof(float), 64);
            xhs_data = out_data_r;
            loc[0] = 1;
        }
        else
            out_data_r = gdata->rhs_data;


        if (gdata->rhs_mapping == nullptr) {
            gdata->rhs_mapping = static_cast<int32_t*>(csr.data->data);
        }
        
        bcast_dgl<BackwardBcastGData<NDIM4, int, float>>(csr, gdata, out_data_r, out_data_l,
                                                         false, true);

        // step 2: Multiply
        if (!loc[0]) 
            xhs_data = (float*) malloc(rlength * sizeof(float));

        // multi thread this
        int i = 0;        
        float *grad_out = gdata->grad_out_data;
        float *gradlhsoff = gdata->grad_lhs_data;
        // float *outoff   = gdata->out_data;        
        float *grad_e = grad_out;

        // multiply lhs with grad
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        int32_t l_restore = static_cast<int32_t*>(csr.indptr->data)[M];
        int32_t *restore = (int32_t*)malloc(l_restore * sizeof(int32_t));
        int K = csr.num_cols;
        int maxval = 0;

        #pragma omp parallel for reduction(max: maxval)
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];            
            for (int eid=row_start; eid<row_end; eid++)
            {
                const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                restore[eid] = dst;
                int eid_ = gdata->rhs_mapping[eid];
                static_cast<int32_t*>(csr.indices->data)[eid] = eid_;
                maxval = maxval < eid_? eid_: maxval;
                
                float* grad_lhs = xhs_data   + eid_ * out_len *len;
                float* rhs_data = out_data_r + eid_ * out_len * len;
                float* grad_e_data = grad_e  + dst * out_len * len;
                #pragma omp simd
                for (auto l=0; l<out_len; l++) {
                    grad_lhs[l] = grad_e_data[l] * rhs_data[l];
                }
            }
        }
        csr.num_cols = maxval + 10;
        
        // step 3: Sparse mm
        // cpu::sparse_mm3(rtcfg, csr, grad_e, gradlhsoff, gdata->out_len);
        cpu::sparse_mkl(rtcfg, csr, xhs_data, gradlhsoff, gdata->out_len);			

        // restore
        csr.num_cols = K;
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
            
            for (int eid=row_start; eid<row_end; eid++)
            {
                static_cast<int32_t*>(csr.indices->data)[eid] = restore[eid];
            }
        }

        // free memory
        free(restore);
        if (loc[0])
            _mm_free(out_data_r);
        else
            _mm_free(xhs_data);
        
        if (loc[1])
            _mm_free(out_data_l);        
	}
}


// BackwardBcast: GradRhs
template <>
void CallBackwardBinaryReduceBcast<kDLCPU, binary_op::kGradRhs, NDIM4, int, float,
                                   SelectSrc, SelectEdge,
                                   BinaryMul<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<NDIM4, int, float>* gdata) {
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        cpu::FallbackCallBackwardBinaryReduceBcast<kDLCPU, binary_op::kGradRhs, NDIM4, int, float,
                                                   SelectSrc, SelectEdge,
                                                   BinaryMul<float>, ReduceSum<kDLCPU, float>>
            (rtcfg, graph, gdata);
        
        #if DTRACE
        std::cout << "If  (Rhs) CallBackwardBinaryReduceBcast u,e, MUL, SUM\n";
        #endif
	}
    else {
		auto csr = graph.GetOutCSRMatrix();
        // step 1: bcast
        float *out_data_r, *out_data_l, *xhs_data = NULL;
        int64_t nlength = gdata->out_len * csr.num_cols;
        int32_t out_len = gdata->out_len;
        
        bool loc[2] = {0};

        bool lhsf = false;
        if (gdata->out_len > gdata->lhs_len) {
            out_data_l= (float*) _mm_malloc(nlength * sizeof(float), 64);
            // xhs_data = out_data_l;
            loc[1] = 1;
        }
        else
            out_data_l = gdata->lhs_data;
        
        bcast_dgl<BackwardBcastGData<NDIM4, int, float>>(csr, gdata, out_data_r, out_data_l,
                                                         lhsf, false); // lhsf, rhsf

        if (gdata->rhs_mapping == nullptr) {
            gdata->rhs_mapping = static_cast<int32_t*>(csr.data->data);
        }

        // step 2: Multiply        
        float *grad_out = gdata->grad_out_data;
        float *gradrhsoff = gdata->grad_rhs_data;
        // float *outoff   = gdata->out_data;
        float *grad_e = grad_out;

        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;

        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];            
            for (int eid=row_start; eid<row_end; eid++)
            {
                const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                int eid_ = gdata->rhs_mapping[eid];
                
                float* grad_e_data = grad_e  + dst * out_len *len;
                float* rhs_data = gradrhsoff + eid_ * out_len * len;
                float* lhs_data = out_data_l + src * out_len * len;
                #pragma omp simd
                for (auto l=0; l<out_len; l++) {
                    rhs_data[l] = lhs_data[l] * grad_e_data[l];  // u * v -> e
                }
            }
        }
        
        // step 3: Sparse mm
        //cpu::sparse_mm3(rtcfg, csr, grad_e, gradrhsoff, gdata->out_len);
        
        if (loc[0])
            _mm_free(out_data_r);
        if (loc[1])
            _mm_free(out_data_l);
	}
}

/*************************** binary_reduce_dot<float> ***************************/
// Used in GCMC.
// BR: u_dot_v_copy_e
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectDst,
                      BinaryDot<float>, ReduceNone<kDLCPU, float>>(
                          const RuntimeConfig& rtcfg,
                          const CSRWrapper& graph,
                          GData<int32_t, float>* gdata) {

    const int64_t D = gdata->x_length;
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectDst,
                                      BinaryDot<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	}
    else {
		auto csr = graph.GetInCSRMatrix();
        const int M = csr.num_rows;
        // const int N = gdata->out_len;
        // const int K = csr.num_cols;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;

        if (SelectSrc::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<int*>(csr.data->data);
        }
        if (SelectDst::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
            gdata->rhs_mapping = static_cast<int*>(csr.data->data);
        }
        if (gdata->out_mapping == nullptr) {
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }
        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
            
            for (int eid=row_start; eid<row_end; eid++)
            {
                const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                int32_t lid = dst;
                int32_t rid = src;
                int32_t oid = eid;
                
                float* lhsoff = gdata->lhs_data + lid * D * len;
                float* rhsoff = gdata->rhs_data + rid * D * len;
                float* outoff = gdata->out_data + oid * D;
                
                for (int64_t tx = 0; tx < D; ++tx) {
                    float* lhsoff_ = lhsoff + tx * len;
                    float* rhsoff_ = rhsoff + tx * len;
                    float* outoff_ = outoff + tx;
                    
                    #pragma omp simd
                    for (int64_t lx=0; lx<len; lx++) {
                        // float out = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
                        // Functors::Write(outoff + tx, out);
                        *outoff_ += lhsoff_[lx] + rhsoff_[lx];                         
                    }
                }                
            }
        }
	}
}

// Backward: Lhs grad
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectSrc, SelectDst,
                              BinaryDot<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

    const int64_t D = gdata->x_length;
    
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                              SelectSrc, SelectDst,
                                              BinaryDot<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	}
    else {
		auto csr = graph.GetOutCSRMatrix();        

        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
            
            for (int eid=row_start; eid<row_end; eid++)
            {
                const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                int32_t lid = src;
                int32_t rid = dst;
                int32_t oid = eid;
                
                // float* lhsoff = gdata->lhs_data + lid * D * len;
                float* rhsoff = gdata->rhs_data + rid * D * len;
                float* gradlhsoff = gdata->grad_lhs_data + lid * D * len;
                float* gradoutoff = gdata->grad_out_data + oid * D;
                for (int64_t tx = 0; tx < D; ++tx)
                {
                    float grad_e = *(gradoutoff + tx);

                    #pragma omp simd
                    for (int64_t i = 0; i < len; ++i) {
                        gradlhsoff[tx * len + i] += rhsoff[tx * len + i] * grad_e;
                    }
                }                
            }
        }
	}
}

// Backward: Rhs grad
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                              SelectSrc, SelectDst,
                              BinaryDot<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

    const int64_t D = gdata->x_length;
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                                              SelectSrc, SelectDst,
                                              BinaryDot<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	}
    else {
        #if DTRACE
        fprintf(stderr, "In Else CallBackwardBinaryReduce_Float_Dot Rhs\n");
        #endif
        
        auto csr = graph.GetInCSRMatrix();        

        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;

        if (gdata->out_mapping == nullptr) {
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }
        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
             
            for (int eid=row_start; eid<row_end; eid++)
            {
                const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                // int32_t lid = src;
                // int32_t rid = dst;
                // int32_t oid = eid;
                int32_t lid = dst;
                int32_t rid = src;
                int32_t oid = eid;                

                if (gdata->out_mapping) 
                    oid = gdata->out_mapping[oid];
                
                float* lhsoff = gdata->lhs_data + lid * D * len;
                float* gradrhsoff = gdata->grad_rhs_data + rid * D * len;
                float* gradoutoff = gdata->grad_out_data + oid * D;
                for (int64_t tx = 0; tx < D; ++tx)
                {
                    float grad_e = *(gradoutoff + tx);

                    #pragma omp simd
                    for (int64_t i = 0; i < len; ++i) {
                        gradrhsoff[tx * len + i] += lhsoff[tx * len + i] * grad_e;
                    }
                }

                
            }
        }
	}
}

/*********************************** GAT Specific CR/BR *****************************/
// Notes: Keeping all CR/BR for GAT together for now, will rearrange later.

// CR: e_copy_sum_v
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectEdge, SelectNone,
                      BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, float>* gdata) {
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "If GAT CallBinaryReduce_Float_Sum In\n";
        #endif
        
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectEdge, SelectNone,
                                      BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
		auto csr = graph.GetInCSRMatrix();
        #if DTRACE
        std::cout << "Else GAT CallBinaryReduce_Float_Sum In\n";
        #endif

        const int M = csr.num_rows;

        int32_t l_restore = static_cast<int32_t*>(csr.indptr->data)[M];
        int32_t *restore = (int32_t*)malloc(l_restore * sizeof(int32_t));
        
        if (gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<int*>(csr.data->data);
        }

        int K = csr.num_cols;
        int maxval = 0;  // not private to the omp threads below
        // #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
            
            for (int eid=row_start; eid<row_end; eid++)
            {
                restore[eid] = static_cast<int32_t*>(csr.indices->data)[eid];
                int val_map = gdata->lhs_mapping[eid];
                static_cast<int32_t*>(csr.indices->data)[eid] = val_map;
                if (maxval < val_map) {
                    maxval = val_map;
                }
            }
        }                

        csr.num_cols = maxval + 1;

        cpu::sparse_mkl(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);
                
        csr.num_cols = K;        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
            
            for (int eid=row_start; eid<row_end; eid++)
            {
                static_cast<int32_t*>(csr.indices->data)[eid] = restore[eid];
            }
        }
        free(restore);
	}
}

// Backpass: e_copy_sum_v
// Needs scatter operation
// Right now, coded using native technique, can also code it using sparse_mm3
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectEdge, SelectNone,
                              BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>(
                                  const RuntimeConfig& rtcfg,
                                  const CSRWrapper& graph,
                                  BackwardGData<int32_t, float>* gdata) {

    #if DTRACE
    std::cout<< "ElseBackward: 2,NONE, UseLhs, SUM\n";
    #endif
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE    
        std::cout << "IF CallBackwardBinaryReduce FLOAT SUM\n";
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                              SelectEdge, SelectNone,
                                              BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
    
        auto csr = graph.GetOutCSRMatrix();
        const int M = csr.num_rows;
        if (gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<int*>(csr.data->data);
        }
   
        int K = csr.num_cols;
        int D = gdata->x_length;
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                int eid_ = gdata->lhs_mapping[eid];
                int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                float *grad_outoff = gdata->grad_out_data + D*dst;
                float *grad_lhsoff = gdata->grad_lhs_data + D*eid_;

                #pragma omp simd
                for (auto tx=0; tx<D; tx++) {
                    grad_lhsoff[tx] = grad_outoff[tx];
                }
            }
        }
    }
}

// CR: e_copy_max_v
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectEdge, SelectNone,
                      BinaryUseLhs<float>, ReduceMax<kDLCPU, float>>(
                          const RuntimeConfig& rtcfg,
                          const CSRWrapper& graph,
                          GData<int32_t, float>* gdata) {

    #if DTRACE
    std::cout << "GAT CallBinaryReduce_Float_Sum MAX\n";
    #endif
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "If GAT CallBinaryReduce_Float_Sum MAX\n";
        #endif
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectEdge, SelectNone,
                      BinaryUseLhs<float>, ReduceMax<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
		auto csr = graph.GetInCSRMatrix();
        #if DTRACE
        std::cout << "Else GAT CallBinaryReduce_Float_Sum MAX\n";
        #endif
        
        const int M = csr.num_rows;

        int32_t l_restore = static_cast<int32_t*>(csr.indptr->data)[M];
        int32_t *restore = (int32_t*)malloc(l_restore * sizeof(int32_t));
        
        if (gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<int*>(csr.data->data);
        }
        int K = csr.num_cols;
        int maxval = 0;
        
        // #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
            
            for (int eid=row_start; eid<row_end; eid++)
            {
                restore[eid] = static_cast<int32_t*>(csr.indices->data)[eid];
                static_cast<int32_t*>(csr.indices->data)[eid] = gdata->lhs_mapping[eid];
                maxval = maxval < gdata->lhs_mapping[eid]? gdata->lhs_mapping[eid]: maxval;
            }
        }

        csr.num_cols = maxval + 10;
        cpu::sparse_mm3_gen(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                            gdata->x_length, cpu::maxv512, cpu::maxs);
        // restore
        csr.num_cols = K;
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
            
            for (int eid=row_start; eid<row_end; eid++)
            {
                static_cast<int32_t*>(csr.indices->data)[eid] = restore[eid];
            }
        }
        free(restore);
	}
}

// Backpass: e_copy_max_v
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectEdge, SelectNone,
                              BinaryUseLhs<float>, ReduceMax<kDLCPU, float>>(
                                  const RuntimeConfig& rtcfg,
                                  const CSRWrapper& graph,
                                  BackwardGData<int32_t, float>* gdata) {
    #if DTRACE
    std::cout << "ElseBackward: 2,NONE, UseLhs, MAX\n";
    #endif
    
    cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                          SelectEdge, SelectNone,
                                           BinaryUseLhs<float>, ReduceMax<kDLCPU, float>>
        (rtcfg, graph, gdata);
    return;
    
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "IF CallBackwardBinaryReduce FLOAT MAX\n";
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                              SelectEdge, SelectNone,
                                              BinaryUseLhs<float>, ReduceMax<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
        #if DTRACE
        std::cout << "Else CallBackwardBinaryReduce FLOAT MAX\n";
        #endif
        
        auto csr = graph.GetOutCSRMatrix();
        const int M = csr.num_rows;
        if (gdata->lhs_mapping == nullptr) {
            gdata->lhs_mapping = static_cast<int*>(csr.data->data);
        }
        
        int K = csr.num_cols;
        int D = gdata->x_length;
        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                int eid_ = gdata->lhs_mapping[eid];
                int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                float *grad_outoff = gdata->grad_out_data + D*dst;
                float *grad_lhsoff = gdata->grad_lhs_data + D*eid_;
                float *out_data = gdata->lhs_data + D * eid_;
                float *gout_data = gdata->out_data + D*dst;
                
                #pragma omp simd
                for (auto tx=0; tx<D; tx++) {
                    float val = grad_outoff[tx];
                    if (out_data[tx] != gout_data[tx])
                        val = 0;
                    grad_lhsoff[tx] = val;
                }
            }
        }
    }
}

// Notes: Re-purposed sparse_mm3 for BR: u_add_v_copy_v 
template <typename DType>
void sparse_mm_u_add_v_e(const RuntimeConfig &rtcfg,
                         const aten::CSRMatrix& csr,
                         DType* BL, DType* BR, DType* C,
                         int x_length)
{
    #define SORT 0
    #define K_BLOCK_SIZE 20480 // 1024 //4096
    #define K_BLOCK_MASK 1023 //4095
    #define N_BLOCK_SIZE 640

	const int M = csr.num_rows;
	const int N = x_length;
	const int K = csr.num_cols;
    
    int NE = static_cast<int32_t*>(csr.indptr->data)[M+1];
    int nthreads = omp_get_max_threads();
    
	int perThreadQuota = (M + nthreads - 1) / nthreads;
#pragma omp parallel num_threads(nthreads)
    {
        int *my_cur_col_id = (int *)_mm_malloc(2 * perThreadQuota * sizeof(int), 64);
         
        int tid = omp_get_thread_num();		
		
        int M_start = tid * perThreadQuota; 
        int M_end = (tid + 1) * perThreadQuota; 
        if(M_end > M) M_end = M;
        
        // int16_t count[K_BLOCK_SIZE];
        // int16_t *count = (int16_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int16_t), 64);
        
        int32_t *count = (int32_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int32_t), 64);
        for(int i = M_start; i < M_end; i++)
        {
            my_cur_col_id[(i - M_start) * 2] = static_cast<int32_t*>(csr.indptr->data)[i];
            my_cur_col_id[(i - M_start) * 2 + 1] = static_cast<int32_t*>(csr.indptr->data)[i + 1];
        }
        
        int max_nnz = 0;
        for(int a_col = 0; a_col < K; a_col += K_BLOCK_SIZE)
        {
            int nnz = 0;
            //printf("%d] a_col = %d, nnz = %ld\n", tid, a_col, nnz);
            for(int i = M_start; i < M_end; i++)
            {
                const int row_start = my_cur_col_id[(i - M_start) * 2];
                const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                for(int eid = row_start; eid < row_end; eid++)
                {
                    const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                    if(dst >= a_col + K_BLOCK_SIZE)
                    {
                        my_cur_col_id[(i - M_start) * 2] = eid;
                        break;
                    }
                    nnz++;
                }
            }
            if(nnz > max_nnz) max_nnz = nnz;
            //printf("%d] a_col = %d, nnz = %ld\n", tid, a_col, nnz);
        }
        //printf("%d] max_nnz = %ld\n", tid, max_nnz);
        int max_nnz_aligned = ((max_nnz + 15) / 16) * 16;
        //int my_nz_cells[max_nnz_aligned];
        //int my_nz_cells_sorted[max_nnz_aligned];
        int *my_nz_cells = (int *)_mm_malloc(3 * max_nnz_aligned * sizeof(int), 64);
        int *my_nz_cells_sorted = (int *)_mm_malloc(3 * max_nnz_aligned * sizeof(int), 64);

        for(int N_block_start = 0; N_block_start < N; N_block_start += N_BLOCK_SIZE)
        {
            int N_block_end = N_block_start + N_BLOCK_SIZE;
            if(N_block_end > N) N_block_end = N;
            for(int i = M_start; i < M_end; i++)
            {
                my_cur_col_id[(i - M_start) * 2] = static_cast<int32_t*>(csr.indptr->data)[i];
                my_cur_col_id[(i - M_start) * 2 + 1] = static_cast<int32_t*>(csr.indptr->data)[i + 1];
            }

            for(int a_col = 0; a_col < K; a_col += K_BLOCK_SIZE)
            {
                int nnz = 0;
                for(int i = M_start; i < M_end; i++)
                {
                    const int row_start = my_cur_col_id[(i - M_start) * 2];
                    const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                    for(int eid = row_start; eid < row_end; eid++)
                    {
                        const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                        if(dst >= a_col + K_BLOCK_SIZE)
                        {
                            break;
                        }
                        my_nz_cells[nnz * 3] = i;
                        my_nz_cells[nnz * 3 + 1] = dst;
                        my_nz_cells[nnz * 3 + 2] = eid;
                        my_cur_col_id[(i - M_start) * 2] = eid + 1;
                        nnz++;
                    }
                }
#if SORT
                //printf("nnz = %d\n", nnz);
                // Radix sort nz_cells according to dst
                // memset(count, 0, K_BLOCK_SIZE * sizeof(int16_t));
                memset(count, 0, K_BLOCK_SIZE * sizeof(int32_t));
                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 3 + 1] & K_BLOCK_MASK;
                    count[v]++;
                }
                int cumulative_sum = 0;
                for(int t = 0; t < K_BLOCK_SIZE; t++)
                {
                    int c = count[t];
                    count[t] = cumulative_sum;
                    cumulative_sum += c;
                }

                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 3 + 1] & K_BLOCK_MASK;
                    int ind = count[v];
                    my_nz_cells_sorted[ind * 3] = my_nz_cells[t * 3];
                    my_nz_cells_sorted[ind * 3 + 1] = my_nz_cells[t * 3 + 1];
                    my_nz_cells_sorted[ind * 3 + 2] = my_nz_cells[t * 3 + 2];
                    count[v]++;
                }
#endif

                for(int t = 0; t < nnz; t++)
                {
#if SORT
                    int src = my_nz_cells_sorted[t * 3];
                    int dst = my_nz_cells_sorted[t * 3 + 1];
                    int eg = my_nz_cells_sorted[t * 3 + 2];
#else
                    int src = my_nz_cells[t * 3];
                    int dst = my_nz_cells[t * 3 + 1];
                    int eg = my_nz_cells[t * 3 + 2];
#endif

#if 1
                    int j;
#pragma unroll(16)
                    for(j = N_block_start; j < N_block_end - 16; j += 16)
                    {
                        //__m512 c512 = _mm512_loadu_ps(&C[i * NE + j]);
                        __m512 bl512 = _mm512_loadu_ps(&BL[src * N + j]);
                        __m512 br512 = _mm512_loadu_ps(&BR[dst * N + j]);
                        __m512 c512 = _mm512_add_ps(bl512, br512);
                        _mm512_storeu_ps(&C[eg * N + j], c512);
                    }
                    for(; j < N_block_end; j++)
                    {
                        C[eg * N + j] = BL[src * N + j] + BR[dst * N + j];
                    }
#else
#pragma omp simd
                    for(int j = N_block_start; j < N_block_end; j++)
                    {
                        // C[i * N + j] += B[dst * N + j];
                        C[eg * NE + j] = BL[src * N + j] + BR[dst * N + j];
                    }
#endif
                }
            }
        }
        _mm_free(count);
        _mm_free(my_cur_col_id);
        _mm_free(my_nz_cells);
        _mm_free(my_nz_cells_sorted);
    }
    
    #undef K_BLOCK_SIZE
    #undef K_BLOCK_MASK
    #undef N_BLOCK_SIZE
    #undef SORT    
}

// BR: u_add_v_copy_e
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectDst,
                      BinaryAdd<float>, ReduceNone<kDLCPU, float> >(
                          const RuntimeConfig& rtcfg,
                          const CSRWrapper& graph,
                          GData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        std::cout << "If (BR)  GAT CallBinaryReduce_Float_Sum In\n";
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectDst,
                                      BinaryAdd<float>, ReduceNone<kDLCPU, float> >
            (rtcfg, graph, gdata);
	}
    else if ( gdata->out_mapping == nullptr || gdata->data_len != 1)
    {
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectDst,
                                      BinaryAdd<float>, ReduceNone<kDLCPU, float> >
            (rtcfg, graph, gdata);        
    }
    else {
        #if DTRACE
        std::cout << "CallBinaryReduce: 0,1, SUM, NONE\n";
        #endif
        
        auto csr = graph.GetOutCSRMatrix();
        const int M = csr.num_rows;
        // const int N = gdata->out_len;
        // const int K = csr.num_cols;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);
        
        // Not using it here.
        // My observation is that native edge numbering and mapping
        // produce same edge id when we use GetOutCSRMatrix()
        if( gdata->out_mapping == nullptr) {
            std::cout << "Test OUT Mapping...\n\n";
            exit(EXIT_FAILURE);
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }
        
        sparse_mm_u_add_v_e(rtcfg, csr, gdata->lhs_data, gdata->rhs_data, gdata->out_data,
                            gdata->x_length);
    }
}

// Backpass: GradLhs
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectSrc, SelectDst,
                              BinaryAdd<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "ELSE Lhs CallBackwardBinaryReduce 0,1, ADD, NONE\n";
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                              SelectSrc, SelectDst,
                                              BinaryAdd<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
        #if DTRACE
        std::cout << "ELSE LHS CallBackwardBinaryReduce 0,1, ADD, NONE\n";
        #endif

        auto csr = graph.GetOutCSRMatrix(); 
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        // Not using it here. Verify once.
        // My observation is that native edge numbering and mapping
        // produce same edge id when we use GetOutCSRMatrix()   
        if( gdata->out_mapping == nullptr) {  // edge mapping
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }
    
        int32_t l_restore = static_cast<int32_t*>(csr.indptr->data)[M];
        int32_t *restore = (int32_t*)malloc(l_restore * sizeof(int32_t));

        int K = csr.num_cols;
        int maxval = 0;  // not private to the omp threads below
        // #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                restore[eid] = static_cast<int32_t*>(csr.indices->data)[eid];
                int eid_ = gdata->out_mapping[eid];
                static_cast<int32_t*>(csr.indices->data)[eid] = eid_;
                if (maxval < eid_) {
                    maxval = eid_;
                }
            }
        }
        
        // sparse_mm3 is affected by this while mkl is not.
        csr.num_cols = maxval + 1;
        cpu::sparse_mkl(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
                        gdata->x_length);

        // restore
        csr.num_cols = K;        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                static_cast<int32_t*>(csr.indices->data)[eid] = restore[eid];
            }
        }
        free(restore);
    }
}

// Backward: GradRhs
// Todo: test it, validate it
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                              SelectSrc, SelectDst,
                              BinaryAdd<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

    const int64_t len = gdata->data_len;
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping || len != 1) {
        #if DTRACE
        std::cout << "ELSE Rhs CallBackwardBinaryReduce 0,1, ADD, NONE\n";
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                                              SelectSrc, SelectDst,
                                              BinaryAdd<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
        #if DTRACE
        std::cout << "ELSE Rhs CallBackwardBinaryReduce 0,1, ADD, NONE\n";
        #endif
        
        auto csr = graph.GetInCSRMatrix(); 
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        if( gdata->out_mapping == nullptr) {
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }
    
        int32_t l_restore = static_cast<int32_t*>(csr.indptr->data)[M];
        int32_t *restore = (int32_t*)malloc(l_restore * sizeof(int32_t));

        int K = csr.num_cols;
        int maxval = 0;  // not private to the omp threads below
        // #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                restore[eid] = static_cast<int32_t*>(csr.indices->data)[eid];
                int eid_ = gdata->out_mapping[eid];
                static_cast<int32_t*>(csr.indices->data)[eid] = eid_;
                if (maxval < eid_) {
                    maxval = eid_;
                }
            }
        }                
        // our matmul is affected by this while mkl is not.
        csr.num_cols = maxval + 1;
        cpu::sparse_mkl(rtcfg, csr, gdata->grad_out_data, gdata->grad_rhs_data,
                        gdata->x_length);

        // restore
        csr.num_cols = K;        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                static_cast<int32_t*>(csr.indices->data)[eid] = restore[eid];
            }
        }
        free(restore);
    }
}

__m512 subv512(__m512 a, __m512 b) {
    return (_mm512_sub_ps(a, b));
}
float subs(float a, float b) {
    return (a-b);
}
__m512 mulv512(__m512 a, __m512 b) {
    return (_mm512_mul_ps(a, b));
}
float muls(float a, float b) {
    return (a*b);
}
__m512 divv512(__m512 a, __m512 b) {
    return (_mm512_div_ps(a, b));
}
float divs(float a, float b) {
    return (a/b);
}


// Notes: Re-purposed sparse_mm3 for BR: e_bop_v_copy_e
template <typename DType>
void sparse_mm_e_op_v_e(const RuntimeConfig &rtcfg,
                         const aten::CSRMatrix& csr,
                         DType* BL, DType* BR, DType* C,
                        int x_length,
                        __m512 (*f_vec)(__m512, __m512),
                        float (*f)(float, float))

{
    #define SORT 0
    #define K_BLOCK_SIZE 20480 // 1024 //4096
    #define K_BLOCK_MASK 1023 //4095
    #define N_BLOCK_SIZE 640

	const int M = csr.num_rows;
	const int N = x_length;
	const int K = csr.num_cols;
    
    int NE = static_cast<int32_t*>(csr.indptr->data)[M+1];
    int nthreads = omp_get_max_threads();
    
	int perThreadQuota = (M + nthreads - 1) / nthreads;
#pragma omp parallel num_threads(nthreads)
    {
        int *my_cur_col_id = (int *)_mm_malloc(2 * perThreadQuota * sizeof(int), 64);
         
        int tid = omp_get_thread_num();		
		
        int M_start = tid * perThreadQuota; 
        int M_end = (tid + 1) * perThreadQuota; 
        if(M_end > M) M_end = M;
        
        // int16_t count[K_BLOCK_SIZE];        
        // int16_t *count = (int16_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int16_t), 64);
        
        int32_t *count = (int32_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int32_t), 64);
        for(int i = M_start; i < M_end; i++)
        {
            my_cur_col_id[(i - M_start) * 2] = static_cast<int32_t*>(csr.indptr->data)[i];
            my_cur_col_id[(i - M_start) * 2 + 1] = static_cast<int32_t*>(csr.indptr->data)[i + 1];
        }

        int max_nnz = 0;
        for(int a_col = 0; a_col < K; a_col += K_BLOCK_SIZE)
        {
            int nnz = 0;
            //printf("%d] a_col = %d, nnz = %ld\n", tid, a_col, nnz);
            for(int i = M_start; i < M_end; i++)
            {
                const int row_start = my_cur_col_id[(i - M_start) * 2];
                const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                for(int eid = row_start; eid < row_end; eid++)
                {
                    const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                    if(dst >= a_col + K_BLOCK_SIZE)
                    {
                        my_cur_col_id[(i - M_start) * 2] = eid;
                        break;
                    }
                    nnz++;
                }
            }
            if(nnz > max_nnz) max_nnz = nnz;
            //printf("%d] a_col = %d, nnz = %ld\n", tid, a_col, nnz);
        }
        //printf("%d] max_nnz = %ld\n", tid, max_nnz);
        int max_nnz_aligned = ((max_nnz + 15) / 16) * 16;
        //int my_nz_cells[max_nnz_aligned];
        //int my_nz_cells_sorted[max_nnz_aligned];
        int *my_nz_cells = (int *)_mm_malloc(2 * max_nnz_aligned * sizeof(int), 64);
        int *my_nz_cells_sorted = (int *)_mm_malloc(2 * max_nnz_aligned * sizeof(int), 64);

        for(int N_block_start = 0; N_block_start < N; N_block_start += N_BLOCK_SIZE)
        {
            int N_block_end = N_block_start + N_BLOCK_SIZE;
            if(N_block_end > N) N_block_end = N;
            for(int i = M_start; i < M_end; i++)
            {
                my_cur_col_id[(i - M_start) * 2] = static_cast<int32_t*>(csr.indptr->data)[i];
                my_cur_col_id[(i - M_start) * 2 + 1] = static_cast<int32_t*>(csr.indptr->data)[i + 1];
            }

            for(int a_col = 0; a_col < K; a_col += K_BLOCK_SIZE)
            {
                int nnz = 0;
                for(int i = M_start; i < M_end; i++)
                {
                    const int row_start = my_cur_col_id[(i - M_start) * 2];
                    const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                    for(int eid = row_start; eid < row_end; eid++)
                    {
                        const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                        if(dst >= a_col + K_BLOCK_SIZE)
                        {
                            break;
                        }
                        my_nz_cells[nnz * 2] = eid;
                        my_nz_cells[nnz * 2 + 1] = dst;
                        // my_nz_cells[nnz * 3 + 2] = eid;
                        my_cur_col_id[(i - M_start) * 2] = eid + 1;
                        nnz++;
                    }
                }
#if SORT
                //printf("nnz = %d\n", nnz);
                // Radix sort nz_cells according to dst
                memset(count, 0, K_BLOCK_SIZE * sizeof(int32_t));
                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 2 + 1] & K_BLOCK_MASK;
                    count[v]++;
                }
                int cumulative_sum = 0;
                for(int t = 0; t < K_BLOCK_SIZE; t++)
                {
                    int c = count[t];
                    count[t] = cumulative_sum;
                    cumulative_sum += c;
                }

                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 2 + 1] & K_BLOCK_MASK;
                    int ind = count[v];
                    my_nz_cells_sorted[ind * 2] = my_nz_cells[t * 3];
                    my_nz_cells_sorted[ind * 2 + 1] = my_nz_cells[t * 3 + 1];
                    // my_nz_cells_sorted[ind * 3 + 2] = my_nz_cells[t * 3 + 2];
                    count[v]++;
                }
#endif

                for(int t = 0; t < nnz; t++)
                {
#if SORT
                    int src = my_nz_cells_sorted[t * 2];
                    int dst = my_nz_cells_sorted[t * 2 + 1];
                    // int eg = my_nz_cells_sorted[t * 3 + 2];
#else
                    int eid = my_nz_cells[t * 2];
                    int dst = my_nz_cells[t * 2 + 1];
                    // int eg = my_nz_cells[t * 3 + 2];
#endif

#if 1
                    int j;
#pragma unroll(16)
                    for(j = N_block_start; j < N_block_end - 16; j += 16)
                    {
                        //__m512 c512 = _mm512_loadu_ps(&C[i * NE + j]);
                        __m512 bl512 = _mm512_loadu_ps(&BL[eid * N + j]);
                        __m512 br512 = _mm512_loadu_ps(&BR[dst * N + j]);
                        // __m512 c512 = _mm512_sub_ps(bl512, br512);
                        __m512 c512 = f_vec(bl512, br512);
                        _mm512_storeu_ps(&C[eid * N + j], c512);
                    }
                    for(; j < N_block_end; j++)
                    {
                        //C[eid * N + j] = BL[eid * N + j] - BR[dst * N + j];
                        C[eid * N + j] = f(BL[eid * N + j], BR[dst * N + j]);
                    }
#else
#pragma omp simd
                    for(int j = N_block_start; j < N_block_end; j++)
                    {
                        // C[eid * NE + j] = BL[eid * N + j] - BR[dst * N + j];
                        C[eid * NE + j] = f(BL[eid * N + j], BR[dst * N + j]);
                    }
#endif
                }
            }
        }
        _mm_free(count);
        _mm_free(my_cur_col_id);
        _mm_free(my_nz_cells);
        _mm_free(my_nz_cells_sorted);
    }
    
    #undef K_BLOCK_SIZE
    #undef K_BLOCK_MASK
    #undef N_BLOCK_SIZE
    #undef SORT    
}

// BR: e_sub_v_copy_e
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectEdge, SelectDst,
                      BinarySub<float>, ReduceNone<kDLCPU, float> >(
                          const RuntimeConfig& rtcfg,
                          const CSRWrapper& graph,
                          GData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "If GAT CallBinaryReduce_Float_Sum In\n";
        #endif
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectEdge, SelectDst,
                                      BinarySub<float>, ReduceNone<kDLCPU, float> >
            (rtcfg, graph, gdata);
	} else {
        
        auto csr = graph.GetOutCSRMatrix();
        const int M = csr.num_rows;
        // const int N = gdata->out_len;
        // const int K = csr.num_cols;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        auto f_vec = [](__m512 a, __m512 b) {
            __m512 c512 = _mm512_sub_ps(a, b);
            return c512;
        };
        auto f = [](float a, float b) {
            return (a - b);
        };

        #if DTRACE
        fprintf(stderr, "CallBinaryReduce: 2,1, SUB, NONE\n");
        #endif
        sparse_mm_e_op_v_e(rtcfg, csr, gdata->lhs_data, gdata->rhs_data, gdata->out_data,
                           gdata->x_length, subv512, subs);
    }
}

// Backpass: GradLhs, e_sub_v_copy_e
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectEdge, SelectDst,
                              BinarySub<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        fprintf(stderr, "IF CallBackwardBinaryReduce FLOAT SUM\n");
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                              SelectEdge, SelectDst,
                                              BinarySub<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
        
        auto csr = graph.GetOutCSRMatrix(); 
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        // Not using it here. Verify once.
        // My observation is that native edge numbering and mapping
        // produce same edge id when we use GetOutCSRMatrix()   
        if( gdata->out_mapping == nullptr) {  // edge mapping
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }

        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                int eid_ = gdata->out_mapping[eid];
                float *grad_lhs = gdata->grad_lhs_data + D * eid_;
                float *grad_out = gdata->grad_out_data + D * eid_;
                #pragma omp simd
                for (auto tx=0; tx<D; tx++)
                    grad_lhs[tx] += grad_out[tx];
            }
        }                
    }
}

// Backpass: GradRhs, e_sub_v_copy_e
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                              SelectEdge, SelectDst,
                              BinarySub<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        fprintf(stderr, "IF CallBackwardBinaryReduce FLOAT SUM\n");
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                                              SelectEdge, SelectDst,
                                              BinarySub<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
            
        auto csr = graph.GetInCSRMatrix(); 
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        if( gdata->out_mapping == nullptr) {  // edge mapping
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }
        int32_t l_restore = static_cast<int32_t*>(csr.indptr->data)[M];
        int32_t *restore = (int32_t*)malloc(l_restore * sizeof(int32_t));

        int K = csr.num_cols;
        int maxval = 0;  // not private to the omp threads below
        // #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                restore[eid] = static_cast<int32_t*>(csr.indices->data)[eid];
                int eid_ = gdata->out_mapping[eid];
                static_cast<int32_t*>(csr.indices->data)[eid] = eid_;
                if (maxval < eid_) {
                    maxval = eid_;
                }
            }
        }                

        csr.num_cols = maxval + 1;
        cpu::sparse_mm3_gen(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data, /*lhs data due to incsr*/
                            gdata->x_length, subv512, subs);

        // restore
        csr.num_cols = K;        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                static_cast<int32_t*>(csr.indices->data)[eid] = restore[eid];
            }
        }
        free(restore);
    }
}



// BR: e_div_v_copy_e
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectEdge, SelectDst,
                      BinaryDiv<float>, ReduceNone<kDLCPU, float> >(
                          const RuntimeConfig& rtcfg,
                          const CSRWrapper& graph,
                          GData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "If GAT CallBinaryReduce_Float_Sum In\n";
        #endif
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectEdge, SelectDst,
                                      BinaryDiv<float>, ReduceNone<kDLCPU, float> >
            (rtcfg, graph, gdata);
	} else {
        #if DTRACE
        fprintf(stderr, "CallBinaryReduce: 2,1, DIV, NONE\n");
        #endif
        auto csr = graph.GetOutCSRMatrix();
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);
    
        auto f_vec = [](__m512 a, __m512 b) {
            __m512 c512 = _mm512_div_ps(a, b);
            return c512;
        };
        auto f = [](float a, float b) {
            return (a/b);
        };

        sparse_mm_e_op_v_e(rtcfg, csr, gdata->lhs_data, gdata->rhs_data, gdata->out_data,
                           gdata->x_length, divv512, divs);
    }
}

// Backpass: GradLhs, e_div_v_copy_e
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectEdge, SelectDst,
                              BinaryDiv<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        fprintf(stderr, "IF CallBackwardBinaryReduce FLOAT SUM\n");
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                              SelectEdge, SelectDst,
                                              BinaryDiv<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
        
        auto csr = graph.GetOutCSRMatrix(); 
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        // Not using it here. Verify once.
        // My observation is that native edge numbering and mapping
        // produce same edge id when we use GetOutCSRMatrix()   
        if( gdata->out_mapping == nullptr) {  // edge mapping
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }
        sparse_mm_e_op_v_e(rtcfg, csr, gdata->grad_out_data, gdata->rhs_data, gdata->grad_lhs_data,
                           gdata->x_length, divv512, divs);
    }
}

// Backpass: GradRhs,e_div_v_copy_e
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                              SelectEdge, SelectDst,
                              BinaryDiv<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        fprintf(stderr, "IF CallBackwardBinaryReduce FLOAT SUM\n");
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                                              SelectEdge, SelectDst,
                                              BinaryDiv<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
        
        auto csr = graph.GetOutCSRMatrix(); 
        const int M = csr.num_rows;
        const int N = csr.num_cols;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        // Not using it here. Verify once.
        // My observation is that native edge numbering and mapping
        // produce same edge id when we use GetOutCSRMatrix()   
        if( gdata->out_mapping == nullptr) {  // edge mapping
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }

        int32_t nedges = static_cast<int32_t*>(csr.indptr->data)[M];
        float *intrim_res_a = (float*)_mm_malloc(nedges * sizeof(float), 64);
        float *intrim_res_b = (float*)_mm_malloc(N * sizeof(float), 64);
    
        // Phase I/III:
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                float *lhsoff = gdata->lhs_data + D * eid;
                float *grad_out = gdata->grad_out_data + D * eid;
                float *intrim_res_a_ptr = intrim_res_a + D * eid;
                #pragma omp simd
                for (auto tx=0; tx<D; tx++)
                    intrim_res_a_ptr[tx] = grad_out[tx] * lhsoff[tx] * (-1);
            }
        }                

        // Phase II/III:
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                const int dst = static_cast<int32_t*>(csr.indices->data)[eid];
                float *rhsoff = gdata->rhs_data + D * dst;
                float *intrim_res_b_ptr = intrim_res_b + D * dst;
                #pragma omp simd
                for (auto tx=0; tx<D; tx++)
                    intrim_res_b_ptr[tx] = rhsoff[tx] * rhsoff[tx];
            }
        }                

        // Phase III/III:
        sparse_mm_e_op_v_e(rtcfg, csr, intrim_res_a, intrim_res_b, gdata->grad_rhs_data,
                           gdata->x_length, divv512, divs);

        // release memory
        _mm_free(intrim_res_a);
        _mm_free(intrim_res_b);
    }
}

// BR: v_mul_e_copy_e
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectDst, SelectEdge,
                      BinaryMul<float>, ReduceNone<kDLCPU, float> >(
                          const RuntimeConfig& rtcfg,
                          const CSRWrapper& graph,
                          GData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        std::cout << "If GAT CallBinaryReduce_Float_Sum In\n";
        #endif
		cpu::FallbackCallBinaryReduce<kDLCPU, int32_t, float, SelectDst, SelectEdge,
                                      BinaryMul<float>, ReduceNone<kDLCPU, float> >
            (rtcfg, graph, gdata);
	} else {
        #if DTRACE
        fprintf(stderr, "CallBinaryReduce: 1,2, SUB, NONE\n");
        #endif
        
        auto csr = graph.GetOutCSRMatrix();
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);
        auto f_vec = [](__m512 a, __m512 b) {
            __m512 c512 = _mm512_mul_ps(a, b);
            return c512;
        };
        auto f = [](float a, float b) {
            return (a * b);
        };

        // switchin lhs and rhs for mul
        sparse_mm_e_op_v_e(rtcfg, csr, gdata->rhs_data, gdata->lhs_data, gdata->out_data,
                           gdata->x_length, mulv512, muls);
    }
}

// Backpass: GradLhs, v_mul_e_copy_e
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectDst, SelectEdge,
                              BinaryMul<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        fprintf(stderr, "IF CallBackwardBinaryReduce FLOAT SUM\n");
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                                              SelectDst, SelectEdge,
                                              BinaryMul<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {
    
        auto csr = graph.GetInCSRMatrix(); 
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        if( gdata->out_mapping == nullptr) {  // edge mapping
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }

        int32_t l_restore = static_cast<int32_t*>(csr.indptr->data)[M];
        int32_t *restore = (int32_t*)malloc(l_restore * sizeof(int32_t));
        // Phase I:
        float *intrim_res = (float*)_mm_malloc(l_restore * sizeof(float), 64);
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                int eid_ = gdata->out_mapping[eid];
                float *rhsoff = gdata->rhs_data + D * eid_;
                float *grad_out = gdata->grad_out_data + D * eid_;
                float *intrim_res_ptr = intrim_res + D * eid_;
                #pragma omp simd
                for (auto tx=0; tx<D; tx++)
                    intrim_res_ptr[tx] += grad_out[tx] * rhsoff[tx];
            }
        }                
    
        // Phase II:
        int K = csr.num_cols;
        int maxval = 0;  // not private to the omp threads below
        // #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                restore[eid] = static_cast<int32_t*>(csr.indices->data)[eid];
                int eid_ = gdata->out_mapping[eid];
                static_cast<int32_t*>(csr.indices->data)[eid] = eid_;
                if (maxval < eid_) {
                    maxval = eid_;
                }
            }
        }                

        csr.num_cols = maxval + 1;
        cpu::sparse_mm3_gen(rtcfg, csr, intrim_res, gdata->grad_lhs_data, /*lhs data due to incsr*/
                            gdata->x_length, subv512, subs);
        
        csr.num_cols = K;        
        #pragma omp parallel for
        for (int src=0; src<M; src++)
        {
            int32_t row_start = static_cast<int32_t*>(csr.indptr->data)[src];
            int32_t row_end = static_cast<int32_t*>(csr.indptr->data)[src + 1];
        
            for (int eid=row_start; eid<row_end; eid++)
            {
                static_cast<int32_t*>(csr.indices->data)[eid] = restore[eid];
            }
        }
        free(restore);
        free(intrim_res);
    }
}

// Backpass: GradRhs, v_mul_e_copy_e
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                              SelectDst, SelectEdge,
                              BinaryMul<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        #if DTRACE
        fprintf(stderr, "IF CallBackwardBinaryReduce FLOAT SUM\n");
        #endif
		cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                                              SelectDst, SelectEdge,
                                              BinaryMul<float>, ReduceNone<kDLCPU, float>>
            (rtcfg, graph, gdata);
	} else {   
    
        auto csr = graph.GetInCSRMatrix(); 
        const int M = csr.num_rows;
        const int64_t len = gdata->data_len;
        const int64_t D = gdata->x_length;
        assert(len == 1);

        if( gdata->out_mapping == nullptr) {  // edge mapping
            gdata->out_mapping = static_cast<int*>(csr.data->data);
        }

        sparse_mm_e_op_v_e(rtcfg, csr, gdata->grad_out_data, gdata->lhs_data, gdata->grad_rhs_data,
                           gdata->x_length, mulv512, muls);
    }
}


#define REDUCER ReduceSum
#define XPU kDLCPU
#define IDX int32_t

// generate definitions	
EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE);
#undef IDX

#define IDX int64_t
EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE);
#undef IDX

}  // namespace kernel
}  // namespace dgl

#endif
