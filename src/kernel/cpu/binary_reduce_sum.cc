/*** 
The code under BASE is the original DGL code, by default BASE is 0.
PCL optimized code is under #else .. #endif portion.
Authors: Md. Vasimuddin (PCL, Intel), Sanchit Misra (PCL, Intel)

Notes: 
***/
#if BASE  
/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_sum.cc
 * \brief CPU kernels for binary reduce sum
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

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
#include <stdlib.h>

#define MKL_ENABLED 1
#if MKL_ENABLED
#include <mkl_spblas.h>
#include <mkl.h>
#endif

#include <omp.h>
#include <sys/syscall.h>
#include <x86intrin.h>
#include <immintrin.h>
#define gettid() ((int)syscall(SYS_gettid))

// kmp_affinity_mask_t mask;

using minigun::advance::RuntimeConfig;

namespace dgl {
namespace kernel {
namespace cpu {

// Original matmul code from Sanchit
template <typename DType>
void sparse_mm3_int16(const RuntimeConfig &rtcfg,
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
    // fprintf(stderr, "nthreads: %d, M: %d, K: %d, N: %d\n", nthreads, M, K, N);

	int perThreadQuota = (M + nthreads - 1) / nthreads;
#pragma omp parallel num_threads(nthreads)
    {
        int *my_cur_col_id = (int *)_mm_malloc(2 * perThreadQuota * sizeof(int), 64);
        //int64_t tst = __rdtsc();
         
        int tid = omp_get_thread_num();		
		
        int M_start = tid * perThreadQuota; 
        int M_end = (tid + 1) * perThreadQuota; 
        if(M_end > M) M_end = M;
        //int16_t count[K_BLOCK_SIZE];
        int16_t *count = (int16_t *)_mm_malloc(K_BLOCK_SIZE * sizeof(int16_t), 64);
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
        // printf("%d] max_nnz = %ld\n", tid, max_nnz);
        
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
                // printf("SORT nnz = %d\n", nnz);
                // Radix sort nz_cells according to dst
                memset(count, 0, K_BLOCK_SIZE * sizeof(int16_t));
                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 2 + 1] & K_BLOCK_MASK;
                    count[v]++;
                    assert(count[v] > 0);
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
#endif
                }
            }
        }
        _mm_free(count);
        _mm_free(my_cur_col_id);
        _mm_free(my_nz_cells);
        _mm_free(my_nz_cells_sorted);
    }		
}


// Vasim: sparse_mm3 with int32
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
        //int64_t tst = __rdtsc();
         
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
        // printf("%d] max_nnz = %ld\n", tid, max_nnz);
        
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
                // printf("SORT nnz = %d\n", nnz);
                // Radix sort nz_cells according to dst
                // memset(count, 0, K_BLOCK_SIZE * sizeof(int16_t));
                memset(count, 0, K_BLOCK_SIZE * sizeof(int32_t));
                for(int t = 0; t < nnz; t++)
                {
                    int v = my_nz_cells[t * 2 + 1] & K_BLOCK_MASK;
                    count[v]++;
                    assert(count[v] > 0);
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
#endif
                }
            }
        }
        _mm_free(count);
        _mm_free(my_cur_col_id);
        _mm_free(my_nz_cells);
        _mm_free(my_nz_cells_sorted);
        //int64_t tet = __rdtsc();
        //printf("%d] %ld ticks\n", tid, tet - tst);
    }		
}
			

#if MKL_ENABLED
template <typename DType>
void sparse_mkl(const RuntimeConfig &rtcfg,
                const aten::CSRMatrix& csr,
                DType* B, DType* C,
                int x_length)

{
	const int M = csr.num_rows;
	const int N = x_length;
	// const int K = csr.num_cols;
	const int nnz = csr.indices->shape[0];
	const DType alpha = 1.0;
	const DType beta = 0.0;
	
	// float alpha = 1.0f;
	// float beta = 0.0f;
	DType* valptr = (DType*) malloc(nnz * sizeof(DType));
	// for (int i=0; i<nnz; i++) valptr[i] = 1.0;
	utils::Fill<kDLCPU>(rtcfg.ctx, valptr, nnz, static_cast<DType>(1.));
	
	matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	
	sparse_matrix_t csr_mat_t; // = &csr_mat;
	// mkl_sparse_s_create_csr(&csr_mat_t, SPARSE_INDEX_BASE_ZERO, csr_mat.rows, csr_mat.cols,
	//						csr_mat.rowPtr, csr_mat.rowPtr + 1, csr_mat.indices, csr_mat.values);
	// int64_t tim = _rdtsc();
	mkl_sparse_s_create_csr(&csr_mat_t, SPARSE_INDEX_BASE_ZERO, M, N,
							static_cast<int32_t*>(csr.indptr->data),
							static_cast<int32_t*>(csr.indptr->data) + 1,
							static_cast<int32_t*>(csr.indices->data),
							valptr);
	
	//fprintf(stderr, "CSR creation Time: M: %d, K: %d, N: %d, nnz: %d,  %ld\n",
	// 		M, K, N, nnz, _rdtsc() - tim);
	//int max_threads = mkl_get_max_threads();
	//printf("Available max MKL threads: %d\n", max_threads);

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
#endif

// forward
template <typename DType>
void FallbackCallBinaryReduce(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLCPU;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceSum<kDLCPU, DType> Reducer;
  typedef cpu::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cpu::BinaryReduce<Idx, DType, Functors> UDF;
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

template <typename DType>
void FallbackCallBackwardBinaryReduce(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLCPU;
  constexpr int Mode = binary_op::kGradLhs;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceSum<kDLCPU, DType> Reducer;
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

}  // namespace cpu


/*************************** binary_reduce<float/double>_unary ***************************/
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectNone,
                      BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, float>* gdata) {
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBinaryReduce<float>(rtcfg, graph, gdata);
	} else {
		// cusparse use rev csr for csrmm
		auto csr = graph.GetInCSRMatrix();
        #if MKL_ENABLED
        cpu::sparse_mkl(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);
        #else
        cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);            
        #endif               
	}
}

template <>
void CallBinaryReduce<kDLCPU, int32_t, double, SelectSrc, SelectNone,
                      BinaryUseLhs<double>, ReduceSum<kDLCPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, double>* gdata) {
	fprintf(stderr, "In CallBinaryReduce_Double_Sum\n");
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBinaryReduce<double>(rtcfg, graph, gdata);
	} else {
		// cusparse use rev csr for csrmm
		auto csr = graph.GetInCSRMatrix();
        // #if MKL_ENABLED
        // cpu::sparse_mkl(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        //                       gdata->x_length);
        // #else
        // cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        //                       gdata->x_length);
        // #endif
        cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);        
	}
}


// backward
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBackwardBinaryReduce<float>(rtcfg, graph, gdata);
	} else {
		auto csr = graph.GetOutCSRMatrix();
        #if MKL_ENABLED
        cpu::sparse_mkl(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
                        gdata->x_length);
        #else
		cpu::sparse_mm3(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
                        gdata->x_length);
        #endif		
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
        cpu::FallbackCallBackwardBinaryReduce<double>(rtcfg, graph, gdata);
    } else {
        auto csr = graph.GetOutCSRMatrix();
        // #if MKL_ENABLED
        // cpu::sparse_mkl(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        //                 gdata->x_length);
        // #else
        // cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        //                 gdata->x_length);
        // #endif
        cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);

    }
}


/*************************** binary_reduce_bcast<float/double>_binary ***************************/
#if 1 //!BASE
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

template<typename T>
void bcast_dgl(const aten::CSRMatrix& csr, T* gdata,
               float* out_data_r, float* out_data_l) // mul(v,e) -> v
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
            if (gdata->out_len > gdata->rhs_len)
            {
                assert(gdata->lhs_len == gdata->out_len);

                float* rhs_data = gdata->rhs_data + eid * gdata->rhs_len * len;
                float* out_data = out_data_r + src * gdata->out_len *len;

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
            
            if (gdata->out_len > gdata->lhs_len)
            {
                assert(gdata->rhs_len == gdata->out_len);
                float* lhs_data = gdata->lhs_data + dst * gdata->lhs_len * len;  // data with len size
                float* out_data = out_data_l + src * gdata->out_len *len; 

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

template <>
void CallBinaryReduceBcast<kDLCPU, NDIM4, int, float, SelectSrc, SelectEdge,
                           BinaryMul<float>, ReduceSum<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BcastGData<NDIM4, int, float>* gdata) {
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		// cpu::FallbackCallBinaryReduce<float>(rtcfg, graph, gdata);
        fprintf(stderr, "If part not defined\n");
	}
    else {
		// cusparse use rev csr for csrmm
		auto csr = graph.GetInCSRMatrix();
        // step 1: bcast
        float *out_data_r, *out_data_l, *xhs_data = NULL;
        int64_t nlength = gdata->out_len * csr.num_cols;

        bool loc[2] = {0};
        if (gdata->out_len > gdata->rhs_len) {
            out_data_r= (float*)malloc(nlength * sizeof(float));
            xhs_data = out_data_r;
            loc[0] = 1;
        }
        else
            out_data_r = gdata->rhs_data;

        if (gdata->out_len > gdata->lhs_len) {
            out_data_l= (float*) malloc(nlength * sizeof(float));
            xhs_data = out_data_l;
            loc[1] = 1;
        }
        else
            out_data_l = gdata->lhs_data;
        
        bcast_dgl<BcastGData<NDIM4, int32_t, float>>(csr, gdata, out_data_r, out_data_l);
        
        // step 2: Multiply
        #define SIMD_WIDTH16 16

        // multi thread this
        int i = 0;
        #pragma omp parallel for lastprivate(i)
        for (i=0; i<nlength - SIMD_WIDTH16; i+= SIMD_WIDTH16) {
            __m512 lhs = _mm512_loadu_ps(out_data_l + i);
            __m512 rhs = _mm512_loadu_ps(out_data_r + i);
            lhs =  _mm512_mul_ps(lhs, rhs);
            _mm512_storeu_ps((__m512*)(xhs_data + i), lhs);
        }
        for (; i<nlength; i++)
            xhs_data[i] = out_data_l[i] * out_data_r[i];
        
        // step 3: Sparse mm
		{
            #if MKL_ENABLED
			cpu::sparse_mkl(rtcfg, csr, xhs_data, gdata->out_data, gdata->out_len);
            #else
			cpu::sparse_mm3(rtcfg, csr, xhs_data, gdata->out_data, gdata->out_len);
            #endif
		}

        if (loc[0])
            free(out_data_r);
        if (loc[1])
            free(out_data_l);
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
    
    // fprintf(stderr, "In CallBinaryReduceBcast Float_Mul_Sum version (MKL stuff)\n");
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        fprintf(stderr, "Backward: If part not defined, TODO !!!\n");
        assert(0);
	}
    else {
		// cusparse use rev csr for csrmm

		auto csr = graph.GetOutCSRMatrix();
        // step 1: bcast
        float *out_data_r, *out_data_l; //, *xhs_data = NULL;
        int64_t nlength = gdata->out_len * csr.num_cols;

        bool loc[2] = {0};
        if (gdata->out_len > gdata->rhs_len) {
            out_data_r= (float*)_mm_malloc(nlength * sizeof(float), 64);
            // xhs_data = out_data_r;
            loc[0] = 1;
        }
        else
            out_data_r = gdata->rhs_data;

        if (gdata->out_len > gdata->lhs_len) {
            out_data_l= (float*) _mm_malloc(nlength * sizeof(float), 64);
            // xhs_data = out_data_l;
            loc[1] = 1;
        }
        else
            out_data_l = gdata->lhs_data;
        
        bcast_dgl<BackwardBcastGData<NDIM4, int, float>>(csr, gdata, out_data_r, out_data_l);   //fused
        
        // step 2: Multiply
        // float *out_mul_e = (float*) _mm_malloc(nlength * sizeof(float), 64);
        #define SIMD_WIDTH16 16

        // multi thread this
        int i = 0;
        
        float *grad_out = gdata->grad_out_data;
        float *gradlhsoff = gdata->grad_lhs_data;
        // float *outoff   = gdata->out_data;
        
        float *grad_e = grad_out;;

        // multiply lhs with grad
        #pragma omp parallel for lastprivate(i)
        for (i=0; i<nlength - SIMD_WIDTH16; i+= SIMD_WIDTH16)
        {
            __m512 lhs = _mm512_loadu_ps(grad_e + i);
            __m512 rhs = _mm512_loadu_ps(out_data_r + i);
            lhs =  _mm512_mul_ps(lhs, rhs);
            _mm512_storeu_ps((__m512*)(grad_e + i), lhs);
        }
        
        for (; i<nlength; i++) grad_e[i] = grad_e[i] * out_data_r[i];
        
        // step 3: Sparse mm
		{
            #if MKL_ENABLED
            cpu::sparse_mkl(rtcfg, csr, grad_e, gradlhsoff, gdata->out_len);
            #else
            cpu::sparse_mm3(rtcfg, csr, grad_e, gradlhsoff, gdata->out_len);
            #endif
		}

        if (loc[0])
            _mm_free(out_data_r);
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
        fprintf(stderr, "Backwardrhs: If part not defined, TODO !!!\n");
        assert(0);
	}
    else {
		// cusparse use rev csr for csrmm
		
		auto csr = graph.GetOutCSRMatrix();
        // step 1: bcast
        float *out_data_r, *out_data_l; //, *xhs_data = NULL;
        int64_t nlength = gdata->out_len * csr.num_cols;

        bool loc[2] = {0};
        if (gdata->out_len > gdata->rhs_len) {
            out_data_r= (float*)_mm_malloc(nlength * sizeof(float), 64);
            // xhs_data = out_data_r;
            loc[0] = 1;
        }
        else
            out_data_r = gdata->rhs_data;

        if (gdata->out_len > gdata->lhs_len) {
            out_data_l= (float*) _mm_malloc(nlength * sizeof(float), 64);
            // xhs_data = out_data_l;
            loc[1] = 1;
        }
        else
            out_data_l = gdata->lhs_data;
        
        bcast_dgl<BackwardBcastGData<NDIM4, int, float>>(csr, gdata, out_data_r, out_data_l);
        
        // step 2: Multiply
        // float *out_mul_e = (float*) _mm_malloc(nlength * sizeof(float), 64);
        #define SIMD_WIDTH16 16

        // multi thread this
        int i = 0;
        
        float *grad_out = gdata->grad_out_data;
        float *gradrhsoff = gdata->grad_rhs_data;
        
        float *grad_e = grad_out;

        // multiply rhs with grad
        #pragma omp parallel for lastprivate(i)
        for (i=0; i<nlength - SIMD_WIDTH16; i+= SIMD_WIDTH16)
        {
            __m512 lhs = _mm512_loadu_ps(grad_e + i);
            __m512 rhs = _mm512_loadu_ps(out_data_l + i);
            lhs =  _mm512_mul_ps(lhs, rhs);
            _mm512_storeu_ps((__m512*)(grad_e + i), lhs);
        }
        
        for (; i<nlength; i++) grad_e[i] = grad_e[i] * out_data_l[i];

        
        // step 3: Sparse mm
		{
            #if MKL_ENABLED
			cpu::sparse_mkl(rtcfg, csr, grad_e, gradrhsoff, gdata->out_len);
            #else
			cpu::sparse_mm3(rtcfg, csr, grad_e, gradrhsoff, gdata->out_len);
            #endif
		}

        if (loc[0])
            _mm_free(out_data_r);
        if (loc[1])
            _mm_free(out_data_l);
	}
}
#endif


/*************************** binary_reduce_dot<float/double>_binary ***************************/
// binary op, so bcast is calling this function call.
template <>
void CallBinaryReduce<kDLCPU, int32_t, float, SelectSrc, SelectDst,
                      BinaryDot<float>, ReduceNone<kDLCPU, float>>(
                          const RuntimeConfig& rtcfg,
                          const CSRWrapper& graph,
                          GData<int32_t, float>* gdata) {
	
	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBinaryReduce<float>(rtcfg, graph, gdata);
	} else {
		// cusparse use rev csr for csrmm
		
		auto csr = graph.GetInCSRMatrix();
        const int M = csr.num_rows;
        // const int N = gdata->out_len;
        // const int K = csr.num_cols;
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

// Lhs grad
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, float,
                              SelectSrc, SelectDst,
                              BinaryDot<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBackwardBinaryReduce<float>(rtcfg, graph, gdata);
	} else {
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
                // float* outoff = gdata->out_data + oid * D;
                float* gradlhsoff = gdata->grad_lhs_data + lid * D * len;
                // float* gradrhsoff = gdata->grad_rhs_data + rid * D * len;
                float* gradoutoff = gdata->grad_out_data + oid * D;
                for (int64_t tx = 0; tx < D; ++tx)
                {
                    float grad_e = *(gradoutoff + tx);
                    // float* lhs_base = lhsoff + tx * len;
                    // float* rhs_base = rhsoff + tx * len;

                    #pragma omp simd
                    for (int64_t i = 0; i < len; ++i) {
                        gradlhsoff[tx * len + i] += rhsoff[tx * len + i] * grad_e;
                    }
                }                
            }
        }
	}
}

// Rhs grad
template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradRhs, int32_t, float,
                              SelectSrc, SelectDst,
                              BinaryDot<float>, ReduceNone<kDLCPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {

	if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
		cpu::FallbackCallBackwardBinaryReduce<float>(rtcfg, graph, gdata);
	} else {
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
                
                float* lhsoff = gdata->lhs_data + lid * D * len;
                // float* rhsoff = gdata->rhs_data + rid * D * len;
                // float* outoff = gdata->out_data + oid * D;
                // DType* gradlhsoff = gdata->grad_lhs_data + lid * D * len;
                float* gradrhsoff = gdata->grad_rhs_data + rid * D * len;
                float* gradoutoff = gdata->grad_out_data + oid * D;
                for (int64_t tx = 0; tx < D; ++tx)
                {
                    float grad_e = *(gradoutoff + tx);
                    // float* lhs_base = lhsoff + tx * len;
                    // float* rhs_base = rhsoff + tx * len;

                    #pragma omp simd
                    for (int64_t i = 0; i < len; ++i) {
                        gradrhsoff[tx * len + i] += lhsoff[tx * len + i] * grad_e;
                    }
                }

                
            }
        }
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
