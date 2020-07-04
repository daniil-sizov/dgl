/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/backward_binary_reduce_impl.h
 * \brief Minigun CPU UDFs for bacward binary reduce
 */
#ifndef DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_

#include <minigun/minigun.h>

#include "../binary_reduce_impl_decl.h"
#include "../utils.h"
#include "./functor.h"
#include "../csr_interface.h"
#include "macro.h"

namespace dgl {
namespace kernel {
namespace cpu {

template <int Mode, typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduceIntel {
    static inline bool CondEdge(
        Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
        return true;
    }
    static inline void ApplyEdge(
        Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
        const int64_t D = gdata->x_length;
        const int64_t len = gdata->data_len;
        // Idx lid = Functors::SelectLeft(src, eid, dst);
        // Idx rid = Functors::SelectRight(src, eid, dst);
        // Idx oid = Functors::SelectOut(src, eid, dst);
        Idx lid = Functors::SelectLeft(dst, eid, src);
        Idx rid = Functors::SelectRight(dst, eid, src);
        Idx oid = Functors::SelectOut(dst, eid, src);

        if (gdata->lhs_mapping) {
            lid = Functors::GetId(lid, gdata->lhs_mapping);
        }
        if (gdata->rhs_mapping) {
            rid = Functors::GetId(rid, gdata->rhs_mapping);
        }
        if (gdata->out_mapping) {
            oid = Functors::GetId(oid, gdata->out_mapping);
        }
        DType* lhsoff = gdata->lhs_data + lid * D * len;
        DType* rhsoff = gdata->rhs_data + rid * D * len;
        DType* outoff = gdata->out_data + oid * D;
        DType* gradlhsoff = gdata->grad_lhs_data + lid * D * len;
        DType* gradrhsoff = gdata->grad_rhs_data + rid * D * len;
        DType* gradoutoff = gdata->grad_out_data + oid * D;
        // #pragma omp simd
        for (int64_t tx = 0; tx < D; ++tx)
        {
            DType out = Functors::Read(outoff + tx);
            DType grad_out = Functors::Read(gradoutoff + tx);
            DType e = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
            DType grad_e = grad_out * Functors::BackwardWrite(e, out);

            DType* lhs_base = lhsoff + tx * len;
            DType* rhs_base = rhsoff + tx * len;
            if (Mode == binary_op::kGradBoth) {
                for (int64_t i = 0; i < len; ++i) {
                    DType lhs = Functors::Read(lhs_base + i);
                    DType rhs = Functors::Read(rhs_base + i);
                    DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
                    DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
                    DType grad = grad_lhs + grad_rhs;
                    // #pragma omp atomic
                    gradlhsoff[tx * len + i] += grad;
                }
            } else if (Mode == binary_op::kGradLhs) {
                for (int64_t i = 0; i < len; ++i) {
                    DType lhs = Functors::Read(lhs_base + i);
                    DType rhs = Functors::Read(rhs_base + i);
                    DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
                    #pragma omp atomic
                    gradlhsoff[tx * len + i] += grad_lhs;
                }
            } else if (Mode == binary_op::kGradRhs) {
                for (int64_t i = 0; i < len; ++i) {
                    DType lhs = Functors::Read(lhs_base + i);
                    DType rhs = Functors::Read(rhs_base + i);
                    DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
                    #pragma omp atomic
                    gradrhsoff[tx * len + i] += grad_rhs;
                }
            }
        }
    }
};

// Minigun UDF to compute backward binary reduce.
template <int Mode, typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduce {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
    const int64_t D = gdata->x_length;
    const int64_t len = gdata->data_len;
    Idx lid = Functors::SelectLeft(src, eid, dst);
    Idx rid = Functors::SelectRight(src, eid, dst);
    Idx oid = Functors::SelectOut(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }
    DType* lhsoff = gdata->lhs_data + lid * D * len;
    DType* rhsoff = gdata->rhs_data + rid * D * len;
    DType* outoff = gdata->out_data + oid * D;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * D * len;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * D * len;
    DType* gradoutoff = gdata->grad_out_data + oid * D;
    for (int64_t tx = 0; tx < D; ++tx) {
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);
      if (0 == grad_e)
        continue;
      DType* lhs_base = lhsoff + tx * len;
      DType* rhs_base = rhsoff + tx * len;
      if (Mode == binary_op::kGradBoth) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
          DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
          DType grad = grad_lhs + grad_rhs;
#pragma omp atomic
          gradlhsoff[tx * len + i] += grad;
        }
      } else if (Mode == binary_op::kGradLhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
#pragma omp atomic
          gradlhsoff[tx * len + i] += grad_lhs;
        }
      } else if (Mode == binary_op::kGradRhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
#pragma omp atomic
          gradrhsoff[tx * len + i] += grad_rhs;
        }
      }
    }
  }
};

// Minigun UDF to compute backward binary reduce with broadcasting.
template <int Mode, int NDim,
          typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduceBcastIntel {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    const int64_t len = gdata->data_len;
    // Idx lid = Functors::SelectLeft(src, eid, dst);
    // Idx rid = Functors::SelectRight(src, eid, dst);
    // Idx oid = Functors::SelectOut(src, eid, dst);
    Idx lid = Functors::SelectLeft(dst, eid, src);
    Idx rid = Functors::SelectRight(dst, eid, src);
    Idx oid = Functors::SelectOut(dst, eid, src);
    // fprintf(stderr, "%d %d %d - %d %d %d\n", src, eid, dst, lid, rid, oid);
    
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len * len;
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len * len;
    DType* outoff = gdata->out_data + oid * gdata->out_len;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * gdata->out_len * len;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * gdata->out_len * len;
    DType* gradoutoff = gdata->grad_out_data + oid * gdata->out_len;
    int64_t tmp[NDim];  // store unraveled idx.

    for (int64_t tx = 0; tx < gdata->out_len; ++tx)
    {
      Unravel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride, tmp);
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(
        lhsoff + Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride) * len,
        rhsoff + Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride) * len,
        len);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);

      DType* lhs_base = lhsoff +
          Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride) * len;
      DType* rhs_base = rhsoff +
          Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride) * len;
      if (Mode == binary_op::kGradBoth) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
          DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
          DType grad = grad_lhs + grad_rhs;
          
          // #pragma omp atomic          
          gradlhsoff[tx * len + i] += grad;
        }
      } else if (Mode == binary_op::kGradLhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
          
          #pragma omp atomic
          gradlhsoff[tx * len + i] += grad_lhs;
        }
      } else if (Mode == binary_op::kGradRhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
          
          #pragma omp atomic
          gradrhsoff[tx * len + i] += grad_rhs;
        }
      }
    }
  }
};

// Minigun UDF to compute backward binary reduce with broadcasting.
template <int Mode, int NDim,
          typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduceBcast {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    const int64_t len = gdata->data_len;
    Idx lid = Functors::SelectLeft(src, eid, dst);
    Idx rid = Functors::SelectRight(src, eid, dst);
    Idx oid = Functors::SelectOut(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len * len;
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len * len;
    DType* outoff = gdata->out_data + oid * gdata->out_len;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * gdata->out_len * len;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * gdata->out_len * len;
    DType* gradoutoff = gdata->grad_out_data + oid * gdata->out_len;
    int64_t tmp[NDim];  // store unraveled idx.
    for (int64_t tx = 0; tx < gdata->out_len; ++tx) {
      Unravel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride, tmp);
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(
        lhsoff + Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride) * len,
        rhsoff + Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride) * len,
        len);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);
      // (pawelpiotrowicz) Although we can technically add the same condition for
      // skipping atomic additions as in BackwardBinaryReduce, doing so made the
      // speed 2% slower in GCMC training on MovieLens-1M with 24 OpenMP threads.
      // For more details, see https://github.com/dmlc/dgl/pull/1527.
      // TODO(BarclayII): Needs further investigation and benchmarking.

      DType* lhs_base = lhsoff +
          Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride) * len;
      DType* rhs_base = rhsoff +
          Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride) * len;
      if (Mode == binary_op::kGradBoth) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
          DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
          DType grad = grad_lhs + grad_rhs;
#pragma omp atomic
          gradlhsoff[tx * len + i] += grad;
        }
      } else if (Mode == binary_op::kGradLhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
#pragma omp atomic
          gradlhsoff[tx * len + i] += grad_lhs;
        }
      } else if (Mode == binary_op::kGradRhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType lhs = Functors::Read(lhs_base + i);
          DType rhs = Functors::Read(rhs_base + i);
          DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
#pragma omp atomic
          gradrhsoff[tx * len + i] += grad_rhs;
        }
      }
    }
  }
};

// Auxiliary template used in UDF.
template <typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct BackwardFunctorsTempl {
    static inline Idx SelectOut(
        Idx src, Idx edge, Idx dst) {
        typedef typename OutSelector<Reducer>::Type OutTarget;
        return SwitchSrcDst<OutTarget>::Type::Call(src, edge, dst);
    }
    static inline Idx SelectLeft(
        Idx src, Idx edge, Idx dst) {
        return LeftSelector::Call(src, edge, dst);
    }
    static inline Idx SelectRight(
        Idx src, Idx edge, Idx dst) {
        return RightSelector::Call(src, edge, dst);
    }
    static inline DType Op(DType* lhs, DType* rhs, int64_t len) {
        return BinaryOp::Call(lhs, rhs, len);
    }
    static inline DType Read(DType* addr) {
        return *addr;
    }
    static inline void Write(DType* addr, DType val) {
        Reducer::Call(addr, val);
    }
    static inline Idx GetId(Idx id, Idx* id_map) {
        return *(id_map + id);
    }
    static inline DType BackwardWrite(DType val, DType accum) {
        return Reducer::BackwardCall(val, accum);
    }
    static inline DType BackwardOpLhs(DType lhs, DType rhs, DType out) {
        return BinaryOp::BackwardLhs(lhs, rhs, out);
    }
    static inline DType BackwardOpRhs(DType lhs, DType rhs, DType out) {
        return BinaryOp::BackwardRhs(lhs, rhs, out);
    }
};

typedef minigun::advance::Config<true, minigun::advance::kV2N> AdvanceConfig;

}  // namespace cpu

#if ORIGINAL
// Template implementation of BackwardBinaryReduce operator.
template <int XPU, int Mode, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<Idx, DType>* gdata) {
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

#else
// Template implementation of BackwardBinaryReduce operator.
template <int XPU, int Mode, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<Idx, DType>* gdata) {

    // std::cout << "Mode: " << Mode << std::endl;
    // For backward computation, we use reverse csr and switch dst and src.
    // This benefits the most common src_op_edge or copy_src case, because the
    // gradients of src are now aggregated into destination buffer to reduce
    // competition of atomic add.
    typedef cpu::BackwardFunctorsTempl<Idx, DType,
                                       typename SwitchSrcDst<LeftSelector>::Type,
                                       typename SwitchSrcDst<RightSelector>::Type,
                                       BinaryOp, Reducer> Functors;
    if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping)
    {
        #if DTRACE
        std::cout << "CallBackwardBinaryReduce BASE..\n";
        #endif
        
        auto incsr = graph.GetInCSRMatrix();
        typedef cpu::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
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
        minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig, BackwardGData<Idx, DType>, UDF>(
            rtcfg, csr, gdata, minigun::IntArray1D<Idx>());        
    }
    else {
        #if BASE
        #if DTRACE
        std::cout << "CallBackwardBinaryReduce BASE..\n";
        #endif
        auto incsr = graph.GetInCSRMatrix();
        typedef cpu::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
        #else
        #if DTRACE
        std::cout << "CallBackwardBinaryReduce Intel..\n";
        #endif
        auto incsr = graph.GetOutCSRMatrix();
        typedef cpu::BackwardBinaryReduceIntel<Mode, Idx, DType, Functors> UDF;
        #endif

        // typedef cpu::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
        // auto incsr = graph.GetInCSRMatrix();

        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);

        // If the user-given mapping is none and the target is edge data, we need to
        // replace the mapping by the edge ids in the csr graph so that the edge
        // data is correctly read/written.
        if (LeftSelector::target == binary_op::kEdge
            && gdata->lhs_mapping == nullptr) {
            #if DTRACE
            std::cout << "lhs mapping backward\n";
            #endif
            gdata->lhs_mapping = static_cast<Idx*>(incsr.data->data);
        }
        if (RightSelector::target == binary_op::kEdge
            && gdata->rhs_mapping == nullptr) {
            #if DTRACE
            std::cout << "rhs mapping backward\n";
            #endif
            gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
        }
        if (OutSelector<Reducer>::Type::target == binary_op::kEdge
            && gdata->out_mapping == nullptr) {
            #if DTRACE
            std::cout << "out mapping backward\n";
            #endif
            gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
        }
        // TODO(minjie): allocator
        minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig, BackwardGData<Idx, DType>, UDF>(
            rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
    }
}
#endif

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_BACKWARD_DEFINE(mode, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduce<XPU,                \
                    mode, IDX, dtype,                           \
                    lhs_tgt, rhs_tgt,                           \
                    op<dtype>, REDUCER<XPU, dtype>>(            \
      const minigun::advance::RuntimeConfig& rtcfg,             \
      const CSRWrapper& graph,                                  \
      BackwardGData<IDX, dtype>* gdata);



#if ORIGINAL
// Template implementation of BackwardBinaryReduce with broadcasting operator.
template <int XPU, int Mode, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<NDim, Idx, DType>* gdata) {
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
  typedef cpu::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
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
#else
// Template implementation of BackwardBinaryReduce with broadcasting operator.
template <int XPU, int Mode, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduceBcast(
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
        std::cout << "CallBackwardBinaryReduceBcast BASE..\n";
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
        #if DTRACE
        std::cout << "CallBackwardBinaryReduceBcast BASE..\n";
        #endif
        typedef cpu::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
        auto incsr = graph.GetInCSRMatrix();
        #else
        #if DTRACE
        std::cout << "CallBackwardBinaryReduceBcast Intel..\n";
        #endif
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
#endif

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_BACKWARD_BCAST_DEFINE(mode, ndim, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduceBcast<XPU,                       \
                    mode, ndim, IDX, dtype,                                 \
                    lhs_tgt, rhs_tgt,                                       \
                    op<dtype>, REDUCER<XPU, dtype>>(                        \
      const minigun::advance::RuntimeConfig& rtcfg,                         \
      const CSRWrapper& graph,                                              \
      BackwardBcastGData<ndim, IDX, dtype>* gdata);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_
