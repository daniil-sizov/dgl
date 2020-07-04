/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_impl.h
 * \brief Minigun CPU UDFs for binary reduce
 */
#ifndef DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_

#include <minigun/minigun.h>

#include <algorithm>

#include "../binary_reduce_impl_decl.h"
#include "../utils.h"
#include "./functor.h"
#include "../csr_interface.h"
#include <omp.h>
#include "macro.h"

namespace dgl {
namespace kernel {
namespace cpu {

template <typename Idx, typename DType, typename Functors>
struct BinaryReduceIntel {
    static inline bool CondEdge(
        Idx src, Idx dst, Idx eid, GData<Idx, DType>* gdata) {
        return true;
    }
    static inline void ApplyEdge(
        Idx src, Idx dst, Idx eid, GData<Idx, DType>* gdata) {
        // fprintf(stderr, "In apply edge..\n");
        const int64_t D = gdata->x_length;
        const int64_t len = gdata->data_len;
        // Idx lid = Functors::SelectLeft(src, eid, dst);
        // Idx rid = Functors::SelectRight(src, eid, dst);
        // Idx oid = Functors::SelectOut(src, eid, dst);
        Idx lid = Functors::SelectLeft(dst, eid, src);
        Idx rid = Functors::SelectRight(dst, eid, src);
        Idx oid = Functors::SelectOut(dst, eid, src);

        // if (omp_get_thread_num() == 0)
        // printf("%d %d %d \t %d %d %d \t %ld\n", src, eid, dst, lid, rid, oid, len);
        
        if (gdata->lhs_mapping) {
            lid = Functors::GetId(lid, gdata->lhs_mapping);
        }
        if (gdata->rhs_mapping) {
            rid = Functors::GetId(rid, gdata->rhs_mapping);
        }
        if (gdata->out_mapping) {
            oid = Functors::GetId(oid, gdata->out_mapping);
        }
        // if (omp_get_thread_num() == 0)
        // printf("%d %d %d \t %d %d %d \t %ld\n", src, eid, dst, lid, rid, oid, len);
        
        DType* lhsoff = gdata->lhs_data + lid * D * len;
        DType* rhsoff = gdata->rhs_data + rid * D * len;
        DType* outoff = gdata->out_data + oid * D;
        for (int64_t tx = 0; tx < D; ++tx) {
            DType out = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
            Functors::WriteIntel(outoff + tx, out);
        }
    }
};

// Minigun UDF to compute binary reduce.
template <typename Idx, typename DType, typename Functors>
struct BinaryReduce {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, GData<Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, GData<Idx, DType>* gdata) {
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
    for (int64_t tx = 0; tx < D; ++tx) {
      DType out = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
      Functors::Write(outoff + tx, out);
    }
  }
};

// Convert flattened index to multi-dimension index (assume row-major).
inline void Unravel(int64_t idx, int ndim,
    const int64_t* shape, const int64_t* stride, int64_t* out) {
  for (int d = 0; d < ndim; ++d) {
    out[d] = (idx / stride[d]) % shape[d];
  }
}

// Convert multi-dimension index to flattened index (assume row-major).
inline int64_t Ravel(const int64_t* idx, int ndim,
    const int64_t* shape, const int64_t* stride) {
  int64_t out = 0;
  for (int d = 0; d < ndim; ++d) {
    out += std::min(idx[d], shape[d] - 1) * stride[d];
  }
  return out;
}

// Minigun UDF to compute binary reduce with broadcasting.
template <int NDim, typename Idx, typename DType, typename Functors>
struct BinaryReduceBcastIntel {
    static inline bool CondEdge(
        Idx src, Idx dst, Idx eid, BcastGData<NDim, Idx, DType>* gdata) {
        return true;
    }
    static inline void ApplyEdge(
        Idx src, Idx dst, Idx eid, BcastGData<NDim, Idx, DType>* gdata) {
        const int64_t len = gdata->data_len;
        // printf("Applyedge: %d, out_len: %d\n", len, gdata->out_len);
    
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
        DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len * len;  // data with len size
        DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len * len;
        DType* outoff = gdata->out_data + oid * gdata->out_len;
        int64_t tmp[NDim];  // store unraveled idx.
        for (int64_t tx = 0; tx < gdata->out_len; ++tx) {
            Unravel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride, tmp);
            DType out = Functors::Op(
                lhsoff + Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride) * len,
                rhsoff + Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride) * len,
                len);

            Functors::WriteIntel(outoff + tx, out);
        }
    }
};

// Minigun UDF to compute binary reduce with broadcasting.
template <int NDim, typename Idx, typename DType, typename Functors>
struct BinaryReduceBcast {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, BcastGData<NDim, Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, BcastGData<NDim, Idx, DType>* gdata) {
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
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len * len;  // data with len size
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len * len;
    DType* outoff = gdata->out_data + oid * gdata->out_len;
    int64_t tmp[NDim];  // store unraveled idx.
    for (int64_t tx = 0; tx < gdata->out_len; ++tx) {
      Unravel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride, tmp);
      DType out = Functors::Op(
          lhsoff + Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride) * len,
          rhsoff + Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride) * len,
          len);

      Functors::Write(outoff + tx, out);
    }
  }
};

// Auxiliary template used in UDF.
template <typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct FunctorsTempl {
    static inline Idx SelectOut(
        Idx src, Idx edge, Idx dst) {
        return OutSelector<Reducer>::Type::Call(src, edge, dst);
    }
    static inline Idx SelectLeft(
        Idx src, Idx edge, Idx dst) {
        return LeftSelector::Call(src, edge, dst);
    }
    static inline Idx SelectRight(
        Idx src, Idx edge, Idx dst) {
        return RightSelector::Call(src, edge, dst);
    }
    static inline DType Op(DType *lhs, DType *rhs, int64_t len) {
        return BinaryOp::Call(lhs, rhs, len);
    }
    static inline void Write(DType* addr, DType val) {
        Reducer::Call(addr, val);
    }
    static inline void WriteIntel(DType* addr, DType val) {
        Reducer::CallIntel(addr, val);
    }        
    static inline Idx GetId(Idx id, Idx* id_map) {
        return *(id_map + id);
    }
};

typedef minigun::advance::Config<true, minigun::advance::kV2N> AdvanceConfig;

}  // namespace cpu


#if ORIGINAL
// Template implementation of BinaryReduce operator.
template <int XPU, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(const minigun::advance::RuntimeConfig& rtcfg,
                      const CSRWrapper& graph,
                      GData<Idx, DType>* gdata) {
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
#else
// Template implementation of BinaryReduce operator.
template <int XPU, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(const minigun::advance::RuntimeConfig& rtcfg,
                      const CSRWrapper& graph,
                      GData<Idx, DType>* gdata) {
    typedef cpu::FunctorsTempl<Idx, DType, LeftSelector,
                               RightSelector, BinaryOp, Reducer> Functors;

    if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        // std::cout << "inbuilt lhs/edge mapping..\n";
        #if DTRACE
        std::cout << "CallBinaryReduce BASE..\n";
        #endif
        
        typedef cpu::BinaryReduce<Idx, DType, Functors> UDF;  
        auto outcsr = graph.GetOutCSRMatrix(); // csr
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
        // If the user-given mapping is none and the target is edge data, we need to
        // replace the mapping by the edge ids in the csr graph so that the edge
        // data is correctly read/written.
        
        if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
            #if DTRACE
            std::cout << "lhs mapping\n";
            #endif
            gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
            #if DTRACE
            std::cout << "rhs mapping\n";
            #endif
            gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        if (OutSelector<Reducer>::Type::target == binary_op::kEdge
            && gdata->out_mapping == nullptr) {
            #if DTRACE
            std::cout << "Out mapping\n";
            #endif
            gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        // TODO(minjie): allocator
        minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig, GData<Idx, DType>, UDF>(
            rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
    }
    else
    {
        #if BASE   // i.e original dgl
        #if DTRACE
        std::cout << "CallBinaryReduce BASE..\n";
        #endif
        typedef cpu::BinaryReduce<Idx, DType, Functors> UDF;  
        auto outcsr = graph.GetOutCSRMatrix(); // csr
        #else
        #if DTRACE
        std::cout << "CallBinaryReduce INTEL..\n";
        #endif
        typedef cpu::BinaryReduceIntel<Idx, DType, Functors> UDF;
        auto outcsr = graph.GetInCSRMatrix();
        #endif
        
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
        // If the user-given mapping is none and the target is edge data, we need to
        // replace the mapping by the edge ids in the csr graph so that the edge
        // data is correctly read/written.
        
        if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
            #if DTRACE
            std::cout << "lhs mapping\n";
            #endif
            gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
            #if DTRACE
            std::cout << "rhs mapping\n";
            #endif
            gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        if (OutSelector<Reducer>::Type::target == binary_op::kEdge
            && gdata->out_mapping == nullptr) {
            #if DTRACE
            std::cout << "Out mapping\n";
            #endif
            gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
        }
        // TODO(minjie): allocator
        minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig, GData<Idx, DType>, UDF>(
            rtcfg, csr, gdata, minigun::IntArray1D<Idx>());        
    }

}
#endif


#if ORIGINAL
// Template implementation of BinaryReduce broadcasting operator.
template <int XPU, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast(
  const minigun::advance::RuntimeConfig& rtcfg,
  const CSRWrapper& graph,
  BcastGData<NDim, Idx, DType>* gdata) {
  typedef cpu::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cpu::BinaryReduceBcast<NDim, Idx, DType, Functors> UDF;
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
  minigun::advance::Advance<XPU, Idx, cpu::AdvanceConfig,
    BcastGData<NDim, Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}
#else
// Template implementation of BinaryReduce broadcasting operator.
template <int XPU, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast(
  const minigun::advance::RuntimeConfig& rtcfg,
  const CSRWrapper& graph,
  BcastGData<NDim, Idx, DType>* gdata) {
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
    
        // csr
        // auto outcsr = graph.GetOutCSRMatrix();
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
#endif

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_DEFINE(dtype, lhs_tgt, rhs_tgt, op)                    \
  template void CallBinaryReduce<XPU, IDX,                      \
        dtype, lhs_tgt, rhs_tgt, op<dtype>, REDUCER<XPU, dtype>>(  \
      const minigun::advance::RuntimeConfig& rtcfg,                \
      const CSRWrapper& graph,                                     \
      GData<IDX, dtype>* gdata);

#define GEN_BCAST_DEFINE(ndim, dtype, lhs_tgt, rhs_tgt, op)         \
  template void CallBinaryReduceBcast<XPU, ndim, IDX, dtype,     \
                                 lhs_tgt, rhs_tgt,                  \
                                 op<dtype>, REDUCER<XPU, dtype>>(   \
      const minigun::advance::RuntimeConfig& rtcfg,                 \
      const CSRWrapper& graph,                                      \
      BcastGData<ndim, IDX, dtype>* gdata);

#define EVAL(F, ...) MSVC_EXPAND(F(__VA_ARGS__))

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_
