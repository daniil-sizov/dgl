/*!
 *  Copyright (c) 2019 by Contributors
 * \file intel/cpu_support.h
 * \brief Intel CPU support
 */
#ifndef INTEL_CPU_SUPPORT_H_
#define INTEL_CPU_SUPPORT_H_
#include <memory>
#include <tuple>
#include <type_traits>
#include "dmlc/logging.h"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
using namespace dmlc;
using namespace dgl::runtime;
using namespace dgl;
namespace intel {

//  #define log_intel(x) if (IntelKernel<>::log_enabled()) { std::cout << x <<
//  std::endl; }
#ifndef log_intel
#define log_intel(x)                  \
  if (IntelKernel<>::log_enabled()) { \
    LOG(INFO) << x;                   \
  }
#endif

static inline Xbyak::Zmm make_zmm(const Xbyak::Xmm &v) {
  return Xbyak::Zmm(v.getIdx());
}
template <int version = 0>
struct IntelKernel {
  static int64_t getValue() {
    int64_t v = 0;
    const char *label = "DGL_CPU_INTEL_KERNEL_ENABLED";
    const char *ptr = std::getenv(label);
    if (ptr) {
      v = atoll(ptr);
      log_intel(label << "=>" << v);
    }
    return v;
  }

  static int64_t enabled() {
    static int64_t r = IntelKernel<version>::getValue();
    return r;
  }

  static int log_enabled() {
    static int r = (std::getenv("DGL_CPU_INTEL_KERNEL_LOG")) ? 1 : 0;
    return r;
  }
};

namespace op_ns = ::dgl::aten::cpu::op;

template <typename T, typename Tuple>
struct has_type;

template <typename T>
struct has_type<T, std::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct has_type<T, std::tuple<U, Ts...>> : has_type<T, std::tuple<Ts...>> {};

template <typename T, typename... Ts>
struct has_type<T, std::tuple<T, Ts...>> : std::true_type {};

typedef std::tuple<float, double> supported_types;

template <class OCmp, template <class> class ToP, class Tup,
          int ok = std::tuple_size<Tup>::value>
struct DeepType;

template <class OCmp, template <class> class ToP, class Tup>
struct DeepType<OCmp, ToP, Tup, 1> {
  typedef typename std::tuple_element<0, Tup>::type EL1;
  enum { value = std::is_same<OCmp, ToP<EL1>>::value };
};

template <class OCmp, template <class> class ToP, class Tup>
struct DeepType<OCmp, ToP, Tup, 2> {
  typedef typename std::tuple_element<0, Tup>::type EL1;
  typedef typename std::tuple_element<1, Tup>::type EL2;
  enum {
    value = (std::is_same<OCmp, ToP<EL1>>::value ||
             std::is_same<OCmp, ToP<EL2>>::value)
  };
};

template <class OCmp, template <class> class ToP, class Tup>
struct DeepType<OCmp, ToP, Tup, 3> {
  typedef typename std::tuple_element<0, Tup>::type EL1;
  typedef typename std::tuple_element<1, Tup>::type EL2;
  typedef typename std::tuple_element<2, Tup>::type EL3;
  enum {
    value = (std::is_same<OCmp, ToP<EL1>>::value ||
             std::is_same<OCmp, ToP<EL2>>::value ||
             std::is_same<OCmp, ToP<EL3>>::value)
  };
};
/*!
 * \brief Intel CPU kernel for SpMMSum .
 * \note it uses AVX512.
 */
template <class Op>
class elem_wise_add_update : public Xbyak::CodeGenerator {
 public:
  typedef Op op_type;
  typedef typename Op::type DType;
  static_assert(
    std::is_base_of<std::true_type, has_type<DType, supported_types>>::value,
    "Use case fail Intel::elem_wise_add< Operator<DType> > DType is not "
    "supported !");

 protected:
  const Xbyak::Reg64 &r_out_;
  const Xbyak::Reg64 &r_left_;
  const Xbyak::Reg64 &r_right;
  const Xbyak::Reg64 &r_size_;

  /* [functional] Does kernel is applicable on this machine ? */
  bool applicable_;

 public:
  static constexpr int UNIT_SIZE_BYTES = sizeof(DType);
  static constexpr int BITS_IN_BYTES = 8;
  static constexpr int REG_BIT_SIZE = 512;
  static constexpr int UNIT_PER_REG =
    REG_BIT_SIZE / (UNIT_SIZE_BYTES * BITS_IN_BYTES);

  template <bool b>
  using Required = typename std::enable_if<b, bool>::type;

  template <class L, class R>
  using CheckCmp = Required<std::is_same<L, R>::value>;

  template <class L, class R1, class R2>
  using CheckCmp_2 =
    Required<std::is_same<L, R1>::value || std::is_same<L, R2>::value>;

  template <class OpType, template <class> class TPP, class Tup>
  using Verify = Required<DeepType<OpType, TPP, Tup>::value>;

  template <class TType, class R1, class R2, CheckCmp<TType, float> = true>
  void alias_load(R1 r1, R2 r2) {
    vmovups(r1, r2);
  }
  template <class TType, class R1, class R2, CheckCmp<TType, double> = true>
  void alias_load(R1 r1, R2 r2) {
    vmovupd(r1, r2);
  }

  template <class TType, class R1, class R2, CheckCmp<TType, float> = true>
  void alias_save(R1 r1, R2 r2) {
    alias_load<TType>(r1, r2);
  }
  template <class TType, class R1, class R2, CheckCmp<TType, double> = true>
  void alias_save(R1 r1, R2 r2) {
    alias_load<TType>(r1, r2);
  }

  template <class TType, class R1, class R2, class R3,
            CheckCmp<TType, float> = true>
  void alias_ADD(R1 r1, R2 r2, R3 r3) {
    vaddps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            CheckCmp<TType, double> = true>
  void alias_ADD(R1 r1, R2 r2, R3 r3) {
    vaddpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            CheckCmp<TType, float> = true>
  void alias_SUB(R1 r1, R2 r2, R3 r3) {
    vsubps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            CheckCmp<TType, double> = true>
  void alias_SUB(R1 r1, R2 r2, R3 r3) {
    vsubpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            CheckCmp<TType, float> = true>
  void alias_DIV(R1 r1, R2 r2, R3 r3) {
    vdivps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            CheckCmp<TType, double> = true>
  void alias_DIV(R1 r1, R2 r2, R3 r3) {
    vdivpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            CheckCmp<TType, float> = true>
  void alias_MUL(R1 r1, R2 r2, R3 r3) {
    vmulps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            CheckCmp<TType, double> = true>
  void alias_MUL(R1 r1, R2 r2, R3 r3) {
    vmulpd(r1, r2, r3);
  }

  template <class Operator,
            Verify<Operator, op_ns::CopyLhs, supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    alias_load<IType>(zmm0, ptr[r_out_ + r9 * sizeof(IType)]);
    alias_load<IType>(zmm1, ptr[r_left_ + r9 * sizeof(IType)]);
    alias_ADD<IType>(zmm2, zmm0, zmm1);
    alias_save<IType>(ptr[r_out_ + r9 * sizeof(IType)], zmm2);
  }
  template <class Operator,
            Verify<Operator, op_ns::CopyRhs, supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    alias_load<IType>(zmm0, ptr[r_out_ + r9 * sizeof(IType)]);
    alias_load<IType>(zmm1, ptr[r_right + r9 * sizeof(IType)]);
    alias_ADD<IType>(zmm2, zmm0, zmm1);
    alias_save<IType>(ptr[r_out_ + r9 * sizeof(IType)], zmm2);
  }
  template <class T>
  void loop_pre() {
    alias_load<T>(zmm0, ptr[r_out_ + r9 * sizeof(T)]);
    alias_load<T>(zmm1, ptr[r_left_ + r9 * sizeof(T)]);
    alias_load<T>(zmm2, ptr[r_right + r9 * sizeof(T)]);
  }
  template <class T>
  void loop_post() {
    alias_ADD<T>(zmm2, zmm0, zmm2);
    alias_save<T>(ptr[r_out_ + r9 * sizeof(T)], zmm2);
  }
  template <class Operator,
            Verify<Operator, op_ns::Add, supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_ADD<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }
  template <class Operator,
            Verify<Operator, op_ns::Sub, supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_SUB<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator,
            Verify<Operator, op_ns::Div, supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_DIV<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator,
            Verify<Operator, op_ns::Mul, supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_MUL<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator,
            Verify<Operator, op_ns::CopyLhs, supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    alias_load<IType>(make_zmm(zmm2) | mask, ptr[r_left_ + r9 * sizeof(IType)]);
  }

  template <class Operator,
            Verify<Operator, op_ns::CopyRhs, supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    alias_load<IType>(make_zmm(zmm2) | mask, ptr[r_right + r9 * sizeof(IType)]);
  }

  template <class T>
  void remainder_fetch_LR(const Xbyak::Opmask mask) {
    alias_load<T>(make_zmm(zmm2) | mask, ptr[r_left_ + r9 * sizeof(T)]);
    alias_load<T>(make_zmm(zmm1) | mask, ptr[r_right + r9 * sizeof(T)]);
  }

  template <class Operator,
            Verify<Operator, op_ns::Mul, supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_MUL<IType>(zmm2, zmm2, zmm1);
  }

  template <class Operator,
            Verify<Operator, op_ns::Add, supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_ADD<DType>(zmm2, zmm2, zmm1);
  }

  template <class Operator,
            Verify<Operator, op_ns::Div, supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_DIV<DType>(zmm2, zmm2, zmm1);
  }

  template <class Operator,
            Verify<Operator, op_ns::Sub, supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_SUB<DType>(zmm2, zmm2, zmm1);
  }

  elem_wise_add_update()
      : r_out_(rdi),
        r_left_(rsi),
        r_right(rdx),
        r_size_(rcx),
        applicable_(false) {
    static Xbyak::util::Cpu current_cpu;

    /* Default case for all */
    if (current_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
      /* prepare REMAINDER */
      mov(r8, r_size_);
      and_(r8,
           UNIT_PER_REG - 1);  // r8_modulo = size/(sizeof(zmm)/sizeof(float))
      xor_(r9, r9);            // reset r9
      cmp(r_size_, UNIT_PER_REG);  // if ( size < 16 ) {  }
      jl("remainder");

      /*  decrease  divident */
      sub(r_size_, r8);  // prepare alignment chunks
      cmp(r_size_, 0);   // do we have any full chunks ?
      jz("remainder");

      L("for_i");
      full_chunk_loop_operations<Op>();
      add(r9, UNIT_PER_REG);  // r9+=sizeof(zmm)/sizeof(float)
      cmp(r_size_, r9);       // more full chunks ?
      jnz("for_i");

      L("remainder");
      cmp(r8, 0);  //  do we have a remainder ?
      jz("done");
      /* prepare a bitmask for k1 */
      mov(rax, 1);
      mov(r_size_, r8);
      sal(rax, cl);
      dec(rax);        // k1= (1 << r8 )-1
      kmovw(k1, eax);  // set bitmask
      alias_load<DType>(make_zmm(zmm0) | k1,
                        ptr[r_out_ + r9 * UNIT_SIZE_BYTES]);
      remainder_operations<Op>(k1);
      alias_ADD<DType>(zmm3, zmm2, zmm0);
      alias_save<DType>(ptr[r_out_ + r9 * UNIT_SIZE_BYTES],
                        make_zmm(zmm3) | k1);
      L("done");
      applicable_ = true;
      log_intel("AVX512F cpu kernel is ready");
    }
    ret();
  }

  bool is_applicable() const { return applicable_; }

  template <class... P>
  void run(P... args) {
    ((void (*)(P...))(this)->getCode())(args...);
  }
};

}  // namespace intel

#endif  // INTEL_CPU_SUPPORT_H_
