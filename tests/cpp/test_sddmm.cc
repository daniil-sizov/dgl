#if !defined(_WIN32)
#include <../../src/array/cpu/sddmm.h>
#include <dgl/array.h>
#include <gtest/gtest.h>
#include <time.h>
#include <random>
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;

int64_t sizes_sddmm[] = {3, 7, 8, 9, 31, 32, 33, 54, 63, 64, 65, 256, 257, 3, 7};

namespace ns_op = dgl::aten::cpu::sddmm_op;
namespace {

template <class T>
void GenerateData(T* data, int64_t dim, T mul) {
  for (int64_t i = 0; i < dim; i++) {
    data[i] = (i + 1) * mul;
    //std::cout  << data[i] << std::endl;
  }
}

template <class T>
void GenerateRandomData(T* data, int64_t dim) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(1, 100);
  for (int64_t i = 0; i < dim; i++) {
    data[i] = (dist(rng) / 10);
  }
}

template <class T>
void GenerateZeroData(T* data, int64_t dim) {
  for (int64_t i = 0; i < dim; i++) {
    data[i] = (int64_t)0;
  }
}

void GenerateRandomLength(int64_t& len, int64_t dim) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(1, dim);
  len = dist(rng);
  len = 3;
}

template <class T>
void Copy(T* exp, T* hs, int64_t dim) {
  for (int64_t i = 0; i < dim; i++) {
    exp[i] = hs[i];
  }
}

template <class T>
void Add(T* exp, T* lhs, T* rhs, int64_t dim) {
  for (int64_t i = 0; i < dim; i++) {
    exp[i] = lhs[i] + rhs[i];
  }
}

template <class T>
void Sub(T* exp, T* lhs, T* rhs, int64_t dim) {
  for (int64_t i = 0; i < dim; i++) {
    exp[i] = lhs[i] - rhs[i];
  }
}

template <class T>
void Mul(T* exp, T* lhs, T* rhs, int64_t dim) {
  for (int64_t i = 0; i < dim; i++) {
    exp[i] = lhs[i] * rhs[i];
  }
}

template <class T>
void Div(T* exp, T* lhs, T* rhs, int64_t dim) {
  for (int64_t i = 0; i < dim; i++) {
    exp[i] = lhs[i] / rhs[i];
  }
}

template <class T>
void Dot(T* exp, T* lhs, T* rhs, int64_t dim, int64_t len) {
  for (int64_t i = 0; i < dim; i++) {
    T res = 0;
    for (int64_t l = 0; l < len; l++) {
      res += lhs[i+l] * rhs[i+l];
    }
    exp[i] = res;
  }
}

template <class T>
void CheckResult(T* exp, T* out, T* out_intel_kernel, int64_t dim) {
  for (int64_t i = 0; i < dim; i++) {
    if(std::isnan(exp[i]) || std::isinf(exp[i])) {
      continue;
    }
    ASSERT_TRUE(exp[i] == out[i]);
    if (out_intel_kernel != nullptr) {
      ASSERT_TRUE(out[i] == out_intel_kernel[i]);
    }
  }
}

}  // namespace

template <class ElemWiseUpd>
ElemWiseUpd* generic_ElemWiseUpd() {
  static std::unique_ptr<ElemWiseUpd> asm_kernel_ptr(
    (dgl::IntelKernel<>::IsEnabled()) ? new ElemWiseUpd() : nullptr);
  ElemWiseUpd* cpu_spec = (asm_kernel_ptr && asm_kernel_ptr->applicable())
                            ? asm_kernel_ptr.get()
                            : nullptr;

  return cpu_spec;
}

template <typename IDX>
void _TestSddmmCopyLhs() {
  for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++) {
    int64_t dim = sizes_sddmm[i];
    IDX out[dim], exp[dim], lhs[dim];
    int64_t len = 0;

    GenerateZeroData(out, dim);
    GenerateRandomData(lhs, dim);

    // Calculation of expected output - 'exp'
    Copy(exp, lhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int64_t k = 0; k < dim; k++) {
      out[k] = ns_op::CopyLhs<IDX>::Call(lhs + k, nullptr, len);
    }

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
      generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::CopyLhs<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, nullptr, dim, len);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SddmmTest, TestSddmmCopyLhs) {
  _TestSddmmCopyLhs<float>();
  _TestSddmmCopyLhs<double>();
}

template <typename IDX>
void _TestSddmmCopyRhs() {
  for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++) {
    int64_t dim = sizes_sddmm[i];
    IDX out[dim], exp[dim], rhs[dim];
    int64_t len = 0;

    GenerateZeroData(out, dim);
    GenerateRandomData(rhs, dim);

    // Calculation of expected output - 'exp'
    Copy(exp, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int64_t k = 0; k < dim; k++) {
      out[k] = ns_op::CopyRhs<IDX>::Call(nullptr, rhs + k, len);
    }

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
      generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::CopyRhs<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, nullptr, rhs, dim, len);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SddmmTest, TestSddmmCopyRhs) {
  _TestSddmmCopyRhs<float>();
  _TestSddmmCopyRhs<double>();
}

template <typename IDX>
void _TestSddmmAdd() {
  for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++) {
    int64_t dim = sizes_sddmm[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    int64_t len = 0;

    GenerateZeroData(out, dim);
    GenerateRandomData(lhs, dim);
    GenerateRandomData(rhs, dim);

    // Calculation of expected output - 'exp'
    Add(exp, lhs, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int64_t k = 0; k < dim; k++) {
      out[k] = ns_op::Add<IDX>::Call(lhs + k, rhs + k, len);
    }

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
      generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::Add<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, rhs, dim, len);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SddmmTest, TestSddmmAdd) {
  _TestSddmmAdd<float>();
  _TestSddmmAdd<double>();
}

template <typename IDX>
void _TestSddmmSub() {
  for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++) {
    int64_t dim = sizes_sddmm[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    int64_t len = 0;

    GenerateZeroData(out, dim);
    GenerateRandomData(lhs, dim);
    GenerateRandomData(rhs, dim);

    // Calculation of expected output - 'exp'
    Sub(exp, lhs, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int64_t k = 0; k < dim; k++) {
      out[k] = ns_op::Sub<IDX>::Call(lhs + k, rhs + k, len);
    }

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
      generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::Sub<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, rhs, dim, len);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SddmmTest, TestSddmmSub) {
  _TestSddmmSub<float>();
  _TestSddmmSub<double>();
}

template <typename IDX>
void _TestSddmmMul() {
  for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++) {
    int64_t dim = sizes_sddmm[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    int64_t len = 0;

    GenerateZeroData(out, dim);
    GenerateRandomData(lhs, dim);
    GenerateRandomData(rhs, dim);

    // Calculation of expected output - 'exp'
    Mul(exp, lhs, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int64_t k = 0; k < dim; k++) {
      out[k] = ns_op::Mul<IDX>::Call(lhs + k, rhs + k, len);
    }

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
      generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::Mul<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, rhs, dim, len);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SddmmTest, TestSddmmMul) {
  _TestSddmmMul<float>();
  _TestSddmmMul<double>();
}

template <typename IDX>
void _TestSddmmDiv() {
  for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++) {
    int64_t dim = sizes_sddmm[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    int64_t len = 0;

    GenerateZeroData(out, dim);
    GenerateData(lhs, dim,(IDX)15);
    GenerateData(rhs, dim, (IDX)1);

    // Calculation of expected output - 'exp'
    Div(exp, lhs, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int64_t k = 0; k < dim; k++) {
      out[k] = ns_op::Div<IDX>::Call(lhs + k, rhs + k, len);
    }

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
      generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::Div<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, rhs, dim, len);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SddmmTest, TestSddmmDiv) {
  _TestSddmmDiv<float>();
  _TestSddmmDiv<double>();
}

template <typename IDX>
void _TestSddmmDot() {
  for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++) {
    int64_t dim = sizes_sddmm[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    int64_t len = 0;

    GenerateZeroData(out, dim);
    GenerateZeroData(exp, dim);
    GenerateRandomData(lhs, dim);
    GenerateRandomData(rhs, dim);
    GenerateRandomLength(len, dim);

    // Calculation of expected output - 'exp'
    Dot(exp, lhs, rhs, dim, len);

    // Calculation of output using legacy path - 'out'
    for (int64_t k = 0; k < dim; k++) {
      out[k] = ns_op::Dot<IDX>::Call(lhs + k, rhs + k, len);
    }

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
      generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::Dot<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      for (int64_t k = 0; k < len; k++) {
        cpu_spec->run(out_intel_kernel, lhs+k, rhs+k, dim);
      }
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

template <class T>
void DotReduce(T *exp, int64_t exp_dim,  T *lhs, T *rhs, int64_t reduce_size)
{
  for (int64_t i = 0; i < exp_dim; i++)
  {
    T res = 0;
    for (int64_t r = 0; r < reduce_size; r++)
    {
      res += lhs[i * reduce_size + r] * rhs[i * reduce_size + r];
    }
    exp[i] = res;
  }
}
template<class T>
using Vec = std::vector<T>;

template <typename IDX>
void _NewTestSddmmDot()
{

  const size_t reduce_size = 20;

   for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++)
  {
    int64_t dim = sizes_sddmm[i];

    for (size_t r = 1; r < reduce_size; r++)
    {
       size_t reduce_size_LR  = dim*r;

       Vec<IDX> out(dim), exp(dim), lhs(reduce_size_LR), rhs(reduce_size_LR);

       GenerateZeroData((IDX*)out.data(), dim);
       GenerateZeroData((IDX *)exp.data(), dim);
       GenerateRandomData((IDX *)lhs.data(), reduce_size_LR);
       GenerateRandomData((IDX *)rhs.data(), reduce_size_LR);

       // Calculation of expected output - 'exp'
       DotReduce((IDX *)exp.data(), dim, (IDX *)lhs.data(), (IDX *)rhs.data(), r);

       for (size_t k = 0; k < dim; ++k)
       {
         size_t offset = r * k;
         out[k] = ns_op::Dot<IDX>::Call((IDX*)lhs.data() + offset, (IDX*)rhs.data() + offset, r);
      }

       CheckResult((IDX*)exp.data(), (IDX*)out.data(), (IDX*)NULL, dim);

         auto *cpu_spec =
             generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::Dot<IDX>>>();
         if (cpu_spec)
         {
           Vec<IDX> out_intel_kernel(dim);
           GenerateZeroData((IDX*)out_intel_kernel.data(), dim);
           cpu_spec->run((IDX*)lhs.data() , (IDX*)rhs.data() , r, (IDX*)out_intel_kernel.data(), dim);
        //   CheckResult(exp, out_intel_kernel, (IDX *)NULL, dim);
         }
         else
         {
              std::cout << "Not" << std::endl;
         }
    }


  }


  // for (size_t i = 0; i < sizeof(sizes_sddmm) / sizeof(int64_t); i++)
  // {
  //   int64_t dim = sizes_sddmm[i];
  //   IDX out[dim], exp[dim], lhs[dim], rhs[dim];
  //   int64_t len = 0;

  //   GenerateZeroData(out, dim);
  //   GenerateZeroData(exp, dim);
  //   GenerateRandomData(lhs, dim*8);
  //   GenerateRandomData(rhs, dim*8);
  //   GenerateRandomLength(len, dim);

  //   // Calculation of expected output - 'exp'

  //   Dot(exp, lhs, rhs, dim, len);

  //   // Calculation of output using legacy path - 'out'
  //   for (int64_t k = 0; k < dim; k++)
  //   {
  //     out[k] = ns_op::Dot<IDX>::Call(lhs + k, rhs + k, len);
  //   }

  //   // Calculation of output using intel path - 'out_intel_kernel'
  //   auto *cpu_spec =
  //       generic_ElemWiseUpd<dgl::ElemWiseUpdate<ns_op::Dot<IDX>>>();
  //   if (cpu_spec)
  //   {
  //     IDX out_intel_kernel[dim];
  //     GenerateZeroData(out_intel_kernel, dim);
  //     for (int64_t k = 0; k < len; k++)
  //     {
  //       cpu_spec->run(out_intel_kernel, lhs + k, rhs + k, dim);
  //     }
  //     CheckResult(exp, out, out_intel_kernel, dim);
  //   }
  //   else
  //   {
  //     IDX *out_intel_kernel = nullptr;
  //     CheckResult(exp, out, out_intel_kernel, dim);
  //   }
  // }
}

TEST(SddmmTest, TestSddmmDot) {
  _NewTestSddmmDot<float>();
 // _TestSddmmDot<double>();
}
#endif
