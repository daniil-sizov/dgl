/*!
 *  Copyright (c) 2019 by Contributors
 * \file intel/cpu_support.h
 * \brief Intel CPU support 
 */
#ifndef INTEL_CPU_SUPPORT_H_
#define INTEL_CPU_SUPPORT_H_
#include <memory>
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
namespace intel {
#define log_intel(x) std::cout << x << std::endl;
#ifndef log_intel
#define log_intel(x)
#endif
template<class T>
using Uptr = std::unique_ptr<T>;

template<class T>
class elem_wise_add_update :  public Xbyak::CodeGenerator {
     typedef elem_wise_add_update<T> self;
     std::size_t size;
     bool applicable;

 public:
     explicit elem_wise_add_update(std::size_t _size) : size(_size), applicable(false) {
       static Xbyak::util::Cpu current_cpu;
       if (current_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
             /* prepare REMAINDER */
             mov(r8, rdx);    // rdx => size
             and_(r8, 0xf);

             /*  decrease  divident */
             sub(rdx, r8);
             cmp(rdx, 0);
             jz("remainder");
             xor_(r9, r9);

            L("for_i");
            vmovups(zmm0, ptr[rdi + r9*4]);
            vmovups(zmm1, ptr[rsi + r9*4]);
            vaddps(zmm2, zmm0, zmm1);
            vmovups(ptr[rdi + r9*4], zmm2);
            add(r9, 16);
            cmp(rdx, r9);
            jnz("for_i");

            L("remainder");
            cmp(r8, 0);  //  do we have remainder ?
            jz("done");
            xor_(rax, rax);
            mov(rax, 1);
            mov(rcx, r8);
            sal(rax, cl);
            dec(rax);
            kmovw(k1, eax);

            const uint8_t ptr_3[7] = { 0x62, 0xb1, 0x7c, 0x49, 0x10, 0x04, 0x8f };
            db(ptr_3, sizeof(ptr_3)/sizeof(uint8_t));

            const uint8_t ptr_2[7] = { 0x62, 0xb1, 0x7c, 0x49, 0x10, 0x0c, 0x8e };
            db(ptr_2, sizeof(ptr_2)/sizeof(uint8_t));
            vaddps(zmm2, zmm0, zmm1);

            /* vmovups ZMMWORD PTR [rdi+r9*4]{k1},zmm2  */
            const uint8_t ptr_1[7] = { 0x62, 0xB1, 0x7C, 0x49, 0x11, 0x14, 0x8F };
            db(ptr_1, sizeof(ptr_1)/sizeof(uint8_t));

            L("done");

            applicable = true;
            log_intel("*** AVX512F cpu kernel is ready  ***");
       }
       ret();
     }

     bool is_applicable() const {
       return applicable;
     }

     template<class ... P>
     void run(P ... args) {
         ((void(*)(P...))(this)->getCode())(args...);
     }
};


}  // namespace intel

#endif  // INTEL_CPU_SUPPORT_H_
