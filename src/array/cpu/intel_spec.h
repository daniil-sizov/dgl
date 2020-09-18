#pragma once
#include <immintrin.h>
#include <type_traits>
namespace intel {

enum class engine_type { avx512, avx };
template<engine_type et,class T>
struct Engine;

template<> struct Engine<engine_type::avx512,float>  {
   typedef float T;
   typedef __m512 store_register;
   enum { offset = 16ULL };
   static inline store_register load(const T* input)  { return _mm512_loadu_ps(input);  }
   static inline store_register load_with_mask(const T* input, __mmask16 mask)  { return _mm512_maskz_loadu_ps(mask, input);  }
   static inline void store(T* output, store_register reg) { _mm512_storeu_ps (output,reg); }
   static inline void store_with_mask(T* output, store_register reg, __mmask16 mask) { _mm512_mask_storeu_ps (output, mask, reg);}
   static inline store_register add(store_register r1, store_register r2) { return _mm512_add_ps(r1,r2); }
   static inline store_register mul(store_register r1, store_register r2) { return _mm512_mul_ps(r1,r2); }
   static inline store_register set(T val) { return _mm512_set1_ps(val); }
};

template<> struct Engine<engine_type::avx512,double>  {
   typedef double T;
   typedef __m512d store_register;
   enum { offset = 8ULL };
   static inline store_register load(const T* input)  { return _mm512_loadu_pd(input);  }
   static inline store_register load_with_mask(const T* input, __mmask16 mask)  { return _mm512_maskz_loadu_pd(mask, input);  }
   static inline void store(T* output, store_register reg) { _mm512_storeu_pd (output,reg); }
   static inline void store_with_mask(T* output, store_register reg, __mmask16 mask) { _mm512_mask_storeu_pd (output, mask, reg); }
   static inline store_register add(store_register r1, store_register r2) { return _mm512_add_pd(r1,r2); }
   static inline store_register mul(store_register r1, store_register r2) { return _mm512_mul_pd(r1,r2); }
   static inline store_register set(T val) { return _mm512_set1_pd(val); }

};


template<> struct Engine<engine_type::avx,float>  {
   typedef float T;
   typedef __m256 store_register;
   enum { offset = 8ULL };
   static inline store_register load(const T* input)  { return _mm256_loadu_ps(input);  }
   static inline store_register load_with_mask(const T* input, __mmask16 mask)  { return _mm256_maskz_loadu_ps(mask, input);  }
   static inline void store(T* output, store_register reg) { _mm256_storeu_ps (output,reg); }
   static inline void store_with_mask(T* output, store_register reg, __mmask16 mask) { _mm256_mask_storeu_ps (output, mask, reg);}
   static inline store_register add(store_register r1, store_register r2) { return _mm256_add_ps(r1,r2); }
   static inline store_register mul(store_register r1, store_register r2) { return _mm256_mul_ps(r1,r2); }
   static inline store_register set(T val) { return _mm256_set1_ps(val); }
   


};


template<> struct Engine<engine_type::avx,double>  {
   typedef double T;
   typedef __m256d store_register;
   enum { offset = 4ULL };
   static inline store_register load(const T* input)  { return _mm256_loadu_pd(input);  }
   static inline store_register load_with_mask(const T* input, __mmask16 mask)  { return _mm256_maskz_loadu_pd(mask, input);  }
   static inline void store(T* output, store_register reg) { _mm256_storeu_pd(output,reg); }
   static inline void store_with_mask(T* output, store_register reg, __mmask16 mask) { _mm256_mask_storeu_pd (output, mask, reg);}
   static inline store_register add(store_register r1, store_register r2) { return _mm256_add_pd(r1,r2); }
   static inline store_register mul(store_register r1, store_register r2) { return _mm256_mul_pd(r1,r2); }
   static inline store_register set(T val) { return _mm256_set1_pd(val); }

};



namespace primitive {

template<class T>
void sum_update(T* sum_out, const T* input, size_t size)
{
        // typedef Engine<engine_type::avx512,T> Intel;
        typedef Engine<engine_type::avx,T> Intel;

         size_t _offset = 0;     
                                
         for (; (_offset < size - Intel::offset) && (size >=Intel::offset) ; _offset+=Intel::offset) {
               Intel::store(sum_out, Intel::add( Intel::load(input) ,Intel::load(sum_out) ));
               sum_out+=Intel::offset;
               input+=Intel::offset;
            
         }

         auto left_size = size - _offset;
           
         if( left_size > 0)
         {
             __mmask16 mask = (1 << left_size )-1;
     
             Intel::store_with_mask(sum_out, Intel::add( Intel::load_with_mask(input,mask) ,Intel::load_with_mask(sum_out,mask) ), mask);
         }

}


template<class T>
void mul_update(T* sum_out, const T* input1, const T* input2, size_t size)
{
        // typedef Engine<engine_type::avx512,T> Intel;
        typedef Engine<engine_type::avx,T> Intel;
        size_t _offset = 0;     
        for (; (_offset < size - Intel::offset) && (size >=Intel::offset) ; _offset+=Intel::offset) {
               Intel::store(sum_out, Intel::add(  Intel::mul( Intel::load(input2) , Intel::load(input1)) ,Intel::load(sum_out) ));
               sum_out+=Intel::offset;
               input1+=Intel::offset;
               input2+=Intel::offset;
         }

         auto left_size = size - _offset;
           
         if( left_size > 0)
         {
             __mmask16 mask = (1 << left_size )-1;
     
             Intel::store_with_mask(sum_out, Intel::add(  Intel::mul( Intel::load_with_mask(input2,mask), Intel::load_with_mask(input1,mask) )   ,Intel::load_with_mask(sum_out,mask) ), mask);
         }

}




template<class T>
void mul_update_withfirst(T* sum_out, const T* input1, const T* input2, size_t size)
{
        // typedef Engine<engine_type::avx512,T> Intel;
        typedef Engine<engine_type::avx,T> Intel;    
        size_t _offset = 0;     
         auto one =  Intel::set(*input2);               
         for (; (_offset < size - Intel::offset) && (size >=Intel::offset) ; _offset+=Intel::offset) {
               Intel::store(sum_out, Intel::add(  Intel::mul( one , Intel::load(input1)) ,Intel::load(sum_out) ));
               sum_out+=Intel::offset;
               input1+=Intel::offset;
             //  input2+=Intel::offset;
            
         }

         auto left_size = size - _offset;
           
         if( left_size > 0)
         {
             __mmask16 mask = (1 << left_size )-1;
              Intel::store_with_mask(sum_out, Intel::add(  Intel::mul( Intel::set(*input2), Intel::load_with_mask(input1,mask) )   ,Intel::load_with_mask(sum_out,mask) ), mask);
         }

}




}


}