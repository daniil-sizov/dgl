/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/utils.cc
 * \brief Utility function implementations on CPU
 */
#include "../utils.h"
#include "quickperf.h"
namespace dgl {
namespace kernel {
namespace utils {
/* ToDo */

template<class T>
struct What {
   static const char* name() { return "unknown"; }
};

template<> struct What<int> {
   static const char *name() { return "int"; }
};
template<> struct What<float> {
   static const char *name() { return "float"; }
};
template<> struct What<double> {
   static const char *name() { return "double"; }
};

template<class T>
void replication(T* begin_mem, T *end_mem, T replicant)
{
          
         *begin_mem=replicant;
          T* last_mem;
          std::size_t bytes_copied=0;
          std::size_t total_bytes_to_copy=(end_mem-begin_mem)*sizeof(T);


         for(size_t i=1;2*i<static_cast<std::size_t>((end_mem-begin_mem));i<<=1)
         {           
             memcpy(begin_mem+i, begin_mem, i*sizeof(T));
             bytes_copied+=i*sizeof(T);
             last_mem=(begin_mem+2*i)-1;
         }


         if(total_bytes_to_copy - bytes_copied)
         {
              memcpy( last_mem, begin_mem, total_bytes_to_copy - bytes_copied );
         }

}

template <int XPU, typename DType>
void Fill(const DLContext& ctx, DType* ptr, size_t length, DType val) {
  quickperf::perf<2> perf("Fill origin");
  //replication(ptr,ptr+length,val); 
   #pragma omp parallel for schedule(static,256)
    for (size_t i = 0; i < length; ++i) {
      *(ptr + i) = val;
    }
  perf.end();
}

template void Fill<kDLCPU, float>(const DLContext& ctx, float* ptr, size_t length, float val);
template void Fill<kDLCPU, double>(const DLContext& ctx, double* ptr, size_t length, double val);

}  // namespace utils
}  // namespace kernel
}  // namespace dgl
