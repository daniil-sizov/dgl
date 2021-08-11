/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/async_transferer.cc
 * \brief The AsyncTransferer implementation.
 */


#include "async_transferer.h"

#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/device_api.h>
#include <vector>
#include <utility>
#include <dgl/array.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif 
#include <sched.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <stdio.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <omp.h>

#ifdef DGL_USE_CUDA
#include <cuda_runtime.h>
#include "../runtime/cuda/cuda_common.h"
#endif


namespace dgl {

using namespace runtime;

namespace dataloading {

using TransferId = AsyncTransferer::TransferId;

struct AsyncTransferer::Event {
  #ifdef DGL_USE_CUDA
  cudaEvent_t id;

  ~Event() {
    CUDA_CALL(cudaEventDestroy(id));
  }
  #endif
};

AsyncTransferer::AsyncTransferer(
    DGLContext ctx) :
  ctx_(ctx),
  next_id_(0),
  transfers_(),
  stream_(nullptr) {
  if (ctx_.device_type == kDLGPU) {
    stream_ = DeviceAPI::Get(ctx_)->CreateStream(ctx_);
  }
}


AsyncTransferer::~AsyncTransferer() {
  if (stream_) {
    DeviceAPI::Get(ctx_)->FreeStream(ctx_, stream_);
  }
}

TransferId AsyncTransferer::StartTransfer(
    NDArray src,
    DGLContext dst_ctx) {
  const TransferId id = GenerateId();

  Transfer t;
  t.src = src;

  DLDataType dtype = src->dtype;
  std::vector<int64_t> shape(src->shape, src->shape+src->ndim);
  t.dst = NDArray::Empty(shape, dtype, dst_ctx);

  if (stream_) {
    #ifdef DGL_USE_CUDA
    // get tensor information
    t.event.reset(new Event);
    CUDA_CALL(cudaEventCreate(&t.event->id));
    t.dst.CopyFrom(t.src, stream_);

    CUDA_CALL(cudaEventRecord(t.event->id, static_cast<cudaStream_t>(stream_)));
    #else
    LOG(FATAL) << "GPU support not compiled.";
    #endif
  } else {
    // copy synchronously since we don't have the notion of streams on the CPU
    t.event.reset(nullptr);
    t.dst.CopyFrom(t.src);
  }
  transfers_.emplace(id, std::move(t));

  return id;
}

NDArray AsyncTransferer::Wait(
    const TransferId id) {
  auto iter = transfers_.find(id);
  CHECK(iter != transfers_.end()) << "Unknown transfer: " << id;

  Transfer t = std::move(iter->second);
  transfers_.erase(iter);

  if (t.event) {
    #ifdef DGL_USE_CUDA
    // wait for it
    CUDA_CALL(cudaEventSynchronize(t.event->id));
    #endif
  }

  return t.dst;
}

TransferId AsyncTransferer::GenerateId() {
  return ++next_id_;
}

#ifndef SYS_gettid
#error "SYS_gettid unavailable on this system"
#endif

#define gettid() ((pid_t)syscall(SYS_gettid))

void setCores(const std::vector<int64_t>& cores) {

 cpu_set_t set;
 CPU_ZERO(&set);
 for(auto core : cores)
 {
    CPU_SET( core , &set);
 }

const char *value = std::getenv("OMP_NUM_THREADS");
if (value) 
  std::cout << "Get value from OMP_NUM_THREADS=" << atoi(value) << std::endl;
int nproc = (value) ? atoi( value ) : sysconf(_SC_NPROCESSORS_ONLN);
 
 
#pragma omp parallel for
for(int i=0;i<nproc;i++)
{
 if (sched_setaffinity(gettid(), sizeof(set), &set) == -1)
 {
   std::cout << "sched_setaffinity" << std::endl;
   exit(1);
 }
 std::cout <<"[" << i << "]["<< omp_get_thread_num() << "/" << omp_get_num_threads() << "] Affinity=" << getpid() << " tid=" << gettid() << std::endl; 
}
// std::cout << "## setCores #  "<<  << "/" << cores.size() << std::endl;      
           
}





DGL_REGISTER_GLOBAL("dataloading.async_transferer._CAPI_DGLAsyncTransfererCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DGLContext ctx = args[0];
    *rv = AsyncTransfererRef(std::make_shared<AsyncTransferer>(ctx));
});

DGL_REGISTER_GLOBAL("dataloading.async_transferer._CAPI_DGLAsyncTransfererStartTransfer")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  AsyncTransfererRef ref = args[0];
  NDArray array = args[1];
  DGLContext ctx = args[2];
  int id = ref->StartTransfer(array, ctx);
  *rv = id;
});

DGL_REGISTER_GLOBAL("dataloading.async_transferer._CAPI_DGLAsyncTransfererWait")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  AsyncTransfererRef ref = args[0];
  int id = args[1];
  NDArray arr = ref->Wait(id);
  *rv = arr;
});

DGL_REGISTER_GLOBAL("dataloading.async_transferer._CAPI_SetCores")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  // AsyncTransfererRef ref = args[0];
  // int id = args[1];
  // NDArray arr = ref->Wait(id);
  // *rv = arr;
  
    IdArray array = args[0];
    const auto& cores = array.ToVector<int64_t>();
    setCores(cores);

});



}  // namespace dataloading
}  // namespace dgl
