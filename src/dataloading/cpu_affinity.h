#ifndef DGL_DATALOADING_CPU_AFFINITY_H_
#define DGL_DATALOADING_CPU_AFFINITY_H_

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

namespace dgl {
namespace dataloading {
void FakeGompAfinity(const std::vector<int64_t>& cores);
void PinOMP2Cores( const std::vector<int64_t>& cores );
std::vector<pid_t> get_all_tids(pid_t self);

}
}

#endif
