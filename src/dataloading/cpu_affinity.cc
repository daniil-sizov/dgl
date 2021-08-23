#include "cpu_affinity.h"
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/device_api.h>
#include <vector>
#include <utility>
#include <sys/types.h>
#include <dirent.h>
#include <pthread.h>
#include <mutex>
#include <memory>

#ifndef SYS_gettid
#error "SYS_gettid unavailable on this system"
#endif

#define gettid() ((pid_t)syscall(SYS_gettid))

namespace dgl {

std::recursive_mutex& getMutex() {

    static std::recursive_mutex mx;
    return mx;
}
using namespace runtime;
#define log_info(x) { std::cout << "[CPU_AFFINITY]: " << x << std::endl; }
#define log_info_lock(x) { std::lock_guard<decltype(getMutex())> l(getMutex()); std::cout << "[CPU_AFFINITY]: " << x << std::endl; }
#define log_error(x) { std::cout << "[ERROR_CPU_AFFINITY]: " << x << std::endl; }
namespace dataloading {

     inline int getNumOfCores() {

      const char* env_name = "OMP_NUM_THREADS";

      const char *value = std::getenv(env_name);

      int nproc = (value) ? atoi( value ) : sysconf(_SC_NPROCESSORS_ONLN);

      if (value)
      {
           log_info("use num of cores  from" << env_name << "=" << nproc );
      }
      else
      {
           log_info("use num of cores sysconf=" <<  nproc );
      }
       return nproc;
     }

     void FakeGompAfinity(const std::vector<int64_t>& cores) {

      if(cores.size())
      {
         log_error("FAKE_GOMP_AFINITY for " << gettid());
         return;
      }

     // auto nproc = getNumOfCores();




      #pragma omp parallel for
      for(int i=0;i<omp_get_max_threads();i++)
      //for(int i=0;i<3;i++)
     /*
      {
       std::lock_guard<decltype(getMutex())> l(getMutex());
        cpu_set_t set;
        CPU_ZERO(&set);
        auto hyperthread = i % 31;  // 32 cores from 36
        CPU_SET( hyperthread , &set);
        auto tid = gettid();
        if (sched_setaffinity(tid, sizeof(set), &set) == -1)
        {
           log_error("sched_setaffinity for " << tid);
           continue;
        }
        log_info_lock("["<< i << "] fake  OK OMP affinity for pid="<< getpid() << " tid=" << tid << " pthread_self=" << pthread_self());
     }
     */
     {

         cpu_set_t cpuset;
         pthread_t tid;
         tid = pthread_self();

/* Set affinity mask to include CPUs 0 to 7 */

          CPU_ZERO(&cpuset);
          auto hyperthread = i % 31;
          CPU_SET(hyperthread, &cpuset);

          auto s = pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
          if (s != 0) {
                   log_error("sched_setaffinity for " << tid);
          }
          log_info_lock("["<< i << "] fake pthread_setaffinity OK OMP affinity for pid="<< getpid() << " tid=" << tid  );
     }




     }



      void PinOMP2Cores( const std::vector<int64_t>& cores ) {
       cpu_set_t set;
       CPU_ZERO(&set);
       for(auto core : cores)
       {
          CPU_SET( core , &set);
       }

      auto nproc = getNumOfCores();

      #pragma omp parallel for
      for(int i=0;i<nproc;i++)
      {
         auto tid = gettid();
        if (sched_setaffinity(tid, sizeof(set), &set) == -1)
        {
           log_error("sched_setaffinity for " << gettid());
           continue;
        }
        log_info("["<< i << "]  OK OMP affinity for pid="<< getpid() << " tid=" << tid );
     }
      /*
      auto all_pids =  get_all_tids(getpid());
      for(auto _pid : all_pids)
      {
        if (sched_setaffinity(_pid, sizeof(set), &set) == -1)
        {
           log_error("sched_setaffinity for " << gettid());
           continue;
        }
         log_info("["<< _pid << "]  OK OMP affinity subprocess for pid="<< getpid() );
      }

     */
    }

      std::vector<pid_t> get_all_tids(pid_t self) {

         std::vector<pid_t> ret;

         if(self==0)
           self = getpid();

         DIR *proc_dir;
         char dirname[100];
         snprintf(dirname, sizeof dirname, "/proc/%d/task", self);
         //proc_dir = opendir(dirname);
         proc_dir = opendir("/proc/self/task");

         if (proc_dir)
         {
        /* /proc available, iterate through tasks... */
         struct dirent *entry;
         while ((entry = readdir(proc_dir)) != NULL)
         {
            if(entry->d_name[0] == '.')
                continue;

             int tid = atoi(entry->d_name);
             ret.push_back(tid);

            /* ... (do stuff with tid) ... */
         }

        closedir(proc_dir);
    }

         return ret;
      }


DGL_REGISTER_GLOBAL("dataloading.cpu_affinity._CAPI_SetOMPThreadAffinity")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  // AsyncTransfererRef ref = args[0];
  // int id = args[1];
  // NDArray arr = ref->Wait(id);
  // *rv = arr;

    IdArray array = args[0];
    const auto& cores = array.ToVector<int64_t>();
    PinOMP2Cores(cores);

});

DGL_REGISTER_GLOBAL("dataloading.cpu_affinity._CAPI_FakeGompAffinity")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  // AsyncTransfererRef ref = args[0];
  // int id = args[1];
  // NDArray arr = ref->Wait(id);
  // *rv = arr;

    IdArray array = args[0];
    const auto& cores = array.ToVector<int64_t>();
    FakeGompAfinity(cores);

});


  }
}