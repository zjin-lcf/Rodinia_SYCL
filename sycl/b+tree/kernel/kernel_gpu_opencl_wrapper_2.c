
#include <string.h>									// (in directory known to compiler)			needed by memset

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/timer/timer.h"					// (in directory provided here)

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_opencl_wrapper_2.h"			// (in directory provided here)

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_opencl_wrapper_2(
    knode *knodes,
    long knodes_elem,
    long knodes_mem,  // not length in byte

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    long *lastKnode,
    long *offset_2,
    int *start,
    int *end,
    int *recstart,
    int *reclength)
{

  //======================================================================================================================================================150
  //	CPU VARIABLES
  //======================================================================================================================================================150

  // timer
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;

  time0 = get_time();

  { // SYCL scopde
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();

    buffer<knode,1> knodesD (knodes, knodes_mem, props);
    buffer<long,1> currKnodeD (currKnode, count, props);
    buffer<long,1> offsetD (offset, count, props);
    buffer<long,1> lastKnodeD (lastKnode, count, props);
    buffer<long,1> offset_2D (offset_2, count, props);
    buffer<int,1> startD (start, count, props);
    buffer<int,1> endD (end, count, props);
    buffer<int,1> ansDStart (recstart, count, props);
    buffer<int,1> ansDLength (reclength, count, props);
    buffer<long,1> heightD (&maxheight, 1, props);
    buffer<long,1> knodes_elemD (&knodes_elem, 1, props);

    //======================================================================================================================================================150
    // findRangeK kernel
    //======================================================================================================================================================150

    //====================================================================================================100
    //	Execution Parameters
    //====================================================================================================100

    size_t local_work_size[1];
#ifdef USE_GPU
    local_work_size[0] = order < 256 ? order : 256;
#else
    local_work_size[0] = order < 1024 ? order : 1024;
#endif
    size_t global_work_size[1];
    global_work_size[0] = count * local_work_size[0];

    printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

    q.submit([&](handler& cgh) {

        auto knodesD_acc = knodesD.get_access<sycl_read>(cgh);
        auto currKnodeD_acc = currKnodeD.get_access<sycl_read_write>(cgh);
        auto lastKnodeD_acc = lastKnodeD.get_access<sycl_read_write>(cgh);
        auto offsetD_acc = offsetD.get_access<sycl_read_write>(cgh);
        auto offset_2D_acc = offset_2D.get_access<sycl_read_write>(cgh);
        auto startD_acc = startD.get_access<sycl_read>(cgh);
        auto endD_acc = endD.get_access<sycl_read>(cgh);
        auto RecstartD_acc = ansDStart.get_access<sycl_read_write>(cgh);
        auto ReclenD_acc = ansDLength.get_access<sycl_write>(cgh);
        auto height_acc = heightD.get_access<sycl_read>(cgh);

        cgh.parallel_for<class findRangeK>(
            nd_range<1>(range<1>(global_work_size[0]),
              range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "findRangeK.sycl"
            });
        });

  } // SYCL scopde
  time6 = get_time();

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("recstart[%d] = %d\n", i, recstart[i]);
  for (int i = 0; i < count; i++)
    printf("reclength[%d] = %d\n", i, reclength[i]);
#endif

  //======================================================================================================================================================150
  //	DISPLAY TIMING
  //======================================================================================================================================================150

  printf("Total time:\n");
  printf("%.12f s\n", (float) (time6-time0) / 1000000);

  //======================================================================================================================================================150
  //	END
  //======================================================================================================================================================150

}

