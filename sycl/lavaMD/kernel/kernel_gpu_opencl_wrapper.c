#include <string.h>

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer
#include "common.h"

//======================================================================================================================================================150
//	KERNEL_GPU_OPENCL_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_opencl_wrapper.h"				// (in the current directory)

//========================================================================================================================================================================================================200
//	KERNEL_GPU_OPENCL_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

  void 
kernel_gpu_opencl_wrapper(	par_str par_cpu,
    dim_str dim_cpu,
    box_str* box_cpu,
    FOUR_VECTOR* rv_cpu,
    fp* qv_cpu,
    FOUR_VECTOR* fv_cpu)
{

  //======================================================================================================================================================150
  //	CPU VARIABLES
  //======================================================================================================================================================150

  // timer
  long long time0;
  //long long time1;
  //long long time2;
  //long long time3;
  //long long time4;
  //long long time5;
  long long time6;

  time0 = get_time();

  { // SYCL scope

    //======================================================================================================================================================150
    //	GPU SETUP
    //======================================================================================================================================================150
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    //====================================================================================================100
    //	EXECUTION PARAMETERS
    //====================================================================================================100

    size_t local_work_size[1];
    local_work_size[0] = NUMBER_THREADS;
    size_t global_work_size[1];
    global_work_size[0] = dim_cpu.number_boxes * local_work_size[0];

#ifdef DEBUG
    printf("# of blocks = %lu, # of threads/block = %lu (ensure that device can handle)\n", 
        global_work_size[0]/local_work_size[0], local_work_size[0]);
#endif

    const property_list props = property::buffer::use_host_ptr();
    //======================================================================================================================================================150
    //	GPU MEMORY				(MALLOC)
    //======================================================================================================================================================150

    //====================================================================================================100
    //	GPU MEMORY				COPY IN
    //====================================================================================================100

    //==================================================50
    //	boxes
    //==================================================50
    buffer<box_str, 1> d_box_gpu(box_cpu, dim_cpu.box_mem/sizeof(box_str), props);

    //==================================================50
    //	rv
    //==================================================50

    buffer<FOUR_VECTOR, 1> d_rv_gpu(rv_cpu, dim_cpu.space_mem/sizeof(FOUR_VECTOR), props);
    //==================================================50
    //	qv
    //==================================================50
    buffer<fp, 1> d_qv_gpu(qv_cpu, dim_cpu.space_mem2/sizeof(fp), props);

    //====================================================================================================100
    //	GPU MEMORY				COPY (IN & OUT)
    //====================================================================================================100

    //==================================================50
    //	fv
    //==================================================50

    buffer<FOUR_VECTOR, 1> d_fv_gpu(fv_cpu, dim_cpu.space_mem/sizeof(FOUR_VECTOR), props);
    //======================================================================================================================================================150
    //	KERNEL
    //======================================================================================================================================================150
    q.submit([&](handler& cgh) {

        auto d_box_gpu_acc = d_box_gpu.get_access<sycl_read>(cgh);
        auto d_rv_gpu_acc = d_rv_gpu.get_access<sycl_read>(cgh);
        auto d_qv_gpu_acc = d_qv_gpu.get_access<sycl_read>(cgh);
        auto d_fv_gpu_acc = d_fv_gpu.get_access<sycl_read_write>(cgh);

        accessor <FOUR_VECTOR, 1, sycl_read_write, access::target::local> rA_shared (100, cgh);
        accessor <FOUR_VECTOR, 1, sycl_read_write, access::target::local> rB_shared (100, cgh);
        accessor <fp, 1, sycl_read_write, access::target::local> qB_shared (100, cgh);

        cgh.parallel_for<class kernel_lavamd>(
            nd_range<1>(range<1>(global_work_size[0]), 
                        range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "kernel.sycl"
            });
        });

  } // SYCL scope

  time6 = get_time();

  //==================================================50
  //	fv
  //==================================================50

#ifdef DEBUG
  // (enable for testing purposes only - prints some range of output, make sure not to initialize input in main.c with random numbers for comparison across runs)
  int g;
  int offset = 395;
  for(g=0; g<10; g++){
    printf("g=%d %f, %f, %f, %f\n", \
        g, fv_cpu[offset+g].v, fv_cpu[offset+g].x, fv_cpu[offset+g].y, fv_cpu[offset+g].z);
  }
#endif



  //======================================================================================================================================================150
  //	DISPLAY TIMING
  //======================================================================================================================================================150

  //printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

  //printf("%15.12f s, %15.12f : GPU: SET DEVICE / DRIVER INIT\n",	(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
  //printf("%15.12f s, %15.12f : GPU MEM: ALO\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
  //printf("%15.12f s, %15.12f : GPU MEM: COPY IN\n",					(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);

  //printf("%15.12f s, %15.12f : GPU: KERNEL\n",						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);

  //printf("%15.12f s, %15.12f : GPU MEM: COPY OUT and FREE\n",				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);

  printf("Total device offloading time:\n");
  printf("%.12f s\n", 												(float) (time6-time0) / 1000000);

}

//========================================================================================================================================================================================================200
//	END KERNEL_GPU_OPENCL_WRAPPER FUNCTION
//========================================================================================================================================================================================================200
