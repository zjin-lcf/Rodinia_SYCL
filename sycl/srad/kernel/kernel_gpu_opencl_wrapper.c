//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h"								// (in the main program folder)

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>									// (in path known to compiler)	needed by printf
#include <string.h>									// (in path known to compiler)	needed by strlen
#include <iostream>


//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "common.h"

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

//#include "./kernel_gpu_opencl_wrapper.h"			// (in directory)

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_opencl_wrapper(	fp* image,											// input image
							int Nr,												// IMAGE nbr of rows
							int Nc,												// IMAGE nbr of cols
							long Ne,											// IMAGE nbr of elem
							int niter,											// nbr of iterations
							fp lambda,											// update step size
							long NeROI,											// ROI nbr of elements
							int* iN,
							int* iS,
							int* jE,
							int* jW,
							int iter)											// primary loop
							
{

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	COMMON VARIABLES
	//====================================================================================================100

	//====================================================================================================100
	//	GET PLATFORMS (Intel, AMD, NVIDIA, based on provided library), SELECT ONE
	//====================================================================================================100


	//====================================================================================================100
	//	TRIGGERING INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	// cudaThreadSynchronize();		// the above does it

	//======================================================================================================================================================150
	// 	GPU VARIABLES
	//======================================================================================================================================================150

	// CUDA kernel execution parameters
	int blocks_x;

	//======================================================================================================================================================150
	// 	ALLOCATE MEMORY IN GPU
	//======================================================================================================================================================150

	//====================================================================================================100
	// common memory size
	//====================================================================================================100

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);


#ifdef DEBUG
try {
#endif
	const property_list props = property::buffer::use_host_ptr();
	buffer<fp,1> d_dN (Ne);
	buffer<fp,1> d_dS (Ne);
	buffer<fp,1> d_dW (Ne);
	buffer<fp,1> d_dE (Ne);
	buffer<fp,1> d_c (Ne);
	buffer<fp,1> d_sums (Ne);
	buffer<fp,1> d_sums2 (Ne);
	buffer<fp,1> d_I (image, Ne, props);
	buffer<int,1> d_iN (iN, Nr, props);
	buffer<int,1> d_iS (iS, Nr, props);
	buffer<int,1> d_jE (jE, Nc, props);
	buffer<int,1> d_jW (jW, Nc, props);

	//======================================================================================================================================================150
	// 	KERNEL EXECUTION PARAMETERS
	//======================================================================================================================================================150

	// threads
	size_t local_work_size[1];
	local_work_size[0] = NUMBER_THREADS;

	// workgroups
	int blocks_work_size;
	size_t global_work_size[1];
	blocks_x = Ne/(int)local_work_size[0];
	if (Ne % (int)local_work_size[0] != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;																	
	}
	blocks_work_size = blocks_x;
	global_work_size[0] = blocks_work_size * local_work_size[0];						// define the number of blocks in the grid

	printf("max # of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n",
		       	(int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

	//======================================================================================================================================================150
	// 	Extract Kernel - SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//======================================================================================================================================================150
	
#ifdef DEBUG
	for (long i = 0; i < 16; i++)
		printf("before extract: %f\n",image[i]);
	printf("\n");
#endif

        q.submit([&](handler& cgh) {
        auto d_I_acc = d_I.get_access<sycl_write>(cgh);
        cgh.parallel_for<class extract>(
            nd_range<1>(range<1>(global_work_size[0]), 
		        range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "kernel_extract.sycl"
            });
        });


	int blocks2_work_size;
	size_t global_work_size2[1];
	long no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;


	//======================================================================================================================================================150
	// 	COMPUTATION
	//======================================================================================================================================================150

	printf("Iterations Progress: ");

	// execute main loop
	for (iter=0; iter<niter; iter++){ // do for the number of iterations input parameter

		printf("%d ", iter);
		fflush(NULL);

		//====================================================================================================100
		// Prepare kernel
		//====================================================================================================100
		q.submit([&](handler& cgh) {
		auto d_I_acc = d_I.get_access<sycl_read>(cgh);
		auto d_sums_acc = d_sums.get_access<sycl_write>(cgh);
		auto d_sums2_acc = d_sums2.get_access<sycl_write>(cgh);  // updated every iteration
			cgh.parallel_for<class prepare>(
			    nd_range<1>(range<1>(global_work_size[0]), 
					range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "kernel_prepare.sycl"
			    });
			});

		// initial values
		blocks2_work_size = blocks_work_size;							// original number of blocks
		global_work_size2[0] = global_work_size[0];
		no = Ne;														// original number of sum elements
		mul = 1;														// original multiplier

		// loop
		while(blocks2_work_size != 0){
#ifdef DEBUG
	printf("max # of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n",
		       	(int)(global_work_size2[0]/local_work_size[0]), (int)local_work_size[0]);
#endif
			q.submit([&](handler& cgh) {
			auto d_sums_acc = d_sums.get_access<sycl_read_write>(cgh);
			auto d_sums2_acc = d_sums2.get_access<sycl_read_write>(cgh);  // updated every iteration
			accessor <fp, 1, sycl_read_write, access::target::local> d_psum (NUMBER_THREADS, cgh);
			accessor <fp, 1, sycl_read_write, access::target::local> d_psum2 (NUMBER_THREADS, cgh);

			cgh.parallel_for<class reduce>(
			    nd_range<1>(range<1>(global_work_size2[0]), 
					range<1>(local_work_size[0])), [=] (nd_item<1> item) {
 #include "kernel_reduce.sycl"
			    });
			});
      q.wait();  // required for correct sum results

			// update execution parameters
			no = blocks2_work_size;	
			if(blocks2_work_size == 1){
				blocks2_work_size = 0;
			}
			else{
				mul = mul * NUMBER_THREADS;										// update the increment
				blocks_x = blocks2_work_size/(int)local_work_size[0];			// number of blocks
				if (blocks2_work_size % (int)local_work_size[0] != 0){			// compensate for division remainder above by adding one grid
					blocks_x = blocks_x + 1;
				}
				blocks2_work_size = blocks_x;
				global_work_size2[0] = blocks2_work_size * (int)local_work_size[0];
			}
		} // while

		auto h_sums_acc = d_sums.get_access<sycl_read>();
		auto h_sums2_acc = d_sums2.get_access<sycl_read>();
		total = h_sums_acc[0]; 
		total2 = h_sums2_acc[0];

#ifdef DEBUG
		printf("total: %f total2: %f\n", total, total2);
#endif

		//====================================================================================================100
		// calculate statistics
		//====================================================================================================100
		
		meanROI	= total / (fp)(NeROI);										// gets mean (average) value of element in ROI
		meanROI2 = meanROI * meanROI;										//
		varROI = (total2 / (fp)(NeROI)) - meanROI2;							// gets variance of ROI								
		q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

		//====================================================================================================100
		// execute srad kernel
		//====================================================================================================100

		// set arguments that were uptaded in this loop
		q.submit([&](handler& cgh) {
		auto d_iN_acc = d_iN.get_access<sycl_read>(cgh);
		auto d_iS_acc = d_iS.get_access<sycl_read>(cgh);
		auto d_jW_acc = d_jW.get_access<sycl_read>(cgh);
		auto d_jE_acc = d_jE.get_access<sycl_read>(cgh);
		auto d_dN_acc = d_dN.get_access<sycl_write>(cgh);
		auto d_dS_acc = d_dS.get_access<sycl_write>(cgh);
		auto d_dW_acc = d_dW.get_access<sycl_write>(cgh);
		auto d_dE_acc = d_dE.get_access<sycl_write>(cgh);
		auto d_c_acc = d_c.get_access<sycl_write>(cgh);
		auto d_I_acc = d_I.get_access<sycl_read>(cgh);

		cgh.parallel_for<class srad>(
		    nd_range<1>(range<1>(global_work_size[0]), 
				range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "kernel_srad.sycl"
		    });
		});

		//====================================================================================================100
		// execute srad2 kernel
		//====================================================================================================100

		// launch kernel
		q.submit([&](handler& cgh) {
		//auto d_iN_acc = d_iN.get_access<sycl_read>(cgh);
		auto d_iS_acc = d_iS.get_access<sycl_read>(cgh);
		//auto d_jW_acc = d_jW.get_access<sycl_read>(cgh);
		auto d_jE_acc = d_jE.get_access<sycl_read>(cgh);
		auto d_dN_acc = d_dN.get_access<sycl_read>(cgh);
		auto d_dS_acc = d_dS.get_access<sycl_read>(cgh);
		auto d_dW_acc = d_dW.get_access<sycl_read>(cgh);
		auto d_dE_acc = d_dE.get_access<sycl_read>(cgh);
		auto d_c_acc = d_c.get_access<sycl_read>(cgh);
		auto d_I_acc = d_I.get_access<sycl_read_write>(cgh);

		cgh.parallel_for<class srad2>(
		    nd_range<1>(range<1>(global_work_size[0]), 
				range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "kernel_srad2.sycl"
		    });
		});



		//====================================================================================================100
		// End
		//====================================================================================================100

	}  // for

  // print a newline after the display of iteration numbers
	printf("\n");


	//======================================================================================================================================================150
	// 	Compress Kernel - SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//======================================================================================================================================================150

        q.submit([&](handler& cgh) {
        auto d_I_acc = d_I.get_access<sycl_read_write>(cgh);
        cgh.parallel_for<class compress>(
            nd_range<1>(range<1>(global_work_size[0]), 
		        range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "kernel_compress.sycl"
            });
        });
	// implicit copy back from device to *image 
#ifdef DEBUG
	auto h_I_acc = d_I.get_access<sycl_read>();
	for (long i = 0; i < 16; i++)
		printf("%f ",h_I_acc[i]);
	printf("\n");

    } catch (cl::sycl::exception e) {
    std::cout << e.what() << std::endl;
#ifdef __COMPUTECPP__
    std::cout << e.get_file_name() << std::endl;
    std::cout << e.get_line_number() << std::endl;
    std::cout << e.get_description() << std::endl;
    std::cout << e.get_cl_error_message() << std::endl;
    std::cout << e.get_cl_code() << std::endl;
#endif
    return;
  }
  catch (std::exception e) {
    std::cout << e.what() << std::endl;
    return;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return;
  }
#endif
}

