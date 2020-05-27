#include <CL/sycl.hpp>									// (in directory specified to compiler)		needed by OpenCL types and functions
#include "./../main.h"								// (in main directory)						needed to recognized input parameters
#include "./../util/avi/avilib.h"					// (in directory)							needed by avi functions
#include "./../util/avi/avimod.h"					// (in directory)							needed by avi functions
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;
constexpr access::mode sycl_read       = access::mode::read;
constexpr access::mode sycl_write      = access::mode::write;
constexpr access::mode sycl_read_write = access::mode::read_write;
//constexpr access::mode sycl_atomic     = access::mode::atomic;

void 
kernel_gpu_wrapper(	params_common common,
							int* endoRow,
							int* endoCol,
							int* tEndoRowLoc,
							int* tEndoColLoc,
							int* epiRow,
							int* epiCol,
							int* tEpiRowLoc,
							int* tEpiColLoc,
							avi_t* frames)
{

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    auto exception_handler = [] (cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
    std::rethrow_exception(e);
      } catch(cl::sycl::exception const& e) {
    std::cout << "Caught asynchronous SYCL exception:\n"
          << e.what() << std::endl;
      }
    }
  };
  cl::sycl::queue q(dev_sel, exception_handler);
  auto device = q.get_device();

#ifdef DEBUG
  auto deviceName = device.get_info<info::device::name>();
  std::cout << " Device Name: " << deviceName << std::endl;
  auto platformName =
      device.get_platform().get_info<info::platform::name>();
  std::cout << " Platform Name " << platformName << std::endl;
#endif

  const property_list props = property::buffer::use_host_ptr();

	// common
	//printf("tSize is %d, sSize is %d\n", common.tSize, common.sSize);
	common.in_rows = common.tSize + 1 + common.tSize;
	common.in_cols = common.in_rows;
	common.in_elem = common.in_rows * common.in_cols;
	common.in_mem = sizeof(fp) * common.in_elem;

	//==================================================50
	// endo points templates
	//==================================================50

  buffer<fp,1> d_endoT(common.in_elem * common.endoPoints);
  //printf("%d\n", common.in_elem * common.endoPoints);

	//==================================================50
	// epi points templates
	//==================================================50

  buffer<fp,1> d_epiT(common.in_elem * common.epiPoints);
  //printf("%d\n", common.in_elem * common.epiPoints);

	//====================================================================================================100
	// 	AREA AROUND POINT		FROM	FRAME	(LOCAL)
	//====================================================================================================100

	// common
	common.in2_rows = common.sSize + 1 + common.sSize;
	common.in2_cols = common.in2_rows;
	common.in2_elem = common.in2_rows * common.in2_cols;
	common.in2_mem = sizeof(fp) * common.in2_elem;

	// unique
  buffer<fp,1> d_in2(common.in2_elem * common.allPoints);
  //printf("%d\n", common.in2_elem * common.allPoints);

	//====================================================================================================100
	// 	CONVOLUTION	(LOCAL)
	//====================================================================================================100

	// common
	common.conv_rows = common.in_rows + common.in2_rows - 1;												// number of rows in I
	common.conv_cols = common.in_cols + common.in2_cols - 1;												// number of columns in I
	common.conv_elem = common.conv_rows * common.conv_cols;													// number of elements
	common.conv_mem = sizeof(fp) * common.conv_elem;
	common.ioffset = 0;
	common.joffset = 0;

	// unique
  buffer<fp,1> d_conv(common.conv_elem * common.allPoints);
  //printf("%d\n", common.conv_elem * common.allPoints);

	//====================================================================================================100
	// 	CUMULATIVE SUM	(LOCAL)
	//====================================================================================================100

	//==================================================50
	// 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
	//==================================================50

	// common
	common.in2_pad_add_rows = common.in_rows;
	common.in2_pad_add_cols = common.in_cols;

	common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
	common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
	common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
	common.in2_pad_cumv_mem = sizeof(fp) * common.in2_pad_cumv_elem;

	// unique
  buffer<fp,1> d_in2_pad_cumv(common.in2_pad_cumv_elem * common.allPoints);
  //printf("%d\n", common.in2_pad_cumv_elem * common.allPoints);

	//==================================================50
	// 	SELECTION
	//==================================================50

	// common
	common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;													// (1 to n+1)
	common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
	common.in2_pad_cumv_sel_collow = 1;
	common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
	common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
	common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
	common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
	common.in2_pad_cumv_sel_mem = sizeof(fp) * common.in2_pad_cumv_sel_elem;

	// unique
  buffer<fp,1> d_in2_pad_cumv_sel(common.in2_pad_cumv_sel_elem * common.allPoints);
  //printf("%d\n", common.in2_pad_cumv_sel_elem * common.allPoints);

	//==================================================50
	// 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//==================================================50

	// common
	common.in2_pad_cumv_sel2_rowlow = 1;
	common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
	common.in2_pad_cumv_sel2_collow = 1;
	common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
	common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
	common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
	common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
	common.in2_sub_cumh_mem = sizeof(fp) * common.in2_sub_cumh_elem;

	// unique
  buffer<fp,1> d_in2_sub_cumh(common.in2_sub_cumh_elem * common.allPoints);
  //printf("%d\n", common.in2_sub_cumh_elem * common.allPoints);

	//==================================================50
	// 	SELECTION
	//==================================================50

	// common
	common.in2_sub_cumh_sel_rowlow = 1;
	common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
	common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
	common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
	common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
	common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
	common.in2_sub_cumh_sel_mem = sizeof(fp) * common.in2_sub_cumh_sel_elem;

	// unique
  buffer<fp,1> d_in2_sub_cumh_sel(common.in2_sub_cumh_sel_elem * common.allPoints);
  //printf("%d\n", common.in2_sub_cumh_sel_elem * common.allPoints);

	//==================================================50
	//	SELECTION 2, SUBTRACTION
	//==================================================50

	// common
	common.in2_sub_cumh_sel2_rowlow = 1;
	common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel2_collow = 1;
	common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
	common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
	common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
	common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
	common.in2_sub2_mem = sizeof(fp) * common.in2_sub2_elem;

	// unique
  buffer<fp,1> d_in2_sub2(common.in2_sub2_elem * common.allPoints);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);

	//====================================================================================================100
	//	CUMULATIVE SUM 2	(LOCAL)
	//====================================================================================================100

	//==================================================50
	//	MULTIPLICATION
	//==================================================50

	// common
	common.in2_sqr_rows = common.in2_rows;
	common.in2_sqr_cols = common.in2_cols;
	common.in2_sqr_elem = common.in2_elem;
	common.in2_sqr_mem = common.in2_mem;

	// unique
  buffer<fp,1> d_in2_sqr(common.in2_elem * common.allPoints);
  //printf("%d\n", common.in2_elem * common.allPoints);

	//==================================================50
	//	SELECTION 2, SUBTRACTION
	//==================================================50

	// common
	common.in2_sqr_sub2_rows = common.in2_sub2_rows;
	common.in2_sqr_sub2_cols = common.in2_sub2_cols;
	common.in2_sqr_sub2_elem = common.in2_sub2_elem;
	common.in2_sqr_sub2_mem = common.in2_sub2_mem;

	// unique
  buffer<fp,1> d_in2_sqr_sub2(common.in2_sub2_elem * common.allPoints);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);

	//====================================================================================================100
	//	FINAL	(LOCAL)
	//====================================================================================================100

	// common
	common.in_sqr_rows = common.in_rows;
	common.in_sqr_cols = common.in_cols;
	common.in_sqr_elem = common.in_elem;
	common.in_sqr_mem = common.in_mem;

	// unique
  buffer<fp,1> d_in_sqr(common.in_elem * common.allPoints);
  //printf("%d\n", common.in_elem * common.allPoints);

	//====================================================================================================100
	//	TEMPLATE MASK CREATE	(LOCAL)
	//====================================================================================================100

	// common
	common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
	common.tMask_cols = common.tMask_rows;
	common.tMask_elem = common.tMask_rows * common.tMask_cols;
	common.tMask_mem = sizeof(fp) * common.tMask_elem;

	// unique
  buffer<fp,1> d_tMask(common.tMask_elem * common.allPoints);
  //printf("%d\n", common.tMask_elem * common.allPoints);

	//====================================================================================================100
	//	POINT MASK INITIALIZE	(LOCAL)
	//====================================================================================================100

	// common
	common.mask_rows = common.maxMove;
	common.mask_cols = common.mask_rows;
	common.mask_elem = common.mask_rows * common.mask_cols;
	common.mask_mem = sizeof(fp) * common.mask_elem;

	//====================================================================================================100
	//	MASK CONVOLUTION	(LOCAL)
	//====================================================================================================100

	// common
	common.mask_conv_rows = common.tMask_rows;												// number of rows in I
	common.mask_conv_cols = common.tMask_cols;												// number of columns in I
	common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;												// number of elements
	common.mask_conv_mem = sizeof(fp) * common.mask_conv_elem;
	common.mask_conv_ioffset = (common.mask_rows-1)/2;
	if((common.mask_rows-1) % 2 > 0.5){
		common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
	}
	common.mask_conv_joffset = (common.mask_cols-1)/2;
	if((common.mask_cols-1) % 2 > 0.5){
		common.mask_conv_joffset = common.mask_conv_joffset + 1;
	}


  //printf("common.endPoints=%d\n", common.endoPoints); // 20 
  //printf("common.epiPoints=%d\n", common.epiPoints); // 31
  //printf("common.in_elem=%d\n", common.in_elem);
  //printf("common.endo_mem=%d\n", common.endo_mem); // 80
  //printf("common.epi_mem=%d\n", common.epi_mem); // 124
  //
  buffer<params_common,1> d_common(&common, 1, props); // range is 1 ?
  buffer<int,1> d_endoRow(endoRow, common.endoPoints, props);
  d_endoRow.set_final_data(nullptr);
  buffer<int,1> d_endoCol(endoCol, common.endoPoints, props);
  d_endoCol.set_final_data(nullptr);
  buffer<int,1> d_tEndoRowLoc(tEndoRowLoc, common.endoPoints * common.no_frames, props);
  buffer<int,1> d_tEndoColLoc(tEndoColLoc, common.endoPoints * common.no_frames, props);
  buffer<int,1> d_epiRow(epiRow, common.epiPoints, props);
  d_epiRow.set_final_data(nullptr);
  buffer<int,1> d_epiCol(epiCol, common.epiPoints, props);
  d_epiCol.set_final_data(nullptr);
  buffer<int,1> d_tEpiRowLoc(tEpiRowLoc, common.epiPoints * common.no_frames, props);
  buffer<int,1> d_tEpiColLoc(tEpiColLoc, common.epiPoints * common.no_frames, props);


  buffer<fp,1> d_mask_conv(common.mask_conv_elem * common.allPoints);
  d_mask_conv.set_final_data(nullptr);
  //printf("%d\n", common.mask_conv_elem * common.allPoints);
  buffer<fp,1> d_in_mod_temp(common.in_elem * common.allPoints);
  d_in_mod_temp.set_final_data(nullptr);
  //printf("%d\n", common.in_elem * common.allPoints);
  buffer<fp,1> d_in_partial_sum(common.in_cols * common.allPoints);
  d_in_partial_sum.set_final_data(nullptr);
  //printf("%d\n", common.in_cols * common.allPoints);
  buffer<fp,1> d_in_sqr_partial_sum(common.in_sqr_rows * common.allPoints);
  d_in_sqr_partial_sum.set_final_data(nullptr);
  //printf("%d\n", common.in_sqr_rows * common.allPoints);
  buffer<fp,1> d_par_max_val(common.mask_conv_rows * common.allPoints);
  d_par_max_val.set_final_data(nullptr);
  //printf("%d\n", common.mask_conv_rows * common.allPoints);
  buffer<int,1> d_par_max_coo( common.mask_conv_rows * common.allPoints);
  d_par_max_coo.set_final_data(nullptr);
  buffer<fp,1> d_in_final_sum(common.allPoints);
  d_in_final_sum.set_final_data(nullptr);
  buffer<fp,1> d_in_sqr_final_sum(common.allPoints);
  d_in_sqr_final_sum.set_final_data(nullptr);
  buffer<fp,1> d_denomT(common.allPoints);
  d_denomT.set_final_data(nullptr);
#ifdef TEST_CHECKSUM
  buffer<fp,1> d_checksum(CHECK);
  d_checksum.set_final_data(nullptr);
  //printf("%d\n", CHECK);
#endif

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	// All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
	size_t local_work_size = NUMBER_THREADS;
	size_t global_work_size = common.allPoints * local_work_size;

#ifdef DEBUG
	printf("# of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n",(int)(global_work_size/local_work_size), (int)local_work_size);
#endif


	printf("frame progress: ");
	fflush(NULL);

	//====================================================================================================100
	//	LAUNCH
	//====================================================================================================100

	// variables
	fp* frame;
	int frame_no;

  buffer<fp,1> d_frame(common.frame_elem);

	for(frame_no=0; frame_no<common.frames_processed; frame_no++){

		//==================================================50
		//	get and write current frame to GPU buffer
		//==================================================50

		// Extract a cropped version of the first frame from the video file
		frame = get_frame(	frames,								// pointer to video file
							frame_no,							// number of frame that needs to be returned
							0,									// cropped?
							0,									// scaled?
							1);									// converted

		// copy frame to GPU memory
    //buffer<fp,1> d_frame(frame, common.frame_elem, props);
    q.submit ([&](handler &cgh) {
        auto d_frame_acc = d_frame.get_access<sycl_write>(cgh);
        cgh.copy(frame, d_frame_acc);
    });
    //buffer<int,1> d_frame_no(&frame_no, 1, props);

		//==================================================50
		//	launch kernel
		//==================================================50
    q.submit ([&](handler &cgh) {

        // read access
		      auto d_common_acc = d_common.get_access<sycl_read>(cgh);
		      auto d_frame_acc = d_frame.get_access<sycl_read>(cgh);
		      //auto d_frame_no_acc = d_frame_no.get_access<sycl_read>(cgh);

		      auto d_endoRow_acc = d_endoRow.get_access<sycl_read_write>(cgh);
		      auto d_endoCol_acc = d_endoCol.get_access<sycl_read_write>(cgh);
		      auto d_tEndoRowLoc_acc = d_tEndoRowLoc.get_access<sycl_read_write>(cgh);
		      auto d_tEndoColLoc_acc = d_tEndoColLoc.get_access<sycl_read_write>(cgh);
		      auto d_epiRow_acc = d_epiRow.get_access<sycl_read_write>(cgh);
		      auto d_epiCol_acc = d_epiCol.get_access<sycl_read_write>(cgh);
		      auto d_tEpiRowLoc_acc = d_tEpiRowLoc.get_access<sycl_read_write>(cgh);
		      auto d_tEpiColLoc_acc = d_tEpiColLoc.get_access<sycl_read_write>(cgh);

          // read/write or write access depending on checksum
#ifdef TEST_CHECKSUM
constexpr access::mode sycl_access_mode     = sycl_read_write;
#else
constexpr access::mode sycl_access_mode     = sycl_write;
#endif
		      auto d_endoT_acc = d_endoT.get_access<sycl_access_mode>(cgh);
		      auto d_epiT_acc = d_epiT.get_access<sycl_access_mode>(cgh);
		      auto d_in2_all_acc = d_in2.get_access<sycl_access_mode>(cgh);
		      auto d_conv_all_acc = d_conv.get_access<sycl_access_mode>(cgh);
		      auto d_in2_pad_cumv_all_acc = d_in2_pad_cumv.get_access<sycl_access_mode>(cgh);
		      auto d_in2_pad_cumv_sel_all_acc = d_in2_pad_cumv_sel.get_access<sycl_access_mode>(cgh);
		      auto d_in2_sub_cumh_all_acc = d_in2_sub_cumh.get_access<sycl_access_mode>(cgh);
		      auto d_in2_sub_cumh_sel_all_acc = d_in2_sub_cumh_sel.get_access<sycl_access_mode>(cgh);
		      auto d_in2_sub2_all_acc = d_in2_sub2.get_access<sycl_access_mode>(cgh);
		      auto d_in2_sqr_all_acc = d_in2_sqr.get_access<sycl_access_mode>(cgh);
		      auto d_in2_sqr_sub2_all_acc = d_in2_sqr_sub2.get_access<sycl_access_mode>(cgh);
		      auto d_in_sqr_all_acc = d_in_sqr.get_access<sycl_access_mode>(cgh);
		      auto d_tMask_all_acc = d_tMask.get_access<sycl_access_mode>(cgh);
		      auto d_mask_conv_all_acc = d_mask_conv.get_access<sycl_access_mode>(cgh);
		      auto d_in_mod_temp_all_acc = d_in_mod_temp.get_access<sycl_access_mode>(cgh);
		      auto d_in_partial_sum_all_acc = d_in_partial_sum.get_access<sycl_access_mode>(cgh);
		      auto d_in_sqr_partial_sum_all_acc = d_in_sqr_partial_sum.get_access<sycl_access_mode>(cgh);
		      auto par_max_val_all_acc = d_par_max_val.get_access<sycl_access_mode>(cgh);
		      auto par_max_coo_all_acc = d_par_max_coo.get_access<sycl_access_mode>(cgh);
		      auto d_in_final_sum_all_acc = d_in_final_sum.get_access<sycl_access_mode>(cgh);
		      auto d_in_sqr_final_sum_all_acc = d_in_sqr_final_sum.get_access<sycl_access_mode>(cgh);
		      auto denomT_all_acc = d_denomT.get_access<sycl_access_mode>(cgh);
#ifdef TEST_CHECKSUM
		      auto checksum_acc = d_checksum.get_access<sycl_access_mode>(cgh);
#endif

		      cgh.parallel_for<class heartwall>(
              nd_range<1>(range<1>(global_work_size), range<1>(local_work_size)), [=] (nd_item<1> item) {
              #include "kernel.sycl"
            });
    });

  try {
    q.wait_and_throw();
  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n"
          << e.what() << std::endl;
  }

		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		free(frame);

		//==================================================50
		//	print frame progress
		//==================================================50

		// print frame progress
		printf("%d ", frame_no);
		fflush(NULL);

		//==================================================50
		//	DISPLAY CHECKSUM (TESTING)
		//==================================================50

#ifdef TEST_CHECKSUM
    auto checksum_host = d_checksum.get_access<sycl_read>();
		printf("CHECKSUM:\n");
		for(i=0; i<CHECK; i++){
				printf("i=%d checksum=%f\n", i, checksum_host[i]);
		}
		printf("\n\n");
#endif

		//==================================================50
		//	End
		//==================================================50

	}

	//====================================================================================================100
	//	PRINT FRAME PROGRESS END
	//====================================================================================================100

	printf("\n");
	fflush(NULL);

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
