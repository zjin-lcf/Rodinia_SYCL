/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#include <string.h>
#include <string>

#include <CL/sycl.hpp>
#include "common.h"


#define BLOCK_SIZE 16

using namespace cl::sycl;
constexpr access::mode sycl_read_write = access::mode::read_write;


double gettime() {
	struct timeval t;
	gettimeofday(&t,NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}


static int do_verify = 0;
void lud_cuda(float *d_m, int matrix_dim);

static struct option long_options[] = {
	/* name, has_arg, flag, val */
	{"input", 1, NULL, 'i'},
	{"size", 1, NULL, 's'},
	{"verify", 0, NULL, 'v'},
	{0,0,0,0}
};


	int
main ( int argc, char *argv[] )
{
	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
	int matrix_dim = 32; /* default matrix_dim */
	int opt, option_index=0;
	func_ret_t ret;
	const char *input_file = NULL;
	float *m, *mm;
	stopwatch sw;

	while ((opt = getopt_long(argc, argv, "::vs:i:", 
					long_options, &option_index)) != -1 ) {
		switch(opt){
			case 'i':
				input_file = optarg;
				break;
			case 'v':
				do_verify = 1;
				break;
			case 's':
				matrix_dim = atoi(optarg);
				printf("Generate input matrix internally, size =%d\n", matrix_dim);
				// fprintf(stderr, "Currently not supported, use -i instead\n");
				// fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
				// exit(EXIT_FAILURE);
				break;
			case '?':
				fprintf(stderr, "invalid option\n");
				break;
			case ':':
				fprintf(stderr, "missing argument\n");
				break;
			default:
				fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
						argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	if ( (optind < argc) || (optind == 1)) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
	}	

	if (input_file) {
		printf("Reading matrix from file %s\n", input_file);
		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	} 

	else if (matrix_dim) {
		printf("Creating matrix internally size=%d\n", matrix_dim);
		ret = create_matrix(&m, matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
			exit(EXIT_FAILURE);
		}
	}

	else {
		printf("No input file specified!\n");
		exit(EXIT_FAILURE);
	}

	if (do_verify){
		printf("Before LUD\n");
		// print_matrix(m, matrix_dim);
		matrix_duplicate(m, &mm, matrix_dim);
	}


	/* beginning of timing point */
	stopwatch_start(&sw);

	{ // SYCL scope

#ifdef USE_GPU
	gpu_selector dev_sel;
#else
	cpu_selector dev_sel;
#endif
	queue q(dev_sel);

	const property_list props = property::buffer::use_host_ptr();
	buffer<float, 1> d_m (m, matrix_dim*matrix_dim, props);

	int offset;
	int i=0;
	for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {

		offset = i;  // add the offset 
		int global_work1[3]  = {BLOCK_SIZE, 1, 1};
		int local_work1[3]  = {BLOCK_SIZE, 1, 1};

		q.submit([&](handler& cgh) {

			auto m_acc = d_m.get_access<sycl_read_write>(cgh);
			accessor <float, 1, sycl_read_write, access::target::local> shadow (BLOCK_SIZE * BLOCK_SIZE, cgh);

			cgh.parallel_for<class diagonal>(
					nd_range<1>(range<1>(global_work1[0]), 
						range<1>(local_work1[0])), [=] (nd_item<1> item) {
#include "kernel_lud_diagonal.sycl"

					});
			});


		int global_work2[3] = {BLOCK_SIZE * 2 * ((matrix_dim-i)/BLOCK_SIZE-1), 1, 1};
		int local_work2[3]  = {BLOCK_SIZE * 2, 1, 1};

		q.submit([&](handler& cgh) {

			auto m_acc = d_m.get_access<sycl_read_write>(cgh);
			accessor <float, 1, sycl_read_write, access::target::local> dia (BLOCK_SIZE * BLOCK_SIZE, cgh);
			accessor <float, 1, sycl_read_write, access::target::local> peri_row (BLOCK_SIZE * BLOCK_SIZE, cgh);
			accessor <float, 1, sycl_read_write, access::target::local> peri_col (BLOCK_SIZE * BLOCK_SIZE, cgh);

			cgh.parallel_for<class peri>(
					nd_range<1>(range<1>(global_work2[0]), 
						range<1>(local_work2[0])), [=] (nd_item<1> item) {
#include "kernel_lud_perimeter.sycl"

					});
			});


		int global_work3[3] = {BLOCK_SIZE * ((matrix_dim-i)/BLOCK_SIZE-1), BLOCK_SIZE * ((matrix_dim-i)/BLOCK_SIZE-1), 1};
		int local_work3[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};

		q.submit([&](handler& cgh) {

			auto m_acc = d_m.get_access<sycl_read_write>(cgh);
			accessor <float, 1, sycl_read_write, access::target::local> peri_col (BLOCK_SIZE * BLOCK_SIZE, cgh);
			accessor <float, 1, sycl_read_write, access::target::local> peri_row (BLOCK_SIZE * BLOCK_SIZE, cgh);

			cgh.parallel_for<class internal>(
					nd_range<2>(range<2>(global_work3[0],global_work3[1]), 
						range<2>(local_work3[0], local_work3[1])), [=] (nd_item<2> item) {
#include "kernel_lud_internal.sycl"

					});
			});
	} // for

	offset = i;  // add the offset 
	int global_work1[3]  = {BLOCK_SIZE, 1, 1};
	int local_work1[3]  = {BLOCK_SIZE, 1, 1};

	q.submit([&](handler& cgh) {

		auto m_acc = d_m.get_access<sycl_read_write>(cgh);
		accessor <float, 1, sycl_read_write, access::target::local> shadow (BLOCK_SIZE * BLOCK_SIZE, cgh);

		cgh.parallel_for<class diagonal2>(
				nd_range<1>(range<1>(global_work1[0]), 
					range<1>(local_work1[0])), [=] (nd_item<1> item) {
#include "kernel_lud_diagonal.sycl"

				});
		});

	} // SYCL scope

	/* end of timing point */
	stopwatch_stop(&sw);
	printf("Device offloading time (s): %lf\n", get_interval_by_sec(&sw));

	if (do_verify){
		printf("After LUD\n");
		// print_matrix(m, matrix_dim);
		printf(">>>Verify<<<<\n");
		lud_verify(mm, m, matrix_dim); 
		free(mm);
	}

	free(m);

}				

/* ----------  end of function main  ---------- */


