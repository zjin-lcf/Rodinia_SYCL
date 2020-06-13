/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA. Main file.
 *
 * Copyright (C) 2009 Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 *
 * Add the SYCL implementation - Zheming Jin 
 */

#include <sys/time.h>
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "stats_logger.h"
#include "load_data.h"
#include "cpuencode.h"
#include "common.h"

#include "scan.cpp"

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}
void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks=1);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, unsigned int *codewords, unsigned int* codewordlens);

int main(int argc, char* argv[]){
  unsigned int num_block_threads = 256;
  if (argc > 1)
    for (int i=1; i<argc; i++)
      runVLCTest(argv[i], num_block_threads);
  else 
    runVLCTest(NULL, num_block_threads, 1024);
  return 0;
}

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks) {
  unsigned int num_elements; //uint num_elements = num_blocks * num_block_threads; 
  unsigned int mem_size; //uint mem_size = num_elements * sizeof(int); 
  unsigned int symbol_type_size = sizeof(int);
  //////// LOAD DATA ///////////////
  double H; // entropy
  initParams(file_name, num_block_threads, num_blocks, num_elements, mem_size, symbol_type_size);
  printf("Parameters: num_elements: %d, num_blocks: %d, num_block_threads: %d\n----------------------------\n", 
      num_elements, num_blocks, num_block_threads);
  ////////LOAD DATA ///////////////
  uint	*sourceData =	(uint*) malloc(mem_size);
  uint	*destData   =	(uint*) malloc(mem_size);
  uint	*crefData   =	(uint*) malloc(mem_size);

  uint	*codewords	   = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
  uint	*codewordlens  = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);

  uint	*cw32 =		(uint*) malloc(mem_size);
  uint	*cw32len =	(uint*) malloc(mem_size);
  uint	*cw32idx =	(uint*) malloc(mem_size);

  uint	*cindex2=	(uint*) malloc(num_blocks*sizeof(int));

  memset(sourceData,   0, mem_size);
  memset(destData,     0, mem_size);
  memset(crefData,     0, mem_size);
  memset(cw32,         0, mem_size);
  memset(cw32len,      0, mem_size);
  memset(cw32idx,      0, mem_size);
  memset(codewords,    0, NUM_SYMBOLS*symbol_type_size);
  memset(codewordlens, 0, NUM_SYMBOLS*symbol_type_size);
  memset(cindex2, 0, num_blocks*sizeof(int));

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  cl::sycl::queue q(dev_sel);

  
  // loadData function also requires GPU offloading
  loadData(q, file_name, sourceData, codewords, codewordlens, num_elements, mem_size, H);

  // CPU ENCODER 
  unsigned int refbytesize;
  long long timer = get_time();
  cpu_vlc_encode((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
  float msec = (float)((get_time() - timer)/1000.0);
  printf("CPU Encoding time (CPU): %f (ms)\n", msec);
  printf("CPU Encoded to %d [B]\n", refbytesize);
  unsigned int num_ints = refbytesize/4 + ((refbytesize%4 ==0)?0:1);

  // Offloading ENCODER to a device
  timer = get_time();
  buffer<unsigned int, 1> d_sourceData (sourceData, num_elements);
  buffer<unsigned int, 1> d_destData (destData, num_elements);
  d_destData.set_final_data(nullptr);
  buffer<unsigned int, 1> d_destDataPacked (num_elements);

  // symbol_type_size is sizeof(int) in cuda
  buffer<unsigned int, 1> d_codewords (codewords, NUM_SYMBOLS);
  buffer<unsigned int, 1> d_codewordlens (codewordlens, NUM_SYMBOLS);

  buffer<unsigned int, 1> d_cw32 (num_elements);
  buffer<unsigned int, 1> d_cw32len (num_elements);
  buffer<unsigned int, 1> d_cw32idx (num_elements);
  buffer<unsigned int, 1> d_cindex (num_blocks);
  buffer<unsigned int, 1> d_cindex2 (num_blocks);

  unsigned int sm_size; 
  unsigned int NT = 10; //number of runs for each execution time

  //////////////////* SM64HUFF KERNEL *///////////////////////////////////
  sm_size			= num_block_threads;
#ifdef CACHECWLUT
  sm_size			= 2*NUM_SYMBOLS + num_block_threads;
#endif

  //dim3 grid_size(num_blocks,1,1);
  //dim3 block_size(num_block_threads, 1, 1);
  size_t global_work_size = num_blocks * num_block_threads;
  size_t local_work_size = num_block_threads;

  for (int i=0; i<NT; i++) {
    q.submit([&](handler& cgh) {
        auto data = d_sourceData.get_access<sycl_read>(cgh);
        auto gm_codewords = d_codewords.get_access<sycl_read>(cgh);
        auto gm_codewordlens = d_codewordlens.get_access<sycl_read>(cgh);
//#ifdef TESTING
        //auto cw32 = d_cw32.get_access<sycl_write>(cgh);
        //auto cw32len = d_cw32len.get_access<sycl_write>(cgh);
        //auto cw32idx = d_cw32idx.get_access<sycl_write>(cgh);
//#endif
        auto out = d_destData.get_access<sycl_write>(cgh);
        auto outidx = d_cindex.get_access<sycl_write>(cgh);

        accessor <unsigned int, 1, sycl_atomic, access::target::local> sm (sm_size, cgh);
        accessor <unsigned int, 1, sycl_read_write, access::target::local> kcmax (1, cgh);
        cgh.parallel_for<class vlc_encode_kernel_sm64huff>(
          nd_range<1>(range<1>(global_work_size), range<1>(local_work_size)), [=] (nd_item<1> item) {
#include "kernel_vlc_encode.sycl"
          });
        });
  }

  // Uncomment when we just measure the variable length encoding time
  //  CUDA_SAFE_CALL(cudaMemcpy(destData, d_destData,	mem_size,	cudaMemcpyDeviceToHost));
  //q.submit([&](handler& cgh) {
  //  auto d_destData_acc = d_destData.get_access<sycl_read>(cgh);
  //  cgh.copy(d_destData_acc, destData);
  //});
  //q.wait();

#ifdef DEBUG
  printf("After SM64HUFF:\n");
  auto h_destData_acc = d_destData.get_access<sycl_read>();
  for (int i = 0; i < num_elements; i++)
    printf("i=%d %u\n", i, h_destData_acc[i]);
  printf("\n");
#endif

#ifdef TESTING
  unsigned int num_scan_elements = num_blocks;
  preallocBlockSums(num_scan_elements);
  //cudaMemset(d_destDataPacked, 0, mem_size);
  q.submit([&](handler& cgh) {
      auto d_destDataPacked_acc = d_destDataPacked.get_access<sycl_write>(cgh);
      cgh.fill(d_destDataPacked_acc, 0u);
  });

  printf("Num_blocks to be passed to scan is %d.\n", num_scan_elements);

  prescanArray(q, d_cindex2, d_cindex, num_scan_elements);

#ifdef DEBUG
  printf("After prescanArray:\n");
  auto h_cindex2_acc = d_cindex2.get_access<sycl_read>();
  for (int i = 0; i < num_blocks; i++)
    printf("cindex2 i=%d %u\n", i, h_cindex2_acc[i]);
  printf("\n");
  auto h_cindex_acc = d_cindex.get_access<sycl_read>();
  for (int i = 0; i < num_blocks; i++)
    printf("cindex i=%d %u\n", i, h_cindex_acc[i]);
  printf("\n");
#endif


  //pack2<<< num_scan_elements/16, 16>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked, num_elements/num_scan_elements);
  unsigned int original_num_block_elements = num_elements/num_scan_elements;
  q.submit([&](handler& cgh) {
    auto srcData = d_destData.get_access<sycl_read>(cgh);
    auto cindex = d_cindex.get_access<sycl_read>(cgh);
    auto cindex2 = d_cindex2.get_access<sycl_read>(cgh);
    auto dstData = d_destDataPacked.get_access<sycl_atomic>(cgh);

    cgh.parallel_for<class pack2>(
      nd_range<1>(range<1>(num_scan_elements), range<1>(16)), [=] (nd_item<1> item) {
#include "kernel_pack2.sycl"
    });
  });

#ifdef DEBUG
  q.wait();
  printf("After pack2:\n");
  auto h_destDataPacked_acc = d_destDataPacked.get_access<sycl_read>();
  for (int i = 0; i < num_elements; i++)
    printf("i=%d %u\n", i, h_destDataPacked_acc[i]);
  printf("\n");
#endif
  //CUDA_SAFE_CALL(cudaMemcpy(destData, d_destDataPacked, mem_size, cudaMemcpyDeviceToHost));
  q.submit([&](handler& cgh) {
    auto d_destDataPacked_acc = d_destDataPacked.get_access<sycl_read>(cgh);
    cgh.copy(d_destDataPacked_acc, destData);
  });
  deallocBlockSums();
  q.wait();
  
  printf("Device offloading time: %f (ms)\n", (get_time() - timer)/1000.f);
  printf("This includes the execution of the variable length encoding kernel %d times", NT);
  printf(" and the execution of the kernels for testing\n");

  // Verification
  compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
#endif 

  free(sourceData); 
  free(destData);  	
  free(codewords);  	
  free(codewordlens); 
  free(cw32);  
  free(cw32len); 
  free(crefData); 
  free(cindex2);
}

