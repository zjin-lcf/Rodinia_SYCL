/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
#include <stdio.h>
#include "common.h"


int runHisto(cl::sycl::queue &q, char* file, unsigned int* freq, unsigned int memSize, unsigned int *source) {

    FILE *f = fopen(file,"rb");
    if (!f) {perror(file); exit(1);}
    fseek(f,0,SEEK_SET);
    size_t result = fread(source,1,memSize,f);
    if(result != memSize) fputs("Cannot read input file", stderr);

    fclose(f);

    int blocks = q.get_device().get_info<info::device::max_compute_units>();
    //printf("max compute units are %d\n", blocks);

    // allocate memory on the GPU for the file's data
    int partSize = memSize/32;
    int totalNum = memSize/sizeof(unsigned int);
    int partialNum = partSize/sizeof(unsigned int);

    //unsigned char *dev_buffer0; 
    //unsigned char *dev_buffer1;
    //unsigned int *dev_histo;
    //cudaMalloc( (void**)&dev_buffer0, partSize ) ;
    //cudaMalloc( (void**)&dev_buffer1, partSize ) ;
    //cudaMalloc( (void**)&dev_histo, 256 * sizeof( int ) ) ;
    
    buffer<unsigned char, 1> dev_buffer0 (partSize);
    buffer<unsigned char, 1> dev_buffer1 (partSize);
    buffer<unsigned int, 1> dev_histo   (256);

    
    //cudaMemset( dev_histo, 0, 256 * sizeof( int ) ) ;
    q.submit([&](handler& cgh) {
      auto dev_histo_acc = dev_histo.get_access<sycl_write>(cgh);
      cgh.fill(dev_histo_acc, 0u);
    });


    for(int i = 0; i < totalNum; i+=partialNum*2)
    {

      //unsigned char *buffer = (unsigned char*)source;
      //CHECK(cudaMemcpyAsync(dev_buffer0, buffer+i, partSize, cudaMemcpyHostToDevice,stream0));
      //CHECK(cudaMemcpyAsync(dev_buffer1, buffer+i+partialNum, partSize, cudaMemcpyHostToDevice,stream1));
      q.submit([&](handler& cgh) {
        auto dev_buffer0_acc = dev_buffer0.get_access<sycl_write>(cgh);
        cgh.copy((unsigned char*)source+i, dev_buffer0_acc);
      });
      q.submit([&](handler& cgh) {
        auto dev_buffer1_acc = dev_buffer1.get_access<sycl_write>(cgh);
        cgh.copy((unsigned char*)source+i+partialNum, dev_buffer1_acc);
      });

      q.submit([&](handler& cgh) {
        auto buffer = dev_buffer0.get_access<sycl_read>(cgh);
        auto histo = dev_histo.get_access<sycl_atomic>(cgh);
        accessor <unsigned int, 1, sycl_atomic, access::target::local> temp (256, cgh);
        cgh.parallel_for<class histogram_0>(
          nd_range<1>(range<1>(512*blocks), range<1>(256)), [=] (nd_item<1> item) {
            int lid = item.get_local_id(0);
            temp[lid].store(0);
            item.barrier(access::fence_space::local_space);
            int i = item.get_global_id(0);
            int offset = item.get_global_range(0);
            while (i < partSize) {
                atomic_fetch_add( temp[buffer[i]], 1u );
                i += offset;
            }
            item.barrier(access::fence_space::local_space);
            atomic_fetch_add( (histo[lid]), atomic_load(temp[lid]) );
          });
        });

      q.submit([&](handler& cgh) {
        auto buffer = dev_buffer1.get_access<sycl_read>(cgh);
        auto histo = dev_histo.get_access<sycl_atomic>(cgh);
        accessor <unsigned int, 1, sycl_atomic, access::target::local> temp (256, cgh);
        cgh.parallel_for<class histogram_1>(
          nd_range<1>(range<1>(512*blocks), range<1>(256)), [=] (nd_item<1> item) {
            int lid = item.get_local_id(0);
            temp[lid].store(0);
            item.barrier(access::fence_space::local_space);
            int i = item.get_global_id(0);
            int offset = item.get_global_range(0);
            while (i < partSize) {
                atomic_fetch_add( temp[buffer[i]], 1u );
                i += offset;
            }
            item.barrier(access::fence_space::local_space);
            atomic_fetch_add( (histo[lid]), atomic_load(temp[lid]) );
          });
        });

    }
    //cudaMemcpy( freq, dev_histo, 256 * sizeof( int ), cudaMemcpyDeviceToHost );
    q.submit([&](handler& cgh) {
      auto dev_histo_acc = dev_histo.get_access<sycl_read>(cgh);
      cgh.copy(dev_histo_acc, freq);
    });

    q.wait();

    //printf( "Time to generate:  %3.1f ms\n", elapsedTime );
#ifdef DEBUG
    printf("After histogram:\n");
    for (int i = 0; i < 256; i++) {
      printf("i=%d freq=%d\n", i, freq[i]);
    }
    printf("\n");
#endif

    return 0;
}
