#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include "common.h"
#include "bucketsort.h"

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints,
    float histo_width);

////////////////////////////////////////////////////////////////////////////////
// Given the input array of floats and the min and max of the distribution,
// sort the elements into float4 aligned buckets of roughly equal size
////////////////////////////////////////////////////////////////////////////////
void bucketSort(float *d_input, float *d_output, int listsize,
    int *sizes, int *nullElements, float minimum, float maximum,
    unsigned int *origOffsets)
{

  const int histosize = 1024;
  //	////////////////////////////////////////////////////////////////////////////
  //	// First pass - Create 1024 bin histogram
  //	////////////////////////////////////////////////////////////////////////////
  unsigned int* h_offsets = (unsigned int *) malloc(DIVISIONS * sizeof(unsigned int));
  for(int i = 0; i < DIVISIONS; i++){
    h_offsets[i] = 0;
  }
  float* pivotPoints = (float *)malloc(DIVISIONS * sizeof(float));
  int* d_indice = (int *)malloc(listsize * sizeof(int));
  float* historesult = (float *)malloc(histosize * sizeof(float));
  //float* l_pivotpoints = (float *)malloc(DIVISIONS*sizeof(float));

  int blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;
  unsigned int* d_prefixoffsets = (unsigned int *)malloc(blocks*BUCKET_BLOCK_MEMORY*sizeof(int));
  unsigned int* d_prefixoffsets_altered = (unsigned int *)malloc(blocks*BUCKET_BLOCK_MEMORY*sizeof(int));

  const property_list props = property::buffer::use_host_ptr();

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);


  { // SYCL scope
    buffer<float,1> histoInput (d_input, listsize, props);
    buffer<unsigned int,1> histoOutput (h_offsets, DIVISIONS , props);
    size_t global = 6144;
    size_t local;

#ifdef HISTO_WG_SIZE_0
    local = HISTO_WG_SIZE_0;
#else
    local = 96;
#endif
    q.submit([&](handler& cgh) {

        auto histoOutput_acc = histoOutput.get_access<sycl_atomic>(cgh);
        auto histoInput_acc = histoInput.get_access<sycl_read>(cgh);
        //accessor <unsigned int, 1, sycl_read_write, access::target::local> 
        accessor <unsigned int, 1, sycl_atomic, access::target::local> 
        s_Hist (HISTOGRAM_BLOCK_MEMORY, cgh);

        cgh.parallel_for<class histogram1024>(
          nd_range<1>(range<1>(global), range<1>(local)), [=] (nd_item<1> item) {
#include "kernel_histogram.sycl"
        });
    });
  } // SYCL scope

  for(int i=0; i<histosize; i++) {
    historesult[i] = (float)h_offsets[i];
  }

  //	///////////////////////////////////////////////////////////////////////////
  //	// Calculate pivot points (CPU algorithm)
  //	///////////////////////////////////////////////////////////////////////////
  calcPivotPoints(historesult, histosize, listsize, DIVISIONS,
      minimum, maximum, pivotPoints,
      (maximum - minimum)/(float)histosize);
  //
  //	///////////////////////////////////////////////////////////////////////////
  //	// Count the bucket sizes in new divisions
  //	///////////////////////////////////////////////////////////////////////////


  {
    buffer<float,1>d_input_buff(d_input, listsize + DIVISIONS*4, props);
    buffer<int,1> d_indice_buff(d_indice, listsize, props);
    buffer<unsigned int,1> d_prefixoffsets_buff(d_prefixoffsets, blocks * BUCKET_BLOCK_MEMORY, props);
    buffer<float,1> l_pivotpoints_buff(pivotPoints, DIVISIONS, props);

    int blocks =((listsize -1) / (BUCKET_THREAD_N*BUCKET_BAND)) + 1;
    size_t global = blocks*BUCKET_THREAD_N;
    size_t local = BUCKET_THREAD_N;

    q.submit([&](handler& cgh) {

        auto input_acc = d_input_buff.get_access<sycl_read>(cgh);
        auto indice_acc = d_indice_buff.get_access<sycl_write>(cgh);
        auto d_prefixoffsets_acc = d_prefixoffsets_buff.get_access<sycl_write>(cgh);
        auto l_pivotpoints_acc = l_pivotpoints_buff.get_access<sycl_read>(cgh);
        accessor <unsigned int, 1, sycl_atomic, access::target::local> 
        s_offset (BUCKET_BLOCK_MEMORY, cgh);

        cgh.parallel_for<class bucketcount>(
          nd_range<1>(range<1>(global),
            range<1>(local)), [=] (nd_item<1> item) {
#include "kernel_bucketcount.sycl"
        });
    });
  }

  //
  //	///////////////////////////////////////////////////////////////////////////
  //	// Prefix scan offsets and align each division to float4 (required by
  //	// mergesort)
  //	///////////////////////////////////////////////////////////////////////////
#ifdef BUCKET_WG_SIZE_0
  size_t localpre = BUCKET_WG_SIZE_0;
#else
  size_t localpre = 128;
#endif
  size_t globalpre = DIVISIONS;

  {// SYCL scope
    buffer<unsigned int,1> d_prefixoffsets_buff(d_prefixoffsets, blocks * BUCKET_BLOCK_MEMORY, props);
    d_prefixoffsets_buff.set_final_data(d_prefixoffsets_altered);

    buffer<unsigned int,1> d_offsets_buff(h_offsets, DIVISIONS, props);

    q.submit([&](handler& cgh) {

        auto d_prefixoffsets_acc = d_prefixoffsets_buff.get_access<sycl_read_write>(cgh);
        auto d_offsets_acc = d_offsets_buff.get_access<sycl_write>(cgh);
        cgh.parallel_for<class prefix>(
          nd_range<1>(range<1>(globalpre), range<1>(localpre)), [=] (nd_item<1> item) {
#include "kernel_bucketprefix.sycl"
          });
        });
  }// SYCL scope

  //	// copy the sizes from device to host
  origOffsets[0] = 0;
  for(int i=0; i<DIVISIONS; i++){
    origOffsets[i+1] = h_offsets[i] + origOffsets[i];
    if((h_offsets[i] % 4) != 0){
      nullElements[i] = (h_offsets[i] & ~3) + 4 - h_offsets[i];
    }
    else nullElements[i] = 0;
  }
  for(int i=0; i<DIVISIONS; i++) sizes[i] = (h_offsets[i] + nullElements[i])/4;
  for(int i=0; i<DIVISIONS; i++) {
    if((h_offsets[i] % 4) != 0)	h_offsets[i] = (h_offsets[i] & ~3) + 4;
  }
  for(int i=1; i<DIVISIONS; i++) h_offsets[i] = h_offsets[i-1] + h_offsets[i];
  for(int i=DIVISIONS - 1; i>0; i--) h_offsets[i] = h_offsets[i-1];
  h_offsets[0] = 0;


  //	///////////////////////////////////////////////////////////////////////////
  //	// Finally, sort the lot
  //	///////////////////////////////////////////////////////////////////////////
  buffer<unsigned int,1> l_offsets_buff(h_offsets, DIVISIONS);
  buffer<float,1>d_input_buff(d_input, listsize + DIVISIONS*4, props);
  buffer<int,1> d_indice_buff(d_indice, listsize, props);
  buffer<unsigned int,1> d_prefixoffsets_buff(d_prefixoffsets_altered, blocks * BUCKET_BLOCK_MEMORY, props);
  buffer<float,1> d_bucketOutput(d_output, listsize + DIVISIONS*4, props);


  size_t localfinal = BUCKET_THREAD_N;
  blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;
  size_t globalfinal = blocks*BUCKET_THREAD_N;

  q.submit([&](handler& cgh) {

      auto input_acc = d_input_buff.get_access<sycl_read>(cgh);
      auto indice_acc = d_indice_buff.get_access<sycl_read>(cgh);
      auto output_acc = d_bucketOutput.get_access<sycl_write>(cgh);
      auto d_prefixoffsets_acc = d_prefixoffsets_buff.get_access<sycl_read>(cgh);
      auto l_offsets_acc = l_offsets_buff.get_access<sycl_read>(cgh);
      accessor <unsigned int, 1, sycl_read_write, access::target::local> 
      s_offset (BUCKET_BLOCK_MEMORY, cgh);

      cgh.parallel_for<class bucketsort>(
        nd_range<1>(range<1>(globalfinal),
          range<1>(localfinal)), [=] (nd_item<1> item) {
#include "kernel_bucketsort.sycl"
        });
      });

  free(pivotPoints);
  free(d_indice);
  free(historesult);
  free(d_prefixoffsets);
  free(d_prefixoffsets_altered);
}
////////////////////////////////////////////////////////////////////////////////
// Given a histogram of the list, figure out suitable pivotpoints that divide
// the list into approximately listsize/divisions elements each
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints, float histo_width)
{
  float elemsPerSlice = listsize/(float)divisions;
  float startsAt = min;
  float endsAt = min + histo_width;
  float we_need = elemsPerSlice;
  int p_idx = 0;
  for(int i=0; i<histosize; i++)
  {
    if(i == histosize - 1){
      if(!(p_idx < divisions)){
        pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      }
      break;
    }
    while(histogram[i] > we_need){
      if(!(p_idx < divisions)){
        printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions);
        exit(0);
      }
      pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      startsAt += (we_need/histogram[i]) * histo_width;
      histogram[i] -= we_need;
      we_need = elemsPerSlice;
    }
    // grab what we can from what remains of it
    we_need -= histogram[i];

    startsAt = endsAt;
    endsAt += histo_width;
  }
  while(p_idx < divisions){
    pivotPoints[p_idx] = pivotPoints[p_idx-1];
    p_idx++;
  }
}
