////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
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
#include "mergesort.h"

////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE	256
#define ROW_LENGTH	BLOCKSIZE * 4
#define ROWS		4096

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////

// Codeplay 
//  error: no viable conversion from 'vec<typename
//        detail::vec_ops::logical_return<sizeof(float)>::type, 1>'
//              (aka 'vec<int, 1>') to 'bool'
//                b.z() = a.y() >= b.z() ? a.y() : b.z();
float4 sortElem(float4 r) {
  float4 nr;

  float xt = r.x();
  float yt = r.y();
  float zt = r.z();
  float wt = r.w();

  float nr_xt = xt > yt ? yt : xt;
  float nr_yt = yt > xt ? yt : xt;
  float nr_zt = zt > wt ? wt : zt;
  float nr_wt = wt > zt ? wt : zt;

  xt = nr_xt > nr_zt ? nr_zt : nr_xt;
  yt = nr_yt > nr_wt ? nr_wt : nr_yt;
  zt = nr_zt > nr_xt ? nr_zt : nr_xt;
  wt = nr_wt > nr_yt ? nr_wt : nr_yt;

  nr.x() = xt;
  nr.y() = yt > zt ? zt : yt;
  nr.z() = zt > yt ? zt : yt;
  nr.w() = wt;
  return nr;
}

float4 getLowest(float4 a, float4 b)
{
  float ax = a.x();
  float ay = a.y();
  float az = a.z();
  float aw = a.w();
  float bx = b.x();
  float by = b.y();
  float bz = b.z();
  float bw = b.w();
  a.x() = ax < bw ? ax : bw;
  a.y() = ay < bz ? ay : bz;
  a.z() = az < by ? az : by;
  a.w() = aw < bx ? aw : bx;
  return a;
}

float4 getHighest(float4 a, float4 b)
{
  float ax = a.x();
  float ay = a.y();
  float az = a.z();
  float aw = a.w();
  float bx = b.x();
  float by = b.y();
  float bz = b.z();
  float bw = b.w();
  b.x() = aw >= bx ? aw : bx;
  b.y() = az >= by ? az : by;
  b.z() = ay >= bz ? ay : bz;
  b.w() = ax >= bw ? ax : bw;
  return b;
}

float4* runMergeSort(int listsize, int divisions,
    float4 *d_origList, float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets){

  int *startaddr = (int *)malloc((divisions + 1)*sizeof(int));
  int largestSize = -1;
  startaddr[0] = 0;
  for(int i=1; i<=divisions; i++)
  {
    startaddr[i] = startaddr[i-1] + sizes[i-1];
    if(sizes[i-1] > largestSize) largestSize = sizes[i-1];
  }
  largestSize *= 4;


#ifdef MERGE_WG_SIZE_0
  const int THREADS = MERGE_WG_SIZE_0;
#else
  const int THREADS = 256;
#endif
  size_t local[] = {THREADS,1,1};
  size_t blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1;
  size_t global[] = {blocks*THREADS,1,1};
  size_t grid[] = {blocks,1,1,1};

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
  const property_list props = property::buffer::use_host_ptr();

  // divided by four ?
  buffer<float4,1> d_resultList_first_buff (listsize/4);
  buffer<float4,1> d_origList_first_buff (listsize/4);


  q.submit([&](handler& cgh) {
      auto result_acc = d_resultList_first_buff.get_access<sycl_write>(cgh);
      cgh.copy(d_resultList, result_acc);
  });

  q.submit([&](handler& cgh) {
      auto input_acc = d_origList_first_buff.get_access<sycl_write>(cgh);
      cgh.copy(d_origList, input_acc);
  });

  q.submit([&](handler& cgh) {
      auto input_acc = d_origList_first_buff.get_access<sycl_read>(cgh);
      auto result_acc = d_resultList_first_buff.get_access<sycl_write>(cgh);
      cgh.parallel_for<class mergesort_first>(
          nd_range<1>(range<1>(global[0]), range<1>(local[0])), [=] (nd_item<1> item) {
            int bx = item.get_group(0);
            int lid = item.get_local_id(0);
            int lsize = item.get_local_range(0);
            if (bx*lsize + lid < listsize/4) {
            float4 r = input_acc[bx*lsize+ lid];
            result_acc[bx*lsize+lid] = sortElem(r);
          }
      });
  });

  q.submit([&](handler& cgh) {
      auto res_acc = d_resultList_first_buff.get_access<sycl_read>(cgh);
      cgh.copy(res_acc, d_resultList);
  });

  buffer<int, 1> d_constStartAddr (startaddr, (divisions+1), props);

  //double mergePassTime = 0;
  int nrElems = 2;

  while(true){
    int floatsperthread = (nrElems*4);
    //printf("FPT %d \n", floatsperthread);
    int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread);
    //printf("TPD %d \n",threadsPerDiv);
    int threadsNeeded = threadsPerDiv * divisions;
    //printf("TN %d \n", threadsNeeded);

#ifdef MERGE_WG_SIZE_1
    local[0] = MERGE_WG_SIZE_1;
#else
    local[0] = 208;
#endif

    grid[0] = ((threadsNeeded%local[0]) == 0) ?
      threadsNeeded/local[0] :
      (threadsNeeded/local[0]) + 1;
    if(grid[0] < 8){
      grid[0] = 8;
      local[0] = ((threadsNeeded%grid[0]) == 0) ?
        threadsNeeded / grid[0] :
        (threadsNeeded / grid[0]) + 1;
    }
    // Swap orig/result list
    float4 *tempList = d_origList;
    d_origList = d_resultList;
    d_resultList = tempList;

    global[0] = grid[0]*local[0];

    q.submit([&](handler& cgh) {
        auto result_acc = d_resultList_first_buff.get_access<sycl_write>(cgh);
        cgh.copy(d_resultList, result_acc);
    });

    q.submit([&](handler& cgh) {
        auto input_acc = d_origList_first_buff.get_access<sycl_write>(cgh);
        cgh.copy(d_origList, input_acc);
    });

    q.submit([&](handler& cgh) {
        auto input_acc = d_origList_first_buff.get_access<sycl_read>(cgh);
        auto result_acc = d_resultList_first_buff.get_access<sycl_write>(cgh);
        auto constStartAddr_acc = d_constStartAddr.get_access<sycl_read>(cgh);
        cgh.parallel_for<class mergepass>(
            nd_range<1>(range<1>(global[0]), range<1>(local[0])), [=] (nd_item<1> item) {
#include "kernel_mergeSortPass.sycl"
        });
    });

    q.submit([&](handler& cgh) {
        auto result_acc = d_resultList_first_buff.get_access<sycl_read>(cgh);
        cgh.copy(result_acc, d_resultList);
    });

    q.wait();

    nrElems *= 2;
    floatsperthread = (nrElems*4);

    if(threadsPerDiv == 1) break;
  }


#ifdef MERGE_WG_SIZE_0
  local[0] = MERGE_WG_SIZE_0;
#else
  local[0] = 256;
#endif
  grid[0] = ((largestSize%local[0]) == 0) ?  largestSize/local[0] : (largestSize/local[0]) + 1;
  grid[1] = divisions;
  global[0] = grid[0]*local[0];
  global[1] = grid[1]*local[1];

  buffer<unsigned int, 1> finalStartAddr(origOffsets, divisions+1, props);
  buffer<int, 1> nullElems(nullElements, divisions, props);
  buffer<float,1> d_res ((float*)d_origList, listsize, props);
  buffer<float,1> d_orig ((float*)d_resultList, listsize, props);

  q.submit([&](handler& cgh) {
      auto orig_acc = d_orig.get_access<sycl_read>(cgh);
      auto result_acc = d_res.get_access<sycl_write>(cgh);
      auto finalStartAddr_acc = finalStartAddr.get_access<sycl_read>(cgh);
      auto nullElems_acc = nullElems.get_access<sycl_read>(cgh);
      auto constStartAddr_acc = d_constStartAddr.get_access<sycl_read>(cgh);
      cgh.parallel_for<class mergepack>(
          nd_range<2>(range<2>(global[0],global[1]), range<2>(local[0], local[1])), [=] (nd_item<2> item) {
          int idx = item.get_global_id(0);
          int division = item.get_group(1);
          if((finalStartAddr_acc[division] + idx) >= finalStartAddr_acc[division + 1]) return;
          result_acc[finalStartAddr_acc[division] + idx] = orig_acc[constStartAddr_acc[division]*4 + nullElems_acc[division] + idx];
          });
      });

  free(startaddr);
  return d_origList;

}
