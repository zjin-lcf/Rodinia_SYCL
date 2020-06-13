/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _SCAN_CPP_
#define _SCAN_CPP_

// includes, kernels
#include <assert.h>
#include <stdio.h>
#include <algorithm>   //std::max

  inline bool
isPowerOfTwo(int n)
{
  return ((n&(n-1))==0) ;
}

  inline int 
floorPow2(int n)
{
#ifdef WIN32
  // method 2
  return 1 << (int)logb((float)n);
#else
  // method 1
  // float nf = (float)n;
  // return 1 << (((*(int*)&nf) >> 23) - 127); 
  int exp;
  frexp((float)n, &exp);
  return 1 << (exp - 1);
#endif
}

// 16 banks 
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

template <bool isNP2>
static void loadSharedChunkFromMem(nd_item<1> &item, 
    const accessor<unsigned int,1,sycl_read_write, access::target::local> &s_data,
    const accessor<unsigned int,1,sycl_read, access::target::global_buffer> &g_idata,
                                       int n, int baseIndex,
                                       int& ai, int& bi, 
                                       int& mem_ai, int& mem_bi, 
                                       int& bankOffsetA, int& bankOffsetB)
{
    int thid = item.get_local_id(0);
    mem_ai = baseIndex + item.get_local_id(0);
    mem_bi = mem_ai + item.get_local_range(0);

    ai = thid;
    bi = thid + item.get_local_range(0);

    // compute spacing to avoid bank conflicts
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    // pad values beyond n with zeros
    s_data[ai + bankOffsetA] = g_idata[mem_ai]; 
    
    if (isNP2) // compile-time decision
    {
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    }
    else
    {
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}

template <bool isNP2>
static void storeSharedChunkToMem(nd_item<1> &item, 
    const accessor<unsigned int,1,sycl_write, access::target::global_buffer> &g_odata,
    const accessor<unsigned int,1,sycl_read_write, access::target::local> &s_data,
                                      int n, 
                                      int ai, int bi, 
                                      int mem_ai, int mem_bi,
                                      int bankOffsetA, int bankOffsetB)
{
    item.barrier(access::fence_space::local_space);

    // write results to global memory
    g_odata[mem_ai] = s_data[ai + bankOffsetA]; 
    if (isNP2) // compile-time decision
    {
        if (bi < n)
            g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
    else
    {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}

///template <bool storeSum>
static void clearLastElement(nd_item<1> &item, 
    const accessor<unsigned int,1,sycl_read_write, access::target::local> &s_data,
    const accessor<unsigned int,1,sycl_write, access::target::global_buffer> &g_blockSums,
                                 int blockIndex)
{
    if (item.get_local_id(0) == 0)
    {
        int index = (item.get_local_range(0) << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        
        //if (storeSum) // compile-time decision
        //{
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        //}

        // zero the last element in the scan so it will propagate back to the front
        s_data[index] = 0;
    }
}

static void clearLastElement(nd_item<1> &item, 
    const accessor<unsigned int,1,sycl_read_write, access::target::local> &s_data,
                                 int blockIndex)
{
    if (item.get_local_id(0) == 0)
    {
        int index = (item.get_local_range(0) << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);

        // zero the last element in the scan so it will propagate back to the front
        s_data[index] = 0;
    }
}

static unsigned int buildSum(nd_item<1> &item, 
    const accessor<unsigned int,1,sycl_read_write, access::target::local> &s_data )
{
    unsigned int thid = item.get_local_id(0);
    unsigned int stride = 1;
    
    // build the sum in place up the tree
    for (int d = item.get_local_range(0); d > 0; d >>= 1)
    {
        item.barrier(access::fence_space::local_space);

        if (thid < d)      
        {
            int i  = cl::sycl::mul24(cl::sycl::mul24(2u, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

static void scanRootToLeaves(
    nd_item<1> &item,
    const accessor<unsigned int,1,sycl_read_write, access::target::local> &s_data ,
    unsigned int stride)
{
     unsigned int thid = item.get_local_id(0);

    // traverse down the tree building the scan in place
    for (int d = 1; d <= item.get_local_range(0); d *= 2)
    {
        stride >>= 1;

        item.barrier(access::fence_space::local_space);

        if (thid < d)
        {
            int i  = cl::sycl::mul24(cl::sycl::mul24(2u, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            unsigned int t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

//template <bool storeSum>
static void prescanBlock(nd_item<1> &item, 
    const accessor<unsigned int,1,sycl_read_write, access::target::local> &data,
    int blockIndex, 
    const accessor<unsigned int,1,sycl_write, access::target::global_buffer> &blockSums )
{
    int stride = buildSum(item, data);               // build the sum in place up the tree
    //clearLastElement<storeSum>(item, data, blockSums, (blockIndex == 0) ? item.get_group(0) : blockIndex);
    clearLastElement(item, data, blockSums, (blockIndex == 0) ? item.get_group(0) : blockIndex);
    scanRootToLeaves(item, data, stride);            // traverse down tree to build the scan 
}

static void prescanBlock(nd_item<1> &item, 
    const accessor<unsigned int,1,sycl_read_write, access::target::local> &data,
    int blockIndex)
{
    int stride = buildSum(item, data);               // build the sum in place up the tree
    clearLastElement(item, data, (blockIndex == 0) ? item.get_group(0) : blockIndex);
    scanRootToLeaves(item, data, stride);            // traverse down tree to build the scan 
}

#define BLOCK_SIZE 256

//static unsigned int** g_scanBlockSums;
std::vector<buffer<unsigned int>> g_scanBlockSums;
static unsigned int g_numEltsAllocated = 0;
static unsigned int g_numLevelsAllocated = 0;

static void preallocBlockSums(unsigned int maxNumElements)
{
  assert(g_numEltsAllocated == 0); // shouldn't be called 

  g_numEltsAllocated = maxNumElements;

  unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
  unsigned int numElts = maxNumElements;
  int level = 0;

  do {       
    unsigned int numBlocks = std::max(1, (int)ceil((float)numElts / (2.f * blockSize)));
    if (numBlocks > 1) level++;
    numElts = numBlocks;
  } while (numElts > 1);

  //g_scanBlockSums = (unsigned int**) malloc(level * sizeof(unsigned int*));
  //std::vector<buffer<unsigned int>> g_scanBlockSums;
  g_numLevelsAllocated = level;
  numElts = maxNumElements;
  level = 0;

  do {       
    unsigned int numBlocks = std::max(1, (int)ceil((float)numElts / (2.f * blockSize)));
    if (numBlocks > 1) {
      //CUDA_SAFE_CALL(cudaMalloc((void**) &g_scanBlockSums[level++], numBlocks * sizeof(unsigned int)));
      g_scanBlockSums.emplace_back(numBlocks);
    }
    numElts = numBlocks;
  } while (numElts > 1);

}

static void deallocBlockSums()
{
  g_numEltsAllocated = 0;
  g_numLevelsAllocated = 0;
}

static void prescanArrayRecursive(cl::sycl::queue &q, 
                                  buffer<unsigned int,1> &outArray, 
                                  buffer<unsigned int,1> &inArray, 
                                  int numElements, 
                                  int level)
{
  unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
  unsigned int numBlocks = 
    std::max(1, (int)ceil((float)numElements / (2.f * blockSize)));
  unsigned int numThreads;

  if (numBlocks > 1)
    numThreads = blockSize;
  else if (isPowerOfTwo(numElements))
    numThreads = numElements / 2;
  else
    numThreads = floorPow2(numElements);

  unsigned int numEltsPerBlock = numThreads * 2;

  // if this is a non-power-of-2 array, the last block will be non-full
  // compute the smallest power of 2 able to compute its scan.
  unsigned int numEltsLastBlock = 
    numElements - (numBlocks-1) * numEltsPerBlock;
  unsigned int numThreadsLastBlock = std::max(1u, numEltsLastBlock / 2);
  unsigned int np2LastBlock = 0;
  unsigned int sharedMemLastBlock = 0;

  if (numEltsLastBlock != numEltsPerBlock)
  {
    np2LastBlock = 1;

    if(!isPowerOfTwo(numEltsLastBlock))
      numThreadsLastBlock = floorPow2(numEltsLastBlock);    

    unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
    //sharedMemLastBlock = sizeof(unsigned int) * (2 * numThreadsLastBlock + extraSpace);
    sharedMemLastBlock = 2 * numThreadsLastBlock + extraSpace;
  }

  // padding space is used to avoid shared memory bank conflicts
  unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
  unsigned int sharedMemSize = (numEltsPerBlock + extraSpace);
  //sizeof(unsigned int) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
  if (numBlocks > 1)
  {
    assert(g_numEltsAllocated >= numElements);
  }
#endif

  // setup execution parameters
  // if NP2, we process the last block separately
  //dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
  int grid = std::max(1u, numBlocks - np2LastBlock);

  //dim3  threads(numThreads, 1, 1);
  int threads = numThreads;
  int n, blockIndex, baseIndex;

  // execute the scan
  if (numBlocks > 1)
  {
    //prescan<true, false><<< grid, threads, sharedMemSize >>>(outArray, inArray, g_scanBlockSums[level], numThreads * 2, 0, 0); CUT_CHECK_ERROR("prescanWithBlockSums");
    n = numThreads * 2;
    blockIndex = 0;
    baseIndex = 0;
    q.submit([&](handler& cgh) {
        auto g_odata = outArray.get_access<sycl_write>(cgh);
        auto g_idata = inArray.get_access<sycl_read>(cgh);
        auto g_blockSums = g_scanBlockSums[level].get_access<sycl_write>(cgh);
        accessor <unsigned int, 1, sycl_read_write, access::target::local> s_data (sharedMemSize, cgh);
        cgh.parallel_for<class prescanWithBlockSums>(
          nd_range<1>(range<1>(grid*threads), range<1>(threads)), [=] (nd_item<1> item) {
          int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
          loadSharedChunkFromMem<false>(item, s_data, g_idata, n, 
            (baseIndex == 0) ? 
            cl::sycl::mul24((int)item.get_group(0),  (int)(item.get_local_range(0)<< 1)) : baseIndex,
            ai, bi, mem_ai, mem_bi, 
            bankOffsetA, bankOffsetB); 
          // scan the data in each block
          //prescanBlock<true>(item, s_data, blockIndex, g_blockSums); 
          prescanBlock(item, s_data, blockIndex, g_blockSums); 
          // write results to device memory
          storeSharedChunkToMem<false>(item, g_odata, s_data, n, 
            ai, bi, mem_ai, mem_bi, 
            bankOffsetA, bankOffsetB);  

          });
    });
    if (np2LastBlock)
    {
      //prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
        //(outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, 
         //numBlocks - 1, numElements - numEltsLastBlock);

      n = numEltsLastBlock;
      blockIndex = numBlocks - 1;
      baseIndex = numElements - numEltsLastBlock;
      q.submit([&](handler& cgh) {
          auto g_odata = outArray.get_access<sycl_write>(cgh);
          auto g_idata = inArray.get_access<sycl_read>(cgh);
          auto g_blockSums = g_scanBlockSums[level].get_access<sycl_write>(cgh);
          accessor <unsigned int, 1, sycl_read_write, access::target::local> s_data (sharedMemLastBlock, cgh);
          cgh.parallel_for<class prescanNP2WithBlockSums>(
            nd_range<1>(range<1>(numThreadsLastBlock), range<1>(numThreadsLastBlock)), [=] (nd_item<1> item) {
            int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
            loadSharedChunkFromMem<true>(item, s_data, g_idata, n, 
              (baseIndex == 0) ? 
              cl::sycl::mul24((int)item.get_group(0),  (int)(item.get_local_range(0)<< 1)):baseIndex,
              ai, bi, mem_ai, mem_bi, 
              bankOffsetA, bankOffsetB); 
            // scan the data in each block
            //prescanBlock<true>(item, s_data, blockIndex, g_blockSums); 
            prescanBlock(item, s_data, blockIndex, g_blockSums); 
            // write results to device memory
            storeSharedChunkToMem<true>(item, g_odata, s_data, n, 
              ai, bi, mem_ai, mem_bi, 
              bankOffsetA, bankOffsetB);  
            });
      });
    } // if (np2LastBlock)

    q.wait();

    // After scanning all the sub-blocks, we are mostly done.  But now we 
    // need to take all of the last values of the sub-blocks and scan those.  
    // This will give us a new value that must be added to each block to 
    // get the final results.
    // recursive (CPU) call
    prescanArrayRecursive(q, g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level+1);

    //uniformAdd<<< grid, threads >>>(outArray, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);
    q.submit([&](handler& cgh) {
        auto g_data = outArray.get_access<sycl_read_write>(cgh);
        auto uniforms = g_scanBlockSums[level].get_access<sycl_read>(cgh);
        accessor <unsigned int, 1, sycl_read_write, access::target::local> uni (1, cgh);
        cgh.parallel_for<class uniformAdd>(
          nd_range<1>(range<1>(grid*threads), range<1>(threads)), [=] (nd_item<1> item) {
          if (item.get_local_id(0) == 0) uni[0] = uniforms[item.get_group(0)];

          unsigned int address = cl::sycl::mul24((int)item.get_group(0), 
                                                 (int)(item.get_local_range(0) << 1)) 
                                 + item.get_local_id(0); 

          item.barrier(access::fence_space::local_space);

          // note two adds per thread
          g_data[address] += uni[0];
          g_data[address + item.get_local_range(0)] += 
            (item.get_local_id(0) + 
             item.get_local_range(0) < (numElements - numEltsLastBlock)) * uni[0];
        });
    });

    if (np2LastBlock)
    {
      //uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
      q.submit([&](handler& cgh) {
        auto g_data = outArray.get_access<sycl_read_write>(cgh);
        auto uniforms = g_scanBlockSums[level].get_access<sycl_read>(cgh);
        accessor <unsigned int, 1, sycl_read_write, access::target::local> uni (1, cgh);
        cgh.parallel_for<class uniformAddLastBlock>(
          nd_range<1>(range<1>(numThreadsLastBlock), range<1>(numThreadsLastBlock)), [=] (nd_item<1> item) {
          if (item.get_local_id(0) == 0) 
             uni[0] = uniforms[item.get_group(0) + numBlocks - 1];

          unsigned int address = cl::sycl::mul24((int)item.get_group(0), (int)(item.get_local_range(0) << 1)) 
                                 + numElements - numEltsLastBlock + item.get_local_id(0); 

          item.barrier(access::fence_space::local_space);

          // note two adds per thread
          g_data[address]              += uni[0];
          g_data[address + item.get_local_range(0)] += 
          (item.get_local_id(0) + item.get_local_range(0) < numEltsLastBlock) * uni[0];
          });
        });
    }
  }
  else if (isPowerOfTwo(numElements))
  {
    //prescan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray, 0, numThreads * 2, 0, 0);
      n = numThreads * 2;
      blockIndex = 0;
      baseIndex = 0;
      q.submit([&](handler& cgh) {
          auto g_odata = outArray.get_access<sycl_write>(cgh);
          auto g_idata = inArray.get_access<sycl_read>(cgh);
          //auto g_blockSums = g_scanBlockSums[level].get_access<sycl_write>(cgh);
          accessor <unsigned int, 1, sycl_read_write, access::target::local> s_data (sharedMemSize, cgh);
          cgh.parallel_for<class prescan>(
            nd_range<1>(range<1>(grid*threads), range<1>(threads)), [=] (nd_item<1> item) {
            int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
            loadSharedChunkFromMem<false>(item, s_data, g_idata, n, 
              cl::sycl::mul24((int)item.get_group(0), (int) (item.get_local_range(0)<< 1)),
              ai, bi, mem_ai, mem_bi, 
              bankOffsetA, bankOffsetB); 
            // scan the data in each block
            //prescanBlock<false>(item, s_data, blockIndex, g_blockSums); 
            prescanBlock(item, s_data, blockIndex);
            // write results to device memory
            storeSharedChunkToMem<false>(item, g_odata, s_data, n, 
              ai, bi, mem_ai, mem_bi, 
              bankOffsetA, bankOffsetB);  

            });
      });
  }
  else
  {
    //prescan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 0, numElements, 0, 0);
      n = numElements;
      blockIndex = 0;
      baseIndex = 0;
      q.submit([&](handler& cgh) {
          auto g_odata = outArray.get_access<sycl_write>(cgh);
          auto g_idata = inArray.get_access<sycl_read>(cgh);
          // still need the accessor though it is not used when storeSum is false
          //auto g_blockSums = g_scanBlockSums[level].get_access<sycl_write>(cgh);
          accessor <unsigned int, 1, sycl_read_write, access::target::local> s_data (sharedMemSize, cgh);
          cgh.parallel_for<class prescanNP2>(
            nd_range<1>(range<1>(grid*threads), range<1>(threads)), [=] (nd_item<1> item) {
            int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
            loadSharedChunkFromMem<true>(item, s_data, g_idata, n, 
              cl::sycl::mul24((int)item.get_group(0),  (int)(item.get_local_range(0)<< 1)),
              ai, bi, mem_ai, mem_bi, 
              bankOffsetA, bankOffsetB); 
            // scan the data in each block
            //prescanBlock<false>(item, s_data, blockIndex, g_blockSums); 
            prescanBlock(item, s_data, blockIndex);
            // write results to device memory
            storeSharedChunkToMem<true>(item, g_odata, s_data, n, 
              ai, bi, mem_ai, mem_bi, 
              bankOffsetA, bankOffsetB);  

            });
      });
  }
  //q.wait();
}

static void prescanArray(cl::sycl::queue &q, 
                         buffer<unsigned int,1> &outArray, 
                         buffer<unsigned int,1> &inArray, 
                         int numElements)
{
  prescanArrayRecursive(q, outArray, inArray, numElements, 0);
}

#endif // _SCAN_CPP_

