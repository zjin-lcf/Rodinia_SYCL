/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <sys/time.h>
#include "common.h"

using namespace std;

// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define IN_RANGE(x, min, max)	((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

int main(int argc, char** argv)
{
  // Program variables.
  int   rows, cols;
  int*  data;
  int** wall;
  int*  result;
  int   pyramid_height;

  if (argc == 4)
  {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  }
  else
  {
    printf("Usage: %s row_len col_len pyramid_height\n", argv[0]);
    exit(0);
  }

  data = new int[rows * cols];
  wall = new int*[rows];
  for (int n = 0; n < rows; n++)
  {
    // wall[n] is set to be the nth row of the data array.
    wall[n] = data + cols * n;
  }
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      wall[i][j] = rand() % 10;
    }
  }
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%d ", wall[i][j]);
    }
    printf("\n");
  }
#endif

  // Pyramid parameters.
  const int borderCols = (pyramid_height) * HALO;


  /* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
     pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

  int size = rows * cols;  // also global work size // 10000000

  // running the opencl application shows lws=4000 (cpu) and lws=250 (gpu)
#ifdef USE_GPU
  int lws = 250;
#else
  int lws = 4000;
#endif
  cl_int* h_outputBuffer = (cl_int*)calloc(16384, sizeof(cl_int));
  int theHalo = HALO;

  double offload_start = get_time();
  { // SYCL scope

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);
    // Allocate device memory.

    const property_list props = property::buffer::use_host_ptr();
    buffer<int,1> d_gpuWall (data + cols, size-cols, props);

    std::vector<buffer<int>> d_gpuResult;
    d_gpuResult.emplace_back(data, cols, props);
    d_gpuResult.emplace_back(cols);
    d_gpuResult[0].set_final_data(nullptr);
    d_gpuResult[1].set_final_data(nullptr);

    buffer<int,1> d_outputBuffer (h_outputBuffer, 16384, props);

    int src = 1, final_ret = 0;
    for (int t = 0; t < rows - 1; t += pyramid_height)
    {
      int temp = src;
      src = final_ret;
      final_ret = temp;

      // Calculate this for the kernel argument...
      int iteration = MIN(pyramid_height, rows-t-1);

      q.submit([&](handler& cgh) {
          auto gpuWall_acc = d_gpuWall.get_access<sycl_read>(cgh);
          auto gpuSrc_acc = d_gpuResult[src].get_access<sycl_read>(cgh);
          auto gpuResult_acc = d_gpuResult[final_ret].get_access<sycl_write>(cgh);
          auto outputBuffer_acc = d_outputBuffer.get_access<sycl_write>(cgh);
          accessor <int, 1, sycl_read_write, access::target::local> prev (lws, cgh);
          accessor <int, 1, sycl_read_write, access::target::local> result (lws, cgh);

          // Set the kernel arguments.
          cgh.parallel_for<class dynproc_kernel>(
              nd_range<1>(range<1>(size), range<1>(lws)), [=] (nd_item<1> item) {
#include "kernel.sycl"
              });
          });

    } // for

    // Copy results back to host.
    q.submit([&](handler& cgh) {
        accessor<int, 1, sycl_read, sycl_global_buffer> 
        d_gpuResult_acc(d_gpuResult[final_ret], cgh, range<1>(cols), id<1>(0));
        cgh.copy(d_gpuResult_acc, result);
        });
  } // SYCL scope

  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

  // add a null terminator at the end of the string.
  h_outputBuffer[16383] = '\0';

#ifdef BENCH_PRINT
  for (int i = 0; i < cols; i++)
    printf("%d ", data[i]);
  printf("\n");
  for (int i = 0; i < cols; i++)
    printf("%d ", result[i]);
  printf("\n");
#endif

  // Memory cleanup here.
  delete[] data;
  delete[] wall;
  delete[] result;
  free(h_outputBuffer);

  return EXIT_SUCCESS;
}
