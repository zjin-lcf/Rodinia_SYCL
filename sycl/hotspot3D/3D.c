#include <sys/types.h>
#include "common.h"
#include "3D_helper.h"

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

/* required precision in degrees	*/
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

#define WG_SIZE_X (64)
#define WG_SIZE_Y (4)
float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;


void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}



int main(int argc, char** argv)
{
  if (argc != 7)
  {
    usage(argc,argv);
  }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[1]);
  int layers       = atoi(argv[2]);

  /* calculating parameters*/

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw                                              = stepDivCap/ Rx;
  cn               = cs                                              = stepDivCap/ Ry;
  ct               = cb                                              = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);


  int size = numCols * numRows * layers;
  float* tIn      = (float*) calloc(size,sizeof(float));
  float* pIn      = (float*) calloc(size,sizeof(float));
  float* tempCopy = (float*)malloc(size * sizeof(float));
  float* tempOut  = (float*) calloc(size,sizeof(float));

  int count = size;
  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  size_t global_work_size[2];                   
  size_t local_work_size[2];
  memcpy(tempCopy,tIn, size * sizeof(float));

  long long start = get_time();

  { // SYCL scope
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();
    buffer<float, 1> d_a (tIn,count, props);
    buffer<float, 1> d_b (pIn, count, props);
    buffer<float, 1> d_c (tempOut, count, props);


    global_work_size[0] = numCols;
    global_work_size[1] = numRows;

    local_work_size[0] = WG_SIZE_X;
    local_work_size[1] = WG_SIZE_Y;
    int j;
    for(j = 0; j < iterations; j++)
    {
      q.submit([&](handler& cgh) {
          auto p_acc = d_b.get_access<sycl_read>(cgh); 
          auto tIn_acc = d_a.get_access<sycl_read>(cgh);
          auto tOut_acc = d_c.get_access<sycl_write>(cgh);

          cgh.parallel_for<class hotspot>(
            nd_range<2>(range<2>(global_work_size[0], global_work_size[1]), 
              range<2>(local_work_size[0], local_work_size[1])), [=] (nd_item<2> item) {
#include "kernel_hotspot.sycl"
            });
          });

      auto temp = d_a;
      d_a         = d_c;
      d_c         = temp;
    }
  } // SYCL scope
  long long stop = get_time();

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tempOut,answer,numRows*numCols*layers);
  float time = (float)((stop - start)/(1000.0 * 1000.0));
  printf("Device offloading time: %.3f (s)\n",time);
  printf("Root-mean-square error: %e\n",acc);

  writeoutput(tempOut,numRows,numCols,layers,ofile);

  return 0;
}

