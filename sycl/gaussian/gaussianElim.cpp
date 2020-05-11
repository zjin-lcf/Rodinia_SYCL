#ifndef __GAUSSIAN_ELIMINATION__
#define __GAUSSIAN_ELIMINATION__

#include <math.h>
#include <sys/time.h>
#include "gaussianElim.h"
#include "common.h"

#define BLOCK_SIZE_0 256
#define BLOCK_SIZE_1_X 16
#define BLOCK_SIZE_1_Y 16


long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) +tv.tv_usec;
}

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
  {
    coe_i = 10*exp(lamda*i); 
    j=size-1+i;     
    coe[j]=coe_i;
    j=size-1-i;     
    coe[j]=coe_i;
  }


  for (i=0; i < size; i++) {
    for (j=0; j < size; j++) {
      m[i*size+j]=coe[size-1-i+j];
    }
  }
}


int main(int argc, char *argv[]) {

  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", BLOCK_SIZE_0, BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y);
  float *a=NULL, *b=NULL, *finalVec=NULL;
  float *m=NULL;
  int size = -1;

  FILE *fp;

  // args
  char filename[200];
  int quiet=0,timing=0;

  // parse command line
  if (parseCommandline(argc, argv, filename,
        &quiet, &timing, &size)) {
    printUsage();
    return 0;
  }


  if(size < 1)
  {
    fp = fopen(filename, "r");
    fscanf(fp, "%d", &size);

    a = (float *) malloc(size * size * sizeof(float));
    InitMat(fp,size, a, size, size);

    b = (float *) malloc(size * sizeof(float));
    InitAry(fp, b, size);

    fclose(fp);

  }
  else
  {
    printf("create input internally before create, size = %d \n", size);

    a = (float *) malloc(size * size * sizeof(float));
    create_matrix(a, size);

    b = (float *) malloc(size * sizeof(float));
    for (int i =0; i< size; i++)
      b[i]=1.0;

  }

  if (!quiet) {    
    printf("The input matrix a is:\n");
    PrintMat(a, size, size, size);

    printf("The input array b is:\n");
    PrintAry(b, size);
  }

  // create the solution matrix
  m = (float *) malloc(size * size * sizeof(float));

  // create a new vector to hold the final answer

  finalVec = (float *) malloc(size * sizeof(float));

  InitPerRun(size,m);

  long long offload_start = get_time();
  ForwardSub(a,b,m,size,timing);
  long long offload_end = get_time();

  if (timing) {
    printf("offload time %lld (us)\n\n",offload_end - offload_start);
  }


  //end timing
  if (!quiet) {
    printf("The result of array a is after forwardsub: \n");
    PrintMat(a, size, size, size);
    printf("The result of array b is after forwardsub: \n");
    PrintAry(b, size);
    printf("The result of matrix m is after forwardsub: \n");
    PrintMat(m, size, size, size);


    BackSub(a,b,finalVec,size);
    printf("The final solution is: \n");
    PrintAry(finalVec,size);
  }

  free(m);
  free(a);
  free(b);
  free(finalVec);
  //OpenClGaussianElimination(context,timing);

  return 0;
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub(float *a, float *b, float *m, int size,int timing){    
  // 1. set up kernels

  // 2. set up memory on device and send ipts data to device


  // 3. Determine block sizes
  int globalWorksizeFan1[1];
  int globalWorksizeFan2[2];
  int localWorksizeFan1Buf[1]={BLOCK_SIZE_0};
  int localWorksizeFan2Buf[2]={BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y};
  int *localWorksizeFan1=NULL;
  int *localWorksizeFan2=NULL;

  globalWorksizeFan1[0] = size;
  globalWorksizeFan2[0] = size;
  globalWorksizeFan2[1] = size;

  if(localWorksizeFan1Buf[0]){
    localWorksizeFan1=localWorksizeFan1Buf;
    globalWorksizeFan1[0]=(int)ceil(globalWorksizeFan1[0]/(double)localWorksizeFan1Buf[0])*localWorksizeFan1Buf[0];
  }
  if(localWorksizeFan2Buf[0]){
    localWorksizeFan2=localWorksizeFan2Buf;
    globalWorksizeFan2[0]=(int)ceil(globalWorksizeFan2[0]/(double)localWorksizeFan2Buf[0])*localWorksizeFan2Buf[0];
    globalWorksizeFan2[1]=(int)ceil(globalWorksizeFan2[1]/(double)localWorksizeFan2Buf[1])*localWorksizeFan2Buf[1];
  }

  printf("fan1: local_work_size = %d, global_work_size = %d \n",
      localWorksizeFan1Buf[0], globalWorksizeFan1[0]);
  printf("fan2: local_work_size = %d, %d, global_work_size = %d, %d \n",
      localWorksizeFan2Buf[0], localWorksizeFan2Buf[1], 
      globalWorksizeFan2[0], globalWorksizeFan2[1]);

  { // SYCL scope

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();
    buffer<float,1> d_a (a, size*size, props);
    buffer<float,1> d_b(b, size, props);
    buffer<float,1> d_m(m, size*size, props);

    // 4. Setup and Run kernels
    for (int t=0; t<(size-1); t++) {
      q.submit([&](handler& cgh) {

          auto a_acc = d_a.get_access<sycl_read>(cgh);
          auto m_acc = d_m.get_access<sycl_write>(cgh);

          cgh.parallel_for<class fan1>(
            nd_range<1>(range<1>(globalWorksizeFan1[0]), 
                        range<1>(localWorksizeFan1Buf[0])), [=] (nd_item<1> item) {
            int globalId = item.get_global_id(0);
            if (globalId < size-1-t) {
              m_acc[size * (globalId + t + 1)+t] = 
              a_acc[size * (globalId + t + 1) + t] / a_acc[size * t + t];
              }
            });
          });

      q.submit([&](handler& cgh) {

          auto a_acc = d_a.get_access<sycl_read_write>(cgh);
          auto b_acc = d_b.get_access<sycl_read_write>(cgh);
          auto m_acc = d_m.get_access<sycl_read>(cgh);

          cgh.parallel_for<class fan2>(
            nd_range<2>(range<2>(globalWorksizeFan2[0],globalWorksizeFan2[1]), 
                        range<2>(localWorksizeFan2Buf[0], localWorksizeFan2Buf[1])), [=] (nd_item<2> item) {
            int globalIdx = item.get_global_id(0);
            int globalIdy = item.get_global_id(1);
            if (globalIdx < size-1-t && globalIdy < size-t) {
              a_acc[size*(globalIdx+1+t)+(globalIdy+t)] -= 
              m_acc[size*(globalIdx+1+t)+t] * a_acc[size*t+(globalIdy+t)];

              if(globalIdy == 0){
                b_acc[globalIdx+1+t] -= 
                m_acc[size*(globalIdx+1+t)+(globalIdy+t)] * b_acc[t];
              }
            }
            });
      });

    } // for (t=0; t<(size-1); t++) 
  } // SYCL scope

}


// Ke Wang add a function to generate input internally
int parseCommandline(int argc, char *argv[], char* filename,
    int *q, int *t, int *size){
  int i;
  if (argc < 2) return 1; // error
  // strncpy(filename,argv[1],100);
  char flag;

  for(i=1;i<argc;i++) {
    if (argv[i][0]=='-') {// flag
      flag = argv[i][1];
      switch (flag) {
        case 's': // matrix size
          i++;
          *size = atoi(argv[i]);
          printf("Create matrix internally in parse, size = %d \n", *size);
          break;
        case 'f': // file name
          i++;
          strncpy(filename,argv[i],100);
          printf("Read file from %s \n", filename);
          break;
        case 'h': // help
          return 1;
          break;
        case 'q': // quiet
          *q = 1;
          break;
        case 't': // timing
          *t = 1;
          break;
      }
    }
  }
  return 0;
}

void printUsage(){
  printf("Gaussian Elimination Usage\n");
  printf("\n");
  printf("gaussianElimination -f [filename] [-hqt]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./gaussianElimination matrix4.txt\n");
  printf("\n");
  printf("filename     the filename that holds the matrix data\n");
  printf("\n");
  printf("-h           Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("-s           Specifiy the matrix size when the path to a matrix data file is not set.\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}


/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(int size,float *m) 
{
  int i;
  for (i=0; i<size*size; i++)
    *(m+i) = 0.0;
}
void BackSub(float *a, float *b, float *finalVec, int size)
{
  // solve "bottom up"
  int i,j;
  for(i=0;i<size;i++){
    finalVec[size-i-1]=b[size-i-1];
    for(j=0;j<i;j++)
    {
      finalVec[size-i-1]-=*(a+size*(size-i-1)+(size-j-1)) * finalVec[size-j-1];
    }
    finalVec[size-i-1]=finalVec[size-i-1]/ *(a+size*(size-i-1)+(size-i-1));
  }
}
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol)
{
  int i, j;

  for (i=0; i<nrow; i++) {
    for (j=0; j<ncol; j++) {
      fscanf(fp, "%f",  ary+size*i+j);
    }
  }  
}
/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(FILE *fp, float *ary, int ary_size)
{
  int i;

  for (i=0; i<ary_size; i++) {
    fscanf(fp, "%f",  &ary[i]);
  }
}  
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int size, int nrow, int ncol)
{
  int i, j;

  for (i=0; i<nrow; i++) {
    for (j=0; j<ncol; j++) {
      printf("%8.2e ", *(ary+size*i+j));
    }
    printf("\n");
  }
  printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
  int i;
  for (i=0; i<ary_size; i++) {
    printf("%.2e ", ary[i]);
  }
  printf("\n\n");
}
#endif

