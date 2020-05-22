#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <getopt.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <vector>
#include "dwt.h"


#define THREADS 256

struct dwt {
  char * srcFilename;
  char * outFilename;
  unsigned char *srcImg;
  int pixWidth;
  int pixHeight;
  int components;
  int dwtLvls;
};


#include "device_functions.sycl"

template <typename T>
void memcpy_H2D(queue &q, T* mem_h, buffer<T,1> &mem_d, int num) {
  q.submit([&] (handler& cgh) {
      accessor<T,1,sycl_write,sycl_global_buffer>  \
      mem_d_acc (mem_d, cgh, range<1>(num), id<1>(0)); // add workgroup size
      cgh.copy(mem_h, mem_d_acc);
      });
  q.wait();
}


template <typename T>
void memcpy_D2H(queue &q, T* mem_h, buffer<T,1> &mem_d, int num) {
  q.submit([&] (handler& cgh) {
      accessor<T,1,sycl_read,sycl_global_buffer>  \
      mem_d_acc (mem_d, cgh, range<1>(num), id<1>(0)); // add workgroup size
      cgh.copy(mem_d_acc, mem_h);
      });
  q.wait();
}

/// param dest  destination bufer
/// param src   source buffer
/// param sx    width of copied image
/// param sy    height of copied image
///from /cuda_gwt/common.h/namespace
template <typename T>
void memcpy_D2D (queue &q, buffer<T,1> &dst,  buffer<T,1> &src, const size_t sx, const size_t sy){

  //errNum = clEnqueueCopyBuffer (commandQueue, src, dest, 0, 0, sx*sy*sizeof(int), 0, NULL, NULL);
  q.submit([&](handler& cgh) {
      auto dst_acc = dst.template get_access<sycl_write>(cgh);
      accessor<T, 1, sycl_read, sycl_global_buffer> 
      src_acc(src, cgh, range<1>(sx*sy), id<1>(0));  // add workgroup size 
      cgh.copy(src_acc, dst_acc);
      });
  q.wait();
}

///
// Load the input image.
//
int getImg(char * srcFilename, unsigned char *srcImg, int inputSize)
{
  char path[] = "../../data/dwt2d/";
  char *newSrc = NULL;

  if((newSrc = (char *)malloc(strlen(srcFilename)+strlen(path)+1)) != NULL)
  {
    newSrc[0] = '\0';
    strcat(newSrc, path);
    strcat(newSrc, srcFilename);
    srcFilename= newSrc;
  }
  printf("Loading ipnput: %s\n", srcFilename);

  //read image
  int i = open(srcFilename, O_RDONLY, 0644);
  if (i == -1) 
  { 
    error(0,errno,"cannot access %s", srcFilename);
    return -1;
  }
  int ret = read(i, srcImg, inputSize);
  printf("precteno %d, inputsize %d\n", ret, inputSize);
  close(i);

  return 0;
}

///
//Show user how to use this program
//
void usage() {
  printf("dwt [otpions] src_img.rgb <out_img.dwt>\n\
      -d, --dimension\t\tdimensions of src img, e.g. 1920x1080\n\
      -c, --components\t\tnumber of color components, default 3\n\
      -b, --depth\t\t\tbit depth, default 8\n\
      -l, --level\t\t\tDWT level, default 3\n\
      -D, --device\t\t\tcuda device\n\
      -f, --forward\t\t\tforward transform\n\
      -r, --reverse\t\t\treverse transform\n\
      -9, --97\t\t\t9/7 transform (not implemented in OpenCL or SYCL)\n\
      -5, --53\t\t\t5/3 transform\n\
      -w  --write-visual\t\twrite output in visual (tiled) fashion instead of the linear\n");
}




///
// Separate compoents of 8bit RGB source image
//in file components.cu
  template <typename T>
void rgbToComponents(queue &q, buffer<T,1> &d_r, buffer<T,1> &d_g, buffer<T,1> &d_b, unsigned char * h_src, int width, int height)
{
  int pixels      = width * height;
  int alignedSize =  DIVANDRND(width*height, THREADS) * THREADS * 3; //aligned to thread block size -- THREADS

  const property_list props = property::buffer::use_host_ptr();
  buffer<unsigned char, 1> d_src (h_src, 3*pixels, props) ;

  // fatal_CL(errNum, __LINE__);

  int globalWorkSize[1] = { alignedSize/3};
  int localWorkSize[1] = { THREADS };

  q.submit([&](handler& cgh) {
      auto d_r_acc = d_r.template get_access<sycl_write>(cgh);
      auto d_g_acc = d_g.template get_access<sycl_write>(cgh);
      auto d_b_acc = d_b.template get_access<sycl_write>(cgh);
      auto d_src_acc = d_src.get_access<sycl_read>(cgh);
      accessor <unsigned char, 1, sycl_read_write, access::target::local> sData (THREADS*3, cgh);

      cgh.parallel_for<class CopySrcToComponents>(
        nd_range<1>(range<1>(globalWorkSize[0]), range<1>(localWorkSize[0])), [=] (nd_item<1> item) {
#include "kernel_CopySrcToComponents.sycl"
        });
      });
}



///
// Copy a 8bit source image data into a color compoment
//in file components.cu
  template<typename T>
void bwToComponent(queue &q, buffer<T,1> &d_c, unsigned char * h_src, int width, int height)
{
  int pixels      = width*height;
  int alignedSize =  DIVANDRND(pixels, THREADS) * THREADS;

  const property_list props = property::buffer::use_host_ptr();
  buffer<unsigned char, 1> d_src (h_src, pixels, props) ;

  int globalWorkSize[1] = { alignedSize/9};
  int localWorkSize[1] = { THREADS };
  assert(alignedSize%(THREADS*3) == 0);

  q.submit([&](handler& cgh) {

      auto d_c_acc = d_c.template get_access<sycl_write>(cgh);
      auto d_src_acc = d_src.get_access<sycl_read>(cgh);
      accessor <unsigned char, 1, sycl_read_write, access::target::local> sData (THREADS, cgh);

      cgh.parallel_for<class CopySrcToComponent>(
        nd_range<1>(range<1>(globalWorkSize[0]), range<1>(localWorkSize[0])), [=] (nd_item<1> item) {
#include "kernel_CopySrcToComponent.sycl"
        });
      });

  std::cout<<"bwToComponent has finished\n";
}



/// Only computes optimal number of sliding window steps, number of threadblocks and then lanches the 5/3 FDWT kernel.
/// @tparam WIN_SX  width of sliding window
/// @tparam WIN_SY  height of sliding window
/// @param in       input image
/// @param out      output buffer
/// @param sx       width of the input image 
/// @param sy       height of the input image
///launchFDWT53Kerneld is in file 
  template<typename T>
void launchFDWT53Kernel (queue &q, int WIN_SX, int WIN_SY, buffer<T,1> &in, buffer<T,1> &out, int sx, int sy)
{
  // compute optimal number of steps of each sliding window
  // cuda_dwt called a function divRndUp from namespace cuda_gwt. this function takes n and d, "return (n / d) + ((n % d) ? 1 : 0);"
  //

  const int steps = ( sy/ (15 * WIN_SY)) + ((sy % (15 * WIN_SY)) ? 1 : 0);	

  int gx = ( sx/ WIN_SX) + ((sx %  WIN_SX) ? 1 : 0);  
  //use function divRndUp(n, d){return (n / d) + ((n % d) ? 1 : 0);}
  int gy = ( sy/ (WIN_SY*steps)) + ((sy %  (WIN_SY*steps)) ? 1 : 0);

  printf("sliding steps = %d , gx = %d , gy = %d \n", steps, gx, gy);

  // prepare grid size
  size_t globalWorkSize[2] = { (size_t)gx*WIN_SX, (size_t)gy};
  size_t localWorkSize[2]  = { (size_t)WIN_SX , 1};
  // printf("\n globalx=%d, globaly=%d, blocksize=%d\n", gx, gy, WIN_SX);

  q.submit([&](handler& cgh) {
      auto in_acc = in.template get_access<sycl_read>(cgh);
      auto out_acc = out.template get_access<sycl_write>(cgh);
      accessor <FDWT53, 1, sycl_read_write, access::target::local> fdwt53_acc (1, cgh);
      cgh.parallel_for<class fdwt53>(
        nd_range<2>(range<2>(globalWorkSize[0],globalWorkSize[1]), 
          range<2>(localWorkSize[0],localWorkSize[1])), [=] (nd_item<2> item) {
#include "kernel_fdwt53.sycl"
        });
      });
  printf("kl_fdwt53Kernel in launchFDW53Kernel has finished\n");
}





/// Forward 5/3 2D DWT. See common rules (above) for more details.
/// @param in      Expected to be normalized into range [-128, 127].
///                Will not be preserved (will be overwritten).
/// @param out     output buffer on GPU
/// @param sizeX   width of input image (in pixels)
/// @param sizeY   height of input image (in pixels)
/// @param levels  number of recursive DWT levels
/// @backup use to test time
//at the end of namespace dwt_cuda (line338)
  template <typename T>
void fdwt53(queue &q, buffer<T,1> in, buffer<T,1> out, int sizeX, int sizeY, int levels)
{
  // select right width of kernel for the size of the image

  if(sizeX >= 960) 
  {
    launchFDWT53Kernel(q, 192, 8, in, out, sizeX, sizeY);
  } 
  else if (sizeX >= 480) 
  {
    launchFDWT53Kernel(q, 128, 8, in, out, sizeX, sizeY);
  } else 
  {
    launchFDWT53Kernel(q, 64, 8, in, out, sizeX, sizeY);
  }

  // if this was not the last level, continue recursively with other levels
  if (levels > 1)
  {
    // copy output's LL band back into input buffer 
    const int llSizeX = (sizeX / 2) + ((sizeX % 2) ? 1 :0);
    const int llSizeY = (sizeY / 2) + ((sizeY % 2) ? 1 :0);
    memcpy_D2D(q, in, out, llSizeX, llSizeY);

    // run remaining levels of FDWT
    fdwt53(q, in, out, llSizeX, llSizeY, levels - 1);
  }	
}


///
// in dwt.cu
  template <typename T>
int nStage2dDWT(queue &q, buffer<T,1> &in, buffer<T,1> &out, buffer<T,1> &backup, int pixWidth, int pixHeight, int stages, bool forward)
{
  printf("\n*** %d stages of 2D forward DWT:\n", stages);

  // create backup of input, because each test iteration overwrites it 
  //const int size = pixHeight * pixWidth * sizeof(int);

  // Measure time of individual levels. 
  if (forward)
    fdwt53(q, in, out, pixWidth, pixHeight, stages );
  //else
  //	rdwt(in, out, pixWidth, pixHeight, stages);
  // rdwt means rdwt53(can be found in file rdwt53.cu) which has not been defined 


  return 0;
}



///
//in file dwt.cu
void samplesToChar(unsigned char * dst, int * src, int samplesNum)
{
  int i;

  for(i = 0; i < samplesNum; i++)
  {
    int r = src[i]+128;
    if (r > 255) r = 255;
    if (r < 0)   r = 0; 
    dst[i] = (unsigned char)r;
  }
}



///
//in file dwt.cu
/// Write output linear orderd
template<typename T>
int writeLinear(queue &q, buffer<T,1> &component, int pixWidth, int pixHeight, const char * filename, const char * suffix)
{
  unsigned char * result;
  int *gpu_output;
  int i;
  int size;
  int samplesNum = pixWidth*pixHeight;

  size = samplesNum*sizeof(int);
  gpu_output = (int *)malloc(size);
  memset(gpu_output, 0, size);
  result = (unsigned char *)malloc(samplesNum);

  memcpy_D2H(q, gpu_output, component, samplesNum);
  //errNum = clEnqueueReadBuffer(commandQueue, component, CL_TRUE, 0, size, gpu_output, 0, NULL, NULL);
  // fatal_CL(errNum, __LINE__);	
  //
#ifdef DEBUG
  printf("filename: %s\n", filename);
  for (int i = 0; i < samplesNum; i++)
    printf("%d %d\n", i, gpu_output[i]);
  printf("\n");
#endif

  // T to char 
  samplesToChar(result, gpu_output, samplesNum);

  // Write component 
  char outfile[strlen(filename)+strlen(suffix)];
  strcpy(outfile, filename);
  strcpy(outfile+strlen(filename), suffix);
  i = open(outfile, O_CREAT|O_WRONLY, 0644);
  if (i == -1) 
  { 
    error(0,errno,"cannot access %s", outfile);
    return -1;
  }
  printf("\nWriting to %s (%d x %d)\n", outfile, pixWidth, pixHeight);
  write(i, result, samplesNum);
  close(i);

  // Clean up 
  free(gpu_output);
  free(result);

  return 0;
}



///
// Write output visual ordered 
//in file dwt.cu
  template <typename T>
int writeNStage2DDWT(queue &q, buffer<T,1> &component, int pixWidth, int pixHeight, int stages, const char * filename, const char * suffix)
{
  struct band {
    int dimX; 
    int dimY;
  };
  struct dimensions {
    struct band LL;
    struct band HL;
    struct band LH;
    struct band HH;
  };

  unsigned char * result;
  int *src;
  int	*dst;
  int i,s;
  int size;
  int offset;
  int yOffset;
  int samplesNum = pixWidth*pixHeight;
  struct dimensions * bandDims;

  bandDims = (struct dimensions *)malloc(stages * sizeof(struct dimensions));

  bandDims[0].LL.dimX = DIVANDRND(pixWidth,2);
  bandDims[0].LL.dimY = DIVANDRND(pixHeight,2);
  bandDims[0].HL.dimX = pixWidth - bandDims[0].LL.dimX;
  bandDims[0].HL.dimY = bandDims[0].LL.dimY;
  bandDims[0].LH.dimX = bandDims[0].LL.dimX;
  bandDims[0].LH.dimY = pixHeight - bandDims[0].LL.dimY;
  bandDims[0].HH.dimX = bandDims[0].HL.dimX;
  bandDims[0].HH.dimY = bandDims[0].LH.dimY;

  for (i = 1; i < stages; i++) 
  {
    bandDims[i].LL.dimX = DIVANDRND(bandDims[i-1].LL.dimX,2);
    bandDims[i].LL.dimY = DIVANDRND(bandDims[i-1].LL.dimY,2);
    bandDims[i].HL.dimX = bandDims[i-1].LL.dimX - bandDims[i].LL.dimX;
    bandDims[i].HL.dimY = bandDims[i].LL.dimY;
    bandDims[i].LH.dimX = bandDims[i].LL.dimX;
    bandDims[i].LH.dimY = bandDims[i-1].LL.dimY - bandDims[i].LL.dimY;
    bandDims[i].HH.dimX = bandDims[i].HL.dimX;
    bandDims[i].HH.dimY = bandDims[i].LH.dimY;
  }

#if 0
  printf("Original image pixWidth x pixHeight: %d x %d\n", pixWidth, pixHeight);
  for (i = 0; i < stages; i++) 
  {
    printf("Stage %d: LL: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].LL.dimX, bandDims[i].LL.dimY);
    printf("Stage %d: HL: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].HL.dimX, bandDims[i].HL.dimY);
    printf("Stage %d: LH: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].LH.dimX, bandDims[i].LH.dimY);
    printf("Stage %d: HH: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].HH.dimX, bandDims[i].HH.dimY);
  }
#endif

  size = samplesNum*sizeof(int);	

  src = (int *)malloc(size);
  memset(src, 0, size);
  dst = (int *)malloc(size);
  memset(dst, 0, size);
  result = (unsigned char *)malloc(samplesNum);

  memcpy_D2H(q, src, component, samplesNum);
  //herrNum = clEnqueueReadBuffer(commandQueue, component, CL_TRUE, 0, size, src, 0, NULL, NULL);


  // LL Band 	
  size = bandDims[stages-1].LL.dimX * sizeof(int);
  for (i = 0; i < bandDims[stages-1].LL.dimY; i++) 
  {
    memcpy(dst+i*pixWidth, src+i*bandDims[stages-1].LL.dimX, size);
  }

  for (s = stages - 1; s >= 0; s--) {
    // HL Band
    size = bandDims[s].HL.dimX * sizeof(int);
    offset = bandDims[s].LL.dimX * bandDims[s].LL.dimY;
    for (i = 0; i < bandDims[s].HL.dimY; i++) 
    {
      memcpy(dst+i*pixWidth+bandDims[s].LL.dimX,
          src+offset+i*bandDims[s].HL.dimX, 
          size);
    }

    // LH band
    size = bandDims[s].LH.dimX * sizeof(int);
    offset += bandDims[s].HL.dimX * bandDims[s].HL.dimY;
    yOffset = bandDims[s].LL.dimY;
    for (i = 0; i < bandDims[s].HL.dimY; i++) 
    {
      memcpy(dst+(yOffset+i)*pixWidth,
          src+offset+i*bandDims[s].LH.dimX, 
          size);
    }

    //HH band
    size = bandDims[s].HH.dimX * sizeof(int);
    offset += bandDims[s].LH.dimX * bandDims[s].LH.dimY;
    yOffset = bandDims[s].HL.dimY;
    for (i = 0; i < bandDims[s].HH.dimY; i++) 
    {
      memcpy(dst+(yOffset+i)*pixWidth+bandDims[s].LH.dimX,
          src+offset+i*bandDims[s].HH.dimX, 
          size);
    }
  }

  // Write component
  samplesToChar(result, dst, samplesNum);	

  char outfile[strlen(filename)+strlen(suffix)];
  strcpy(outfile, filename);
  strcpy(outfile+strlen(filename), suffix);
  i = open(outfile, O_CREAT|O_WRONLY, 0644);

  if (i == -1) 
  {
    error(0,errno,"cannot access %s", outfile);
    return -1;
  }

  printf("\nWriting to %s (%d x %d)\n", outfile, pixWidth, pixHeight);
  write(i, result, samplesNum);
  close(i);

  free(src);
  free(dst);
  free(result);
  free(bandDims);

  return 0;
}





///
// Process of DWT algorithm
//
  template <typename T>
void processDWT(queue &q, struct dwt *d, int forward, int writeVisual)
{

  int pixelSize = d->pixWidth * d->pixHeight;
  int componentSize = pixelSize* sizeof(T);

  // initialize to zeros
  T *temp = (T *)malloc(componentSize);
  memset(temp, 0, componentSize);

  buffer<T, 1> cl_c_r_out (pixelSize);
  buffer<T, 1> cl_backup (pixelSize);

  if (d->components == 3) {
    buffer<T, 1> cl_c_g_out (pixelSize);
    buffer<T, 1> cl_c_b_out (pixelSize);

    buffer<T, 1> cl_c_g(pixelSize);
    buffer<T, 1> cl_c_b(pixelSize);
    buffer<T, 1> cl_c_r(pixelSize);

    memcpy_H2D(q, temp, cl_c_g_out, pixelSize);
    memcpy_H2D(q, temp, cl_c_b_out, pixelSize);
    memcpy_H2D(q, temp, cl_c_g, pixelSize);
    memcpy_H2D(q, temp, cl_c_b, pixelSize);
    memcpy_H2D(q, temp, cl_c_r, pixelSize);

    rgbToComponents(q, cl_c_r, cl_c_g, cl_c_b, d->srcImg, d->pixWidth, d->pixHeight);


    //Compute DWT and always store int file

    nStage2dDWT(q, cl_c_r, cl_c_r_out, cl_backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
    nStage2dDWT(q, cl_c_g, cl_c_g_out, cl_backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
    nStage2dDWT(q, cl_c_b, cl_c_b_out, cl_backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);


#ifdef OUTPUT        
    // Store DWT to file
    if(writeVisual){
      writeNStage2DDWT(q, cl_c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
      writeNStage2DDWT(q, cl_c_g_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".g");
      writeNStage2DDWT(q, cl_c_b_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".b");

    } else {
      writeLinear(q, cl_c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".r");
      writeLinear(q, cl_c_g_out, d->pixWidth, d->pixHeight, d->outFilename, ".g");
      writeLinear(q, cl_c_b_out, d->pixWidth, d->pixHeight, d->outFilename, ".b");
    }
#endif		

  } else if(d->components == 1) { 
    // Load components 
    buffer<T, 1> cl_c_r(pixelSize);
    memcpy_H2D(q, temp, cl_c_r, pixelSize);

    bwToComponent(q, cl_c_r, d->srcImg, d->pixWidth, d->pixHeight);

    // Compute DWT
    nStage2dDWT(q, cl_c_r, cl_c_r_out, cl_backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);

#ifdef OUTPUT        
    //Store DWT to file
    if(writeVisual){
      writeNStage2DDWT(q, cl_c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
    } else {
      writeLinear(q, cl_c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".r");
    }
#endif		

  } 

  free(temp);
}



int main(int argc, char **argv) 
{
  int optindex = 0;
  char ch;
  struct option longopts[] = 
  {
    {"dimension",   required_argument, 0, 'd'}, //dimensions of src img
    {"components",  required_argument, 0, 'c'}, //numger of components of src img
    {"depth",       required_argument, 0, 'b'}, //bit depth of src img
    {"level",       required_argument, 0, 'l'}, //level of dwt
    {"device",      required_argument, 0, 'D'}, //cuda device
    {"forward",     no_argument,       0, 'f'}, //forward transform
    {"reverse",     no_argument,       0, 'r'}, //forward transform
    {"97",          no_argument,       0, '9'}, //9/7 transform
    {"53",          no_argument,       0, '5' }, //5/3transform
    {"write-visual",no_argument,       0, 'w' }, //write output (subbands) in visual (tiled) order instead of linear
    {"help",        no_argument,       0, 'h'}  
  };

  int pixWidth    = 0; //<real pixWidth
  int pixHeight   = 0; //<real pixHeight
  int compCount   = 3; //number of components; 3 for RGB or YUV, 4 for RGBA
  int bitDepth    = 8; 
  int dwtLvls     = 3; //default numuber of DWT levels
  int device      = 0;
  int forward     = 1; //forward transform
  int dwt97       = 0; //1=dwt9/7, 0=dwt5/3 transform
  int writeVisual = 0; //write output (subbands) in visual (tiled) order instead of linear
  char * pos;

  while ((ch = getopt_long(argc, argv, "d:c:b:l:D:fr95wh", longopts, &optindex)) != -1) 
  {
    switch (ch) {
      case 'd':
        pixWidth = atoi(optarg);
        pos = strstr(optarg, "x");
        if (pos == NULL || pixWidth == 0 || (strlen(pos) >= strlen(optarg))) 
        {
          usage();
          return -1;
        }
        pixHeight = atoi(pos+1);
        break;
      case 'c':
        compCount = atoi(optarg);
        break;
      case 'b':
        bitDepth = atoi(optarg);
        break;
      case 'l':
        dwtLvls = atoi(optarg);
        break;
      case 'D':
        device = atoi(optarg);
        break;
      case 'f':
        forward = 1;
        break;
      case 'r':
        forward = 0;
        break;
      case '9':
        dwt97 = 1;
        break;
      case '5':
        dwt97 = 0;
        break;
      case 'w':
        writeVisual = 1;
        break;
      case 'h':
        usage();
        return 0;
      case '?':
        return -1;
      default :
        usage();
        return -1;
    }
  }
  argc -= optind;
  argv += optind;

  if (argc == 0) 
  { // at least one filename is expected
    printf("Please supply src file name\n");
    usage();
    return -1;
  }

  if (pixWidth <= 0 || pixHeight <=0) 
  {
    printf("Wrong or missing dimensions\n");
    usage();
    return -1;
  }

  if (forward == 0) 
  {
    writeVisual = 0; //do not write visual when RDWT
  }
  struct dwt *d;
  d = (struct dwt *)malloc(sizeof(struct dwt));
  d->srcImg = NULL;
  d->pixWidth = pixWidth;
  d->pixHeight = pixHeight;
  d->components = compCount;
  d->dwtLvls  = dwtLvls;

  // file names
  d->srcFilename = (char *)malloc(strlen(argv[0]));
  strcpy(d->srcFilename, argv[0]);
  if (argc == 1) 
  { // only one filename supplyed
    d->outFilename = (char *)malloc(strlen(d->srcFilename)+4);
    strcpy(d->outFilename, d->srcFilename);
    strcpy(d->outFilename+strlen(d->srcFilename), ".dwt");
  } else {
    d->outFilename = strdup(argv[1]);
  }

  //Input review
  printf("\nSource file:\t\t%s\n", d->srcFilename);
  printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
  printf(" Components count:\t%d\n", compCount);
  printf(" Bit depth:\t\t%d\n", bitDepth);
  printf(" DWT levels:\t\t%d\n", dwtLvls);
  printf(" Forward transform:\t%d\n", forward);
  printf(" 9/7 transform:\t\t%d\n", dwt97);

  //data sizes
  int inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess

  //load img source image
  d->srcImg = (unsigned char *) malloc (inputSize);
  if (getImg(d->srcFilename, d->srcImg, inputSize) == -1) 
    return -1;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  if(dwt97 == 1) {
    //processDWT<float>(q, d, forward, writeVisual);
    fprintf(stderr, "Warnning: dwt97 is not implemented in SYCL\n");
    return 0;
  }
  else // 5/3
    processDWT<int>(q, d, forward, writeVisual);

  return 0;

}
