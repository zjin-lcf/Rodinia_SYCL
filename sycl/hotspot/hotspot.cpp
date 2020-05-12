#include "common.h"
#include "hotspot.h"

// Returns the current system time in microseconds
long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 )
    printf( "The file was not opened\n" );


  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
    {

      sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
      fputs(str,fp);
      index++;
    }

  fclose(fp);	
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 ) {
    printf( "The file %s was not opened successfully", file );
    exit(-1);
  }


  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++)
    {
      if (fgets(str, STR_SIZE, fp) == NULL) {
        printf("Error reading file\n");
        exit(-1);
      }
      if (feof(fp)) {
        printf("not enough lines in file");
        exit(-1);
      }
      //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
      if ((sscanf(str, "%f", &val) != 1)) {
        printf("invalid file format");
        exit(-1);
      }
      vect[i*grid_cols+j] = val;
    }

  fclose(fp);	

}


/*
   compute N time steps
   */

int compute_tran_temp(queue &q, 
    buffer<float,1> &MatrixPower, 
    std::vector<buffer<float,1>> &MatrixTemp, 
    int col, int row, \
    int total_iterations, int num_iterations, 
    int blockCols, int blockRows, int borderCols, int borderRows)
{ 

  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  int t;
#ifdef DEBUG
  printf("%f %f %f %f %f %f %f\n", grid_height,grid_width,Cap,Rx,Ry,Rz,step);
#endif

  int src = 0, dst = 1;

  // Determine GPU work group grid
  size_t global_work_size[2];
  global_work_size[0] = BLOCK_SIZE * blockCols;
  global_work_size[1] = BLOCK_SIZE * blockRows;
  size_t local_work_size[2];
  local_work_size[0] = BLOCK_SIZE;
  local_work_size[1] = BLOCK_SIZE;

  //long long start_time = get_time();	

  for (t = 0; t < total_iterations; t += num_iterations) {

    // Specify kernel arguments
    int iter = MIN(num_iterations, total_iterations - t);

    q.submit([&](handler& cgh) {
        auto power_acc = MatrixPower.get_access<sycl_read>(cgh);
        auto temp_src_acc = MatrixTemp[src].get_access<sycl_read>(cgh);
        auto temp_dst_acc = MatrixTemp[dst].get_access<sycl_write>(cgh);
        accessor <float, 2, sycl_read_write, access::target::local> temp_on_device ({BLOCK_SIZE,BLOCK_SIZE}, cgh);
        accessor <float, 2, sycl_read_write, access::target::local> power_on_device ({BLOCK_SIZE,BLOCK_SIZE}, cgh);
        accessor <float, 2, sycl_read_write, access::target::local> temp_t ({BLOCK_SIZE,BLOCK_SIZE}, cgh);
        cgh.parallel_for<class hotspot>(
            nd_range<2>(range<2>(global_work_size[0], global_work_size[1]), 
                        range<2>(local_work_size[0], local_work_size[1])), [=] (nd_item<2> item) {
#include "kernel_hotspot.sycl"
            });
        });

    // Swap input and output GPU matrices
    src = 1 - src;
    dst = 1 - dst;
  }

  //long long end_time = get_time();
  //long long total_time = (end_time - start_time);	
  //printf("\nNumber of iterations: %d Kernel time: %.3f seconds\n", 
  //        total_iterations, ((float) total_time) / (1000*1000));

#ifdef DEBUG
  q.wait();
  auto vect_acc = MatrixTemp[src].get_access<sycl_read>();
  for (int i=0; i < 1; i++) 
    for (int j=0; j < 16; j++)
    {

      printf("%g\n", vect_acc[i*col+j]);
    }
#endif
  return src;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file\n");
  exit(1);
}

int main(int argc, char** argv) {

  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int size;
  int grid_rows,grid_cols = 0;
  float *FilesavingTemp,*FilesavingPower; //,*MatrixOut; 
  char *tfile, *pfile, *ofile;

  int total_iterations = 60;  // this can be overwritten by the commandline argument
  int pyramid_height = 1; // number of iterations

  if (argc < 7) usage(argc, argv);

  if((grid_rows = atoi(argv[1]))<=0||
     (grid_cols = atoi(argv[1]))<=0||
     (pyramid_height = atoi(argv[2]))<=0||
     (total_iterations = atoi(argv[3]))<=0)
    usage(argc, argv);

  tfile=argv[4];
  pfile=argv[5];
  ofile=argv[6];

  size=grid_rows*grid_cols;

  // --------------- pyramid parameters --------------- 
  int borderCols = (pyramid_height)*EXPAND_RATE/2;
  int borderRows = (pyramid_height)*EXPAND_RATE/2;
  int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
  int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
  int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
  int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

  FilesavingTemp = (float *) malloc(size*sizeof(float));
  FilesavingPower = (float *) malloc(size*sizeof(float));
  // MatrixOut = (float *) calloc (size, sizeof(float));

  if( !FilesavingPower || !FilesavingTemp) {
    printf("unable to allocate memory");
    exit(-1);
  }

  // Read input data from disk
  readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
  readinput(FilesavingPower, grid_rows, grid_cols, pfile);

  long long start_time = get_time();
  { 
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();
    buffer<float, 1> MatrixPower(FilesavingPower, size, props);

    std::vector<buffer<float>> MatrixTemp;
    MatrixTemp.emplace_back(FilesavingTemp, size, props);
    MatrixTemp.emplace_back(size);
    MatrixTemp[0].set_final_data(nullptr);
    MatrixTemp[1].set_final_data(nullptr);

    // Perform the computation
    int ret = compute_tran_temp(q, MatrixPower, MatrixTemp, grid_cols, grid_rows, 
        total_iterations, pyramid_height, blockCols, blockRows, borderCols, borderRows);

    q.submit([&](handler& cgh) {
        auto gpuResult_acc = MatrixTemp[ret].get_access<sycl_read>(cgh);
        cgh.copy(gpuResult_acc, FilesavingPower);
    });
  } // SYCL scope

  long long end_time = get_time();	
  printf("Total offload time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));

  // Write final output to output file
  writeoutput(FilesavingPower, grid_rows, grid_cols, ofile);

  free(FilesavingTemp);
  free(FilesavingPower);

  return 0;
}
