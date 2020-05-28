#include <sys/time.h>
#include "nearestNeighbor.h"
#include "common.h"

long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) +tv.tv_usec;
}

int main(int argc, char *argv[]) {
  std::vector<Record> records;
  float *recordDistances;
  //LatLong locations[REC_WINDOW];
  std::vector<LatLong> locations;
  int i;
  // args
  char filename[100];
  int resultsCount=10,quiet=0,timing=0;
  float lat=0.0,lng=0.0;
  
  // parse command line
  if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing)) {
    printUsage();
    return 0;
  }
  
  int numRecords = loadData(filename,records,locations);
  
  //for(i=0;i<numRecords;i++)
  //    printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);


  if (!quiet) {
    printf("Number of records: %d\n",numRecords);
    printf("Finding the %d closest neighbors.\n",resultsCount);
  }

  if (resultsCount > numRecords) resultsCount = numRecords;

  long long offload_start = get_time();
  recordDistances = (float *)malloc(sizeof(float) * numRecords);
  SyclFindNearestNeighbors(numRecords,locations,lat,lng,recordDistances,timing);
  long long offload_end = get_time();
  if (timing)
    printf("Device offloading time %lld (us)\n", offload_end - offload_start);

  // find the resultsCount least distances
  findLowest(records,recordDistances,numRecords,resultsCount);


  // print out results
  if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
  free(recordDistances);
  return 0;
}

void SyclFindNearestNeighbors(
	int numRecords,
	std::vector<LatLong> &locations,float lat,float lng,
	float* distances,
	int timing) {

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

#ifdef DEBUG
  try {
#endif

    const property_list props = property::buffer::use_host_ptr();
    buffer<LatLong,1> d_locations(locations.data(), numRecords,props);
    buffer<float,1> d_distances(distances, numRecords,props);

    // 4. enqueue kernel
    size_t globalWorkSize[1];
    globalWorkSize[0] = numRecords;
    if (numRecords % 64) globalWorkSize[0] += 64 - (numRecords % 64);
    size_t localWorkSize[0];
    localWorkSize[0] = 64;
#ifdef DEBUG
    printf("Global Work Size: %zu\n",globalWorkSize[0]);      
    printf("Local Work Size: %zu\n",localWorkSize[0]);      
#endif

        q.submit([&](handler& cgh) {

        auto d_locations_acc = d_locations.get_access<sycl_read>(cgh);
        auto d_distances_acc = d_distances.get_access<sycl_write>(cgh);

        cgh.parallel_for<class nn>(
            nd_range<1>(range<1>(globalWorkSize[0]), 
		        range<1>(localWorkSize[0])), [=] (nd_item<1> item) {
#include "kernel_nn.sycl"

            });
        });
#ifdef DEBUG
	 auto h_distances_acc =	d_distances.get_access<sycl_read>();
		for (int i = 0; i < 10; i++) {
			printf("%f ", h_distances_acc[i]);
		}
		printf("\n");
   } catch (cl::sycl::exception e) {
    std::cout << e.what() << std::endl;
#ifdef __COMPUTECPP__
    std::cout << e.get_file_name() << std::endl;
    std::cout << e.get_line_number() << std::endl;
    std::cout << e.get_description() << std::endl;
    std::cout << e.get_cl_error_message() << std::endl;
    std::cout << e.get_cl_code() << std::endl;
#endif
    return;
  }
  catch (std::exception e) {
    std::cout << e.what() << std::endl;
    return;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return;
  }

#endif
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;
	
    /**Main processing **/
    
    flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		* Read in REC_WINDOW records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;
            
            // parse for lat and long
            char substr[6];
            
            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);
            
            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);
            
            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;
  
  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;
    
    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;
    
    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t) {
    int i;
    if (argc < 2) return 1; // error
    strncpy(filename,argv[1],100);
    char flag;
    
    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
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
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] \n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
}

