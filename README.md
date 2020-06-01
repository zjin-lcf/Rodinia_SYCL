##  Rodinia Benchmark Suite in SYCL

The benchmark can be used for CPU, GPU, FPGA, or other architectures that support OpenCL and SYCL. It was written with GPUs in mind, so targeting other architectures may require heavy optimization.

##  Prerequisites

Intel<sup>®</sup> DPC++ compiler in OneAPI Base Toolkit (https://software.intel.com/oneapi/base-kit)  
Codeplay ComputeCpp<sup>™</sup> (https://www.codeplay.com/products/computesuite/computecpp) 


## Compilation

To compile each benchmark with the default settings, navigate to your selected source directory and use the following command:

```bash
make
```

 You can alter compiler settings in the included Makefile. For example, use the Codeplay SYCL compiler
```bash
make VENDOR=codeplay
```

### Debugging, Optimization 

There are also a number of switches that can be set in the makefile. Here is a sample of the control panel at the top of the makefile:

```bash
OPTIMIZE = no
DEBUG    = yes
```
- Optimization enables the -O3 optimization flag
- Debugging enables the -g and/or -DDEBUG flag 

## Running a benchmark

To run a benchmark, use the following command:
```bash
make run
```

Note the dataset, which is needed for certain benchmarks, can be downloaded at http://lava.cs.virginia.edu/Rodinia/download.htm.

## Running all benchmarks

A bash script is provided to attempt to run all the benchmarks:
```bash
./run_all.sh
```

## Development Team
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) with help from Nevin Liber

## Contributions
Comments and suggestions are welcome. 
