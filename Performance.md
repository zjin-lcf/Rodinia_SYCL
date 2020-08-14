##  Performance Evaluation of the Rodinia Benchmark Suite in SYCL

###### Software 
Intel<sup>®</sup> oneAPI Beta08 Toolkit, Ubuntu 18.04
Codeplay ComputeCpp<sup>™</sup> CE 2.1.0
###### Platform
Intel<sup>®</sup> Xeon E-2176G with a Gen9.5 UHD630 integrated GPU


### Device offloading time in seconds (PI: plugin interface)

| Name | OpenCL PI | Level0 PI | ComputeCpp |
| --- | --- | --- | 
| b++tree | 0.86/0.54 | 0.89/0.52 | 0.26/0.09 |
| backprop | 0.36 | 0.4 | 0.26 |
| bfs | 0.36 | 0.4 | 0.27 |
| cfd | 6.3 | 10.5 | 10.2 |
| dwt2d | 0.69 | 0.73 | 0.43 |
| gaussian | 0.54 | 0.88 | 0.75 |
| heatwall | 26.9 | 27.8 | 8.4 |
| hotspot | 0.38 | 0.41 | 0.26 |
| hotspot3D | 0.55 | 0.61 | 0.49 |
| huffman | 0.0095 | 0.016 | 0.013 |
| hybridsort | 0.73 | 0.73 | NA |
| kmeans | 1.0 | 1.04 | 0.83 |
| lavaMD | 0.4 | 0.44 | 0.31 |
| leukocyte | 0.49 | 0.53 | 0.33 |
| lud | 1.42 | 1.55 | 1.42 |
| myoctye | 3.4 | 6.2 | 5.6 |
| nn | 0.31 | 0.35 | 0.24 |
| nw | 0.53 | 0.61 | 0.47 |
| particle-filter | 54.8 | 54.8 | 51.1 |
| pathfinder | 0.43 | 0.47 | 0.36 |
| srad | 1.0 | 1.3 | 0.92 |
| streamcluster | 9.9 | 14.1 | 9.9 |

