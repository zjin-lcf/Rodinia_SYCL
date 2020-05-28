#!/bin/bash

run() {
	for (( i = 0; i < 10; i++ ))
	do
	  make run
	  [ $? -ne 0 ] && break 
	done
}

# CPATH may be changed accordingly
run_dpcpp () {
	export CPATH=/opt/intel/inteloneapi/compiler/latest/linux/include/sycl
	make clean; make; run
	make clean; make DEVICE=cpu; run
}

# CPATH may be changed accordingly
run_codeplay () {
	export CPATH=/home/cc/ComputeCpp-2.0.0/include:/opt/intel/inteloneapi/compiler/latest/linux/include/sycl
	make clean; make VENDOR=codeplay; run
	make clean; make DEVICE=cpu VENDOR=codeplay; run
}


# It should the following directories
#bfs  dwt2d     heartwall  hotspot3D   kmeans  lud      nn  particlefilter  srad
#backprop  cfd  gaussian  hotspot    hybridsort  lavaMD  myocyte  nw  pathfinder      streamcluster

for dir in `find . -mindepth 1 -maxdepth 1 -type d | grep -v '.\.git'`
do
	cd ${dir}
	echo "######## Start ${dir} #########"
	run_dpcpp
	run_codeplay
	echo "######## Finish ${dir} #########"
	cd ..
done
