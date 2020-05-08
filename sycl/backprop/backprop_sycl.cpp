// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "backprop.h"

#include "common.h"

////////////////////////////////////////////////////////////////////////////////


double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
  int
main( int argc, char** argv) 
{
  setup(argc, argv);
}



int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  float * partial_sum;
  float sum;
  unsigned int num_blocks = in / BLOCK_SIZE;

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));

  // set global and local workitems
  size_t global_work[3] = { BLOCK_SIZE, BLOCK_SIZE * num_blocks, 1 }; 
  size_t local_work[3] = { BLOCK_SIZE, BLOCK_SIZE, 1 };

  // this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
  // todo: fix mem allocation
  int m = 0;
  for (int k = 0; k <= in; k++) {	
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
      m++;
    }
  }

  printf("Performing GPU computation\n");

  double offload_start = get_time();
  { // SYCL scope
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();

    buffer<float,1> input_sycl (net->input_units, in+1, props);
    buffer<float,1> input_weights_sycl (input_weights_one_dim, (in+1)*(hid+1), props);
    buffer<float,1> hidden_partial_sum (partial_sum, num_blocks*WIDTH, props);

    buffer<float,1> hidden_delta_sycl (net->hidden_delta, hid+1, props);
    buffer<float,1> input_prev_weights_sycl (input_weights_prev_one_dim, (in+1)*(hid+1), props);

    q.submit([&](handler& cgh) {

        auto input_acc = input_sycl.get_access<sycl_read>(cgh);
        auto input_weights_acc = input_weights_sycl.get_access<sycl_read_write>(cgh);
        auto hidden_partial_sum_acc = hidden_partial_sum.get_access<sycl_write>(cgh);
        accessor <float, 1, sycl_read_write, access::target::local> input_node (HEIGHT, cgh);
        accessor <float, 1, sycl_read_write, access::target::local> weight_matrix (HEIGHT * WIDTH, cgh);

        cgh.parallel_for<class forward>(
            nd_range<2>(range<2>(global_work[0], global_work[1]),
              range<2>(local_work[0],local_work[1])), [=] (nd_item<2> item) {
#include "bpnn_layerforward.sycl"
            });
        });

    q.submit([&](handler& cgh) {
        accessor<float, 1, access::mode::read, access::target::global_buffer> 
        hidden_partial_sum_acc(hidden_partial_sum, cgh, range<1>(num_blocks * WIDTH), id<1>(0));
        cgh.copy(hidden_partial_sum_acc, partial_sum);
        });
    q.wait();

    // use host accessor instead of copy from device to host
    //auto h_partial_sum_acc = hidden_partial_sum.get_access<sycl_read>(); 	

    for (int j = 1; j <= hid; j++) {
      sum = 0.0;
      for (int k = 0; k < num_blocks; k++) {	
        sum += partial_sum[k * hid + j-1] ;
        //sum += h_partial_sum_acc[k * hid + j-1] ;
      }
#ifdef DEBUG
      printf("j=%d sum=%f\n", j,sum);
#endif
      sum += net->input_weights[0][j];
      net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }

    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

    // copy 
    // err = clEnqueueWriteBuffer(cmd_queue, input_hidden_ocl,       1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, 0);
    // input_weights_sycl has been written in the first kernel, so it needs
    // to be restored.
    q.submit([&](handler& cgh) {
        accessor<float, 1, access::mode::write, access::target::global_buffer> 
        input_weights_acc(input_weights_sycl, cgh, range<1>((in + 1) * (hid + 1)), id<1>(0));
        cgh.copy(input_weights_one_dim, input_weights_acc);
        });

    q.submit([&](handler& cgh) {

        auto delta_acc = hidden_delta_sycl.get_access<sycl_read>(cgh);
        auto ly_acc = input_sycl.get_access<sycl_read>(cgh);
        auto w_acc = input_weights_sycl.get_access<sycl_read_write>(cgh);
        auto oldw_acc = input_prev_weights_sycl.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class adjust_weights>(
            nd_range<2>(range<2>(global_work[0], global_work[1]),
              range<2>(local_work[0],local_work[1])), [=] (nd_item<2> item) {
#include "bpnn_adjust_weights.sycl"
            });
        });
  } // SYCL scope
  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

#ifdef OUTPUT
  for (int i = 0; i < (in+1); i++) 
    printf("i=%d input_units=%f\n", i,net->input_units[i]);
  for (int i = 0; i < (in+1)*(hid+1); i++) 
    printf("i=%d input_weights=%f\n", i,input_weights_one_dim[i]);
#endif
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);

  return 0;
}
