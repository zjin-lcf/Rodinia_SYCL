/********************************************************************
  euler3d.cpp
  : parallelized code of CFD

  - original code from the AIAA-2009-4001 by Andrew Corrigan, acorriga@gmu.edu
  - parallelization with OpenCL API has been applied by
  Jianbin Fang - j.fang@tudelft.nl
  Delft University of Technology
  Faculty of Electrical Engineering, Mathematics and Computer Science
  Department of Software Technology
  Parallel and Distributed Systems Group
  on 24/03/2011
 ********************************************************************/

#include <iostream>
#include <fstream>
#include <math.h>
#include "common.h"
#include "util.h"

/*
 * Options 
 * 
 */ 
#define GAMMA 1.4f
#define iterations 2000
#ifndef block_length
#define block_length 192
#endif

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)
/*
 * not options
 */


#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

//self-defined user type
typedef struct{
  float x;
  float y;
  float z;
} Float3;

inline void compute_velocity(float  density, Float3 momentum, Float3* velocity){
  velocity->x = momentum.x / density;
  velocity->y = momentum.y / density;
  velocity->z = momentum.z / density;
}

inline float compute_speed_sqd(Float3 velocity){
  return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline float compute_pressure(float density, float density_energy, float speed_sqd){
  return ((float)(GAMMA) - (float)(1.0f))*(density_energy - (float)(0.5f)*density*speed_sqd);
}
// sqrt is a device function
inline float compute_speed_of_sound(float density, float pressure){
  return cl::sycl::sqrt((float)(GAMMA)*pressure/density);
}
inline void compute_flux_contribution(float density, Float3 momentum, float density_energy, float pressure, Float3 velocity, Float3* fc_momentum_x, Float3* fc_momentum_y, Float3* fc_momentum_z, Float3* fc_density_energy)
{
  fc_momentum_x->x = velocity.x*momentum.x + pressure;
  fc_momentum_x->y = velocity.x*momentum.y;
  fc_momentum_x->z = velocity.x*momentum.z;


  fc_momentum_y->x = fc_momentum_x->y;
  fc_momentum_y->y = velocity.y*momentum.y + pressure;
  fc_momentum_y->z = velocity.y*momentum.z;

  fc_momentum_z->x = fc_momentum_x->z;
  fc_momentum_z->y = fc_momentum_y->z;
  fc_momentum_z->z = velocity.z*momentum.z + pressure;

  float de_p = density_energy+pressure;
  fc_density_energy->x = velocity.x*de_p;
  fc_density_energy->y = velocity.y*de_p;
  fc_density_energy->z = velocity.z*de_p;
}


template <typename T>
void copy(queue &q, buffer<T,1> &dst, buffer<T,1> &src, int N){

  q.submit([&](handler& cgh) {
      accessor<T, 1, access::mode::write, sycl_global_buffer> 
      dst_acc(dst, cgh, range<1>(N), id<1>(0));  // add workgroup size
      accessor<T, 1, sycl_read, sycl_global_buffer> 
      src_acc(src, cgh, range<1>(N), id<1>(0));  // add workgroup size 
      cgh.copy(src_acc, dst_acc);
      });
}

void dump(queue &q, buffer<float,1> &variables, int nel, int nelr){
  float* h_variables = new float[nelr*NVAR];
  //download(h_variables, variables, nelr*NVAR);
  q.submit([&](handler& cgh) {
      accessor<float, 1, sycl_read, sycl_global_buffer> 
      variables_acc(variables, cgh, range<1>(nelr*NVAR), id<1>(0));  // add workgroup size
      cgh.copy(variables_acc, h_variables);
      });

  q.wait();

  {
    std::ofstream file("density");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
  }


  {
    std::ofstream file("momentum");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++)
    {
      for(int j = 0; j != NDIM; j++)
        file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
      file << std::endl;
    }
  }

  {
    std::ofstream file("density_energy");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
  }
  delete[] h_variables;
}

void initialize_buffer(queue &q, buffer<float,1> &mem_d, float val, int number_words)throw(std::string){
  q.submit([&] (handler& cgh) {
      accessor<float,1,sycl_write,sycl_global_buffer>  \
      mem_d_acc (mem_d, cgh, range<1>(number_words), id<1>(0)); // add workgroup size
      cgh.fill(mem_d_acc, val);
      });
}

void initialize_variables(queue &q, int nelr, buffer<float,1> &variables, buffer<float,1> &ff_variable) throw(std::string){

  int work_items = nelr;
  int work_group_size = BLOCK_SIZE_1;

  q.submit([&](handler& cgh) {
      auto variables_acc = variables.get_access<sycl_write>(cgh);
      auto ff_variable_acc = ff_variable.get_access<sycl_read>(cgh);

      cgh.parallel_for<class init_vars>(
        nd_range<1>(range<1>(work_items), range<1>(work_group_size)), [=] (nd_item<1> item) {
#include "kernel_initialize_variables.sycl"
        });
      });
}

void compute_step_factor(queue &q, int nelr, 
    buffer<float,1> &variables, 
    buffer<float,1> &areas, 
    buffer<float,1> &step_factors){

  int work_items = nelr;
  int work_group_size = BLOCK_SIZE_2;


  q.submit([&](handler& cgh) {
      auto variables_acc = variables.get_access<sycl_read>(cgh);
      auto areas_acc = areas.get_access<sycl_read>(cgh);
      auto step_factors_acc = step_factors.get_access<sycl_write>(cgh);

      cgh.parallel_for<class compute_step_factor>(
        nd_range<1>(range<1>(work_items), range<1>(work_group_size)), [=] (nd_item<1> item) {
#include "kernel_compute_step_factor.sycl"
        });
      });
}

void compute_flux(queue &q, 
    int nelr, 
    buffer<int,1> &elements_surrounding_elements,
    buffer<float,1> &normals,
    buffer<float,1> &variables,
    buffer<float,1> &ff_variable,
    buffer<float,1> &fluxes,
    buffer<Float3,1> &ff_flux_contribution_density_energy,
    buffer<Float3,1> &ff_flux_contribution_momentum_x,
    buffer<Float3,1> &ff_flux_contribution_momentum_y,
    buffer<Float3,1> &ff_flux_contribution_momentum_z){

  int work_items = nelr;
  int work_group_size = BLOCK_SIZE_3;

  q.submit([&](handler& cgh) {

      auto elements_surrounding_elements_acc = elements_surrounding_elements.get_access<sycl_read>(cgh);
      auto normals_acc = normals.get_access<sycl_read>(cgh);
      auto variables_acc = variables.get_access<sycl_read>(cgh);
      auto ff_variable_acc = ff_variable.get_access<sycl_read>(cgh);
      auto fluxes_acc = fluxes.get_access<sycl_write>(cgh);
      auto ff_flux_contribution_density_energy_acc = ff_flux_contribution_density_energy.get_access<sycl_read>(cgh);
      auto ff_flux_contribution_momentum_x_acc = ff_flux_contribution_momentum_x.get_access<sycl_read>(cgh);
      auto ff_flux_contribution_momentum_y_acc = ff_flux_contribution_momentum_y.get_access<sycl_read>(cgh);
      auto ff_flux_contribution_momentum_z_acc = ff_flux_contribution_momentum_z.get_access<sycl_read>(cgh);

      cgh.parallel_for<class compute_flux>(
        nd_range<1>(range<1>(work_items), range<1>(work_group_size)), [=] (nd_item<1> item) {
#include "kernel_compute_flux.sycl"
        });
      });
}

void time_step(queue &q, int j, int nelr, 
    buffer<float,1> &old_variables, 
    buffer<float,1> &variables, 
    buffer<float,1> &step_factors, 
    buffer<float,1> &fluxes) {

  int work_items = nelr;
  int work_group_size = BLOCK_SIZE_4;

  q.submit([&](handler& cgh) {
      auto variables_acc = variables.get_access<sycl_write>(cgh);
      auto old_variables_acc = old_variables.get_access<sycl_read>(cgh);
      auto fluxes_acc = fluxes.get_access<sycl_read>(cgh);
      auto step_factors_acc = step_factors.get_access<sycl_read>(cgh);

      cgh.parallel_for<class compute_time_step>(
        nd_range<1>(range<1>(work_items), range<1>(work_group_size)), [=] (nd_item<1> item) {
#include "kernel_time_step.sycl"
        });
      });
}
inline void compute_flux_contribution(float& density, Float3& momentum, float& density_energy, float& pressure, Float3& velocity, Float3& fc_momentum_x, Float3& fc_momentum_y, Float3& fc_momentum_z, Float3& fc_density_energy)
{
  fc_momentum_x.x = velocity.x*momentum.x + pressure;
  fc_momentum_x.y = velocity.x*momentum.y;
  fc_momentum_x.z = velocity.x*momentum.z;


  fc_momentum_y.x = fc_momentum_x.y;
  fc_momentum_y.y = velocity.y*momentum.y + pressure;
  fc_momentum_y.z = velocity.y*momentum.z;

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
  fc_momentum_z.z = velocity.z*momentum.z + pressure;

  float de_p = density_energy+pressure;
  fc_density_energy.x = velocity.x*de_p;
  fc_density_energy.y = velocity.y*de_p;
  fc_density_energy.z = velocity.z*de_p;
}

/*
 * Main function
 */
int main(int argc, char** argv){
  printf("WG size of kernel:initialize = %d\nWG size of kernel:compute_step_factor = %d\nWG size of kernel:compute_flux = %d\nWG size of kernel:time_step = %d\n", BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3, BLOCK_SIZE_4);

  if (argc < 2){
    std::cout << "Please specify data file name" << std::endl;
    return 0;
  }
  const char* data_file_name = argv[1];
  float h_ff_variable[NVAR];

  // set far field conditions and load them into constant memory on the gpu
  //{
  const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

  h_ff_variable[VAR_DENSITY] = float(1.4);

  float ff_pressure = float(1.0f);
  float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
  float ff_speed = float(ff_mach)*ff_speed_of_sound;

  Float3 ff_velocity;
  ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
  ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
  ff_velocity.z = 0.0f;

  h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
  h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
  h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;

  h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

  Float3 h_ff_momentum;
  h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
  h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
  h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
  Float3 h_ff_flux_contribution_momentum_x;
  Float3 h_ff_flux_contribution_momentum_y;
  Float3 h_ff_flux_contribution_momentum_z;
  Float3 h_ff_flux_contribution_density_energy;
  compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, 
      h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure,
      ff_velocity, h_ff_flux_contribution_momentum_x, 
      h_ff_flux_contribution_momentum_y, 
      h_ff_flux_contribution_momentum_z,
      h_ff_flux_contribution_density_energy);

  int nel;
  int nelr;
  std::ifstream file(data_file_name, std::ifstream::in);
  if(!file.good()){
    throw(std::string("can not find/open file! ")+data_file_name);
  }
  file >> nel;
  nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));
  std::cout<<"--cambine: nel="<<nel<<", nelr="<<nelr<<std::endl;
  float* h_areas = new float[nelr];
  int* h_elements_surrounding_elements = new int[nelr*NNB];
  float* h_normals = new float[nelr*NDIM*NNB];


  // read in data
  for(int i = 0; i < nel; i++)
  {
    file >> h_areas[i];
    for(int j = 0; j < NNB; j++)
    {
      file >> h_elements_surrounding_elements[i + j*nelr];
      if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
      h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				

      for(int k = 0; k < NDIM; k++)
      {
        file >> h_normals[i + (j + k*NNB)*nelr];
        h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
      }
    }
  }

  // fill in remaining data
  int last = nel-1;
  for(int i = nel; i < nelr; i++)
  {
    h_areas[i] = h_areas[last];
    for(int j = 0; j < NNB; j++)
    {
      // duplicate the last element
      h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
      for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
    }
  }

  double offload_start = get_time();
  { // SYCL scope
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif

    queue q(dev_sel);

    // copy far field conditions to the gpu

    const property_list props = property::buffer::use_host_ptr();
    buffer<float,1>  d_ff_variable (h_ff_variable, NVAR, props);
    buffer<Float3,1> d_ff_flux_contribution_momentum_x (&h_ff_flux_contribution_momentum_x,1,props);
    buffer<Float3,1> d_ff_flux_contribution_momentum_y (&h_ff_flux_contribution_momentum_y,1,props);
    buffer<Float3,1> d_ff_flux_contribution_momentum_z (&h_ff_flux_contribution_momentum_z,1,props);
    buffer<Float3,1> d_ff_flux_contribution_density_energy (&h_ff_flux_contribution_density_energy,1,props);


    //areas = alloc<float>(nelr);
    //upload<float>(areas, h_areas, nelr);
    buffer<float, 1> d_areas (h_areas, nelr, props);

    //elements_surrounding_elements = alloc<int>(nelr*NNB);
    //upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);
    buffer<int, 1> d_elements_surrounding_elements (h_elements_surrounding_elements, nelr*NNB, props);
    //normals = alloc<float>(nelr*NDIM*NNB);
    //upload<float>(normals, h_normals, nelr*NDIM*NNB);
    buffer<float, 1> d_normals (h_normals, nelr*NDIM*NNB, props);


    // Create arrays and set initial conditions
    //variables = alloc<float>(nelr*NVAR);				
    buffer<float,1> d_variables (nelr*NVAR);

    //old_variables = alloc<float>(nelr*NVAR);   	
    buffer<float,1> d_old_variables (nelr*NVAR);

    //fluxes = alloc<float>(nelr*NVAR);
    buffer<float,1> d_fluxes (nelr*NVAR);

    //step_factors = alloc<float>(nelr); 
    buffer<float,1> d_step_factors (nelr);


    initialize_variables(q, nelr, d_variables, d_ff_variable);
    initialize_variables(q, nelr, d_old_variables, d_ff_variable);	
    initialize_variables(q, nelr, d_fluxes, d_ff_variable);		
    initialize_buffer(q, d_step_factors, 0, nelr);

    // these need to be computed the first time in order to compute time step
    //std::cout << "Starting..." << std::endl;

    // Begin iterations
    for(int i = 0; i < iterations; i++){
      copy<float>(q, d_old_variables, d_variables, nelr*NVAR);
      // for the first iteration we compute the time step
#ifdef DEBUG
      auto h_old_variables_acc = d_old_variables.get_access<sycl_read>();
      auto h_variables_acc = d_variables.get_access<sycl_read>();
      for (int i = 0; i < 16; i++) printf("copy: i=%d %f %f\n", 
          i, h_old_variables_acc[i], h_variables_acc[i]);
#endif
      compute_step_factor(q, nelr, d_variables, d_areas, d_step_factors);
#ifdef DEBUG
      auto h_step_factors_acc = d_step_factors.get_access<sycl_read>();
      for (int i = 0; i < 16; i++) printf("step factor: i=%d %f\n", i, h_step_factors_acc[i]);
#endif
      for(int j = 0; j < RK; j++){
        compute_flux(q, nelr, d_elements_surrounding_elements, d_normals, 
            d_variables, d_ff_variable, d_fluxes, d_ff_flux_contribution_density_energy, \
            d_ff_flux_contribution_momentum_x, d_ff_flux_contribution_momentum_y, 
            d_ff_flux_contribution_momentum_z);
        time_step(q, j, nelr, d_old_variables, d_variables, d_step_factors, d_fluxes);
      }
    }
#ifdef OUTPUT
    std::cout << "Saving solution..." << std::endl;
    dump(q, d_variables, nel, nelr);
#endif

  } // SYCL scope

  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

  delete[] h_areas;
  delete[] h_elements_surrounding_elements;
  delete[] h_normals;
  std::cout << "Done..." << std::endl;

  return 0;
}
