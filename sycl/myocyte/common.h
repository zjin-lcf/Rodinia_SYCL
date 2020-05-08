// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE
//========================================================================================================================================================================================================200
#ifndef __COMMON
#define __COMMON

#define fp float

#define NUMBER_THREADS 2

#define EQUATIONS 91
#define PARAMETERS 18

#include <CL/sycl.hpp>

using namespace cl::sycl;
constexpr access::mode sycl_read       = access::mode::read;
constexpr access::mode sycl_write      = access::mode::write;
constexpr access::mode sycl_read_write = access::mode::read_write;
constexpr access::mode sycl_discard_read_write = access::mode::discard_read_write;
constexpr access::mode sycl_discard_write = access::mode::discard_write;
constexpr access::target sycl_global_buffer = access::target::global_buffer;

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
#endif

// #ifdef __cplusplus
// }
// #endif
