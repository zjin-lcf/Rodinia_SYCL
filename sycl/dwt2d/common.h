#ifndef _COMMON_H
#define _COMMON_H

#include <CL/sycl.hpp>

using namespace cl::sycl;
constexpr access::mode sycl_read       = access::mode::read;
constexpr access::mode sycl_write      = access::mode::write;
constexpr access::mode sycl_read_write = access::mode::read_write;
constexpr access::mode sycl_discard_read_write = access::mode::discard_read_write;
constexpr access::mode sycl_discard_write = access::mode::discard_write;
constexpr access::target sycl_global_buffer = access::target::global_buffer;
constexpr access::target sycl_local_buffer = access::target::local;

//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

#define DIVANDRND(a, b) ((((a) % (b)) != 0) ? ((a) / (b) + 1) : ((a) / (b)))

#endif
