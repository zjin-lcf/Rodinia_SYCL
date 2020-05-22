
#ifndef _DWT_H
#define _DWT_H

#include "common.h"

template<typename T> 
int nStage2dDWT(queue &q, buffer<T,1> &in, buffer<T,1> &out, buffer<T,1> & backup, int pixWidth, int pixHeight, int stages, bool forward);

template<typename T>
int writeNStage2DDWT(queue &q, buffer<T,1> &component_cuda, int width, int height, 
                     int stages, const char * filename, const char * suffix);
template<typename T>
int writeLinear(queue &q, buffer<T,1> &component_cuda, int width, int height, 
                     const char * filename, const char * suffix);

#endif
