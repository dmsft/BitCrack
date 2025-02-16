#ifndef _EC_CUH
#define _EC_CUH

#include <cuda_runtime.h>

namespace ec
{
	extern __constant__ unsigned int *_xPtr[1];
	extern __constant__ unsigned int *_yPtr[1];

	// __device__ unsigned int *getXPtr();
	// __device__ unsigned int *getYPtr();
}

#endif