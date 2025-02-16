#ifndef _CUDA_UTIL_H
#define _CUDA_UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace cuda
{
	typedef struct
	{
		int id;
		int major;
		int minor;
		int mpCount;
		int cores;
		uint64_t mem;
		std::string name;

		// size_t sharedMemPerBlock;          /**< Shared memory available per block in bytes */
		// int regsPerBlock;               /**< 32-bit registers available per block */
		// int warpSize;                   /**< Warp size in threads */
		// int maxThreadsPerBlock;         /**< Maximum number of threads per block */
		// size_t totalConstMem;              /**< Constant memory available on device in bytes */
		// int concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
		// int l2CacheSize;

		// size_t sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
		// int regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
		// int maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
		// size_t reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */

		cudaDeviceProp props;

	} CudaDeviceInfo;


	class CudaException
	{
	public:
		cudaError_t error;
		std::string msg;

		CudaException(cudaError_t err)
		{
			this->error = err;
			this->msg = std::string(cudaGetErrorString(err));
		}
	};

	CudaDeviceInfo getDeviceInfo(int device);

	std::vector<CudaDeviceInfo> getDevices();

	int getDeviceCount();
}
#endif