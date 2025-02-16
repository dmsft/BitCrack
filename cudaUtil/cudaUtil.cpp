#include "cudaUtil.h"


cuda::CudaDeviceInfo cuda::getDeviceInfo(int device)
{
	cuda::CudaDeviceInfo devInfo;
	cudaDeviceProp properties;
	cudaError_t err = cudaSuccess;

	err = cudaSetDevice(device);
	if (err)
		throw cuda::CudaException(err);

	err = cudaGetDeviceProperties(&properties, device);
	if (err)
		throw cuda::CudaException(err);

	memcpy((void *) &devInfo.props, (void *) &properties, sizeof(properties));

	devInfo.id = device;
	devInfo.major = properties.major;
	devInfo.minor = properties.minor;
	devInfo.mpCount = properties.multiProcessorCount;
	devInfo.mem = properties.totalGlobalMem;
	devInfo.name = std::string(properties.name);

	// identify amount of cores per SM based on CUDA capability major/minor
	int cores = 0;
	switch(devInfo.major)
	{
		case 2:
			cores = 48;
			if (devInfo.minor == 0)
				cores = 32;
			break;
		
		case 3:
			cores = 192;
			break;
		
		case 5:
		case 8:
			cores = 128;
			break;
		
		case 6:
			cores = 64;
			if (devInfo.minor == 1 || devInfo.minor == 2)
				cores = 128;
			break;
		
		case 7:
			cores = 64;
			break;
		
		default:
			cores = 8;
			break;
	}
	devInfo.cores = cores;

	return devInfo;
}


std::vector<cuda::CudaDeviceInfo> cuda::getDevices()
{
	int count = getDeviceCount();
	std::vector<CudaDeviceInfo> devList;

	for (int device = 0; device < count; device++) {
		devList.push_back(getDeviceInfo(device));
	}

	return devList;
}

int cuda::getDeviceCount()
{
	int count = 0;

	cudaError_t err = cudaGetDeviceCount(&count);

    if(err) {
        throw cuda::CudaException(err);
    }

	return count;
}