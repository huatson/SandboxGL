#ifndef __SANDBOX_CUDA_INFO__
#define __SANDBOX_CUDA_INFO__

#pragma once

#include "SandboxCUDA.h"

extern "C" bool GetCUDAInfo()
{
	printf("CUDA Info:\n");
	// number of CUDA GPUs
	int num_gpus = 0;   
	cudaGetDeviceCount(&num_gpus);
	int num_cpus = omp_get_num_procs();
	if (num_gpus < 1 || num_cpus < 0)
	{
		printf("no CPU/CUDA capable devices were detected\n");
		return false;
	}
	// display CPU and GPU configuration
	printf("number of host CPUs:\t%d\n", num_cpus);
	printf("number of CUDA devices:\t%d\n", num_gpus);
	for (int i = 0; i < num_gpus; i++)
	{
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop, i);
		printf("   %d: %s\n", i, dprop.name);
	}
	printf("---------------------------\n");
	return true;
}
#endif