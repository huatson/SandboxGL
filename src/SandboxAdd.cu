

// CUDA runtime
#include "SandboxCUDA.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif


/**
 * init values
 */
__global__ void init_kernel(int N, float* X, float* Y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
	{
		X[i] = 1.0f;
		Y[i] = 2.0f;
	}
}

/**
 * sum vectors
 */
__global__ void add_kernel(int N, float* X, float* Y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
	{
		Y[i] = X[i] + Y[i];
	}
}

/**
 * check max errors, max always should be 0
 */
__global__ void maxerror_kernel(float MaxError, int N, float* Y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
	{
		MaxError = fmax(MaxError, fabs(Y[i] - 3.0f));
	}
}

extern "C" bool AddVector(int N)
{
	/**
	 * Memory Allocation in CUDA:
	 * To allocate data in unified memory, call 'cudaMallocManaged()'.
	 * To free the data, just pass the pointer to 'cudaFree()'.
	 * Replace the calls to 'new' in the code with 'cudaMallocManaged()',
	 * Replace calls to 'delete[]' with 'cudaFree(VAR)'.
	 */
	float* x; // ptr to allocate memory X
	float* y; // ptr to allocate memory Y
	const size_t MemorySizeBuffer = (N * sizeof(float));
	checkCudaErrors(cudaMallocManaged(&x, MemorySizeBuffer));
	checkCudaErrors(cudaMallocManaged(&y, MemorySizeBuffer));


	/**
	 * Parallelism:
	 * <<<1,1>>> : execution configuration, and it tells the CUDA runtime how many
	 * parallel threads to use for the launch on the GPU.
	 * There are two in this case, by changing the second one: the number of threads in a thread block.
	 * CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size,
	 * so 256 threads is a reasonable size to choose.
	 * If I run the code with only this change:
	 * add_simple_kernel <<<1, 256>>>(N, x, y);
	 * it will do the computation once per thread, rather than spreading the computation across the parallel threads.
	 * calculate number-of-blocks A and block-size B for <<<A,B>>>
	 */
	int threadsperblock = 256;
	int numblocks = (N + threadsperblock - 1) / threadsperblock;
	init_kernel<<<numblocks, threadsperblock >>>(N, x, y);
	add_kernel<<<numblocks, threadsperblock >>>(N, x, y);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	float MaxError = 0.0f;
	maxerror_kernel<<<numblocks, threadsperblock>>>(MaxError, N, y);

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	std::cout << "MaxError: " << MaxError << "\n";

	//delete[] x;
	//delete[] y;
	checkCudaErrors(cudaFree(x));
	checkCudaErrors(cudaFree(y));

	return true;
}




