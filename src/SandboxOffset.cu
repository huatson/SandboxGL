
// CUDA runtime
#include "SandboxCUDA.h"

// shared memory : thread size * size of float --> 4, range: 3.4E +/- 38 (7 digits)
#define SHARED_MEMORY_SIZE 256


template <unsigned int blockSize>
__global__ void sum_reduction_kernel(float *values, float *result_values, unsigned int n)
{
	extern __shared__ float partial_sum[];
	unsigned int scaled_thread_index = blockIdx.x*(blockSize * 2) + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	partial_sum[threadIdx.x] = 0;
	while (scaled_thread_index < n)
	{
		partial_sum[threadIdx.x] += values[scaled_thread_index] + values[scaled_thread_index + blockSize];
		scaled_thread_index += gridSize;
	}
	__syncthreads();

	if (blockSize >= 1024)
	{
		if (threadIdx.x < 512)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 512];
		}
		__syncthreads();
	}

	if (blockSize >= 512)
	{
		if (threadIdx.x < 256)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 256];
		}
		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (threadIdx.x < 128)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (threadIdx.x < 64)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 64];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32)
	{
		if (blockSize >= 64) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 32];
		if (blockSize >= 32) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 16];
		if (blockSize >= 16) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 8];
		if (blockSize >= 8) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 4];
		if (blockSize >= 4) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 2];
		if (blockSize >= 2) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + 1];
	}

	if (threadIdx.x == 0)
	{
		result_values[blockIdx.x] = partial_sum[0];
	}
}

/**
 * final sum reduction
 */
__global__ void sum_kernel(float *values, float *result_values)
{
	//__shared__ float partial_sum[SHARED_MEMORY_SIZE];
	extern __shared__ float partial_sum[];
	unsigned int scaled_thread_index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	partial_sum[threadIdx.x] = values[scaled_thread_index] + values[scaled_thread_index + blockDim.x];
	__syncthreads();
	for (unsigned int stride_idx = blockDim.x / 2; stride_idx > 0; stride_idx >>= 1)
	{
		if (threadIdx.x < stride_idx)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride_idx];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		result_values[blockIdx.x] = partial_sum[0];
	}
}

/**
 * sum will work only with power of 2 maps: 128,256,512, etc
 */
extern "C" float computeOffsetCUDA(std::vector<float> fdata, int map_size)
{
	int size = fdata.size(); //262144
	size_t byte_size = size * sizeof(float); // 262144*4 = 1,048,576 bytes

	float *device_vector;
	float *device_vector_result;
	float *device_vector_result_final;
	checkCudaErrors(cudaMallocManaged(&device_vector, byte_size));
	checkCudaErrors(cudaMallocManaged(&device_vector_result, byte_size));
	checkCudaErrors(cudaMallocManaged(&device_vector_result_final, byte_size));

	checkCudaErrors(cudaMemcpy(device_vector, fdata.data(), byte_size, cudaMemcpyKind::cudaMemcpyHostToDevice));

	const int THREADS_PER_BLOCK = 256;
	const int BLOCK_SIZE = (int)ceil(size / THREADS_PER_BLOCK / 2);
	sum_kernel<<<BLOCK_SIZE, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(device_vector, device_vector_result);
	sum_kernel<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >>>(device_vector_result, device_vector_result_final);
	cudaDeviceSynchronize();

	float offset_result_sum = device_vector_result_final[0];
	printf("GPU offset SUM: %.4f\n", offset_result_sum);
	float offset_result_avg = offset_result_sum / map_size;
	printf("GPU offset AVG: %.4f\n", offset_result_avg);

	checkCudaErrors(cudaFree(device_vector));
	checkCudaErrors(cudaFree(device_vector_result));
	checkCudaErrors(cudaFree(device_vector_result_final));
	return offset_result_avg;
}