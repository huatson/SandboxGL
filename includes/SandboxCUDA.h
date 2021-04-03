#ifndef __SANDBOX_CUDA__
#define __SANDBOX_CUDA__

#pragma once

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <vector>

// System OMP (MSVC 14 VS2017)
#include <omp.h>

//GLM
#include <GLM/glm.hpp>
#include <GLM/gtc/matrix_transform.hpp>
#include <GLM/gtc/type_ptr.hpp>
#include <GLM/gtx/quaternion.hpp>
#include <GLM/gtx/transform.hpp>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) 
{
	return cudaGetErrorName(error);
}
#endif

#ifdef CUDA_DRIVER_API
static const char *_cudaGetErrorEnum(CUresult error) 
{
	static char unknown[] = "<unknown>";
	const char *ret = NULL;
	cuGetErrorName(error, &ret);
	return ret ? ret : unknown;
}
#endif

#ifdef CUBLAS_API_H_
static const char *_cudaGetErrorEnum(cublasStatus_t error) 
{
	switch (error) 
	{
	case CUBLAS_STATUS_SUCCESS:					return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:			return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:			return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:			return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:			return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:			return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:			return "CUBLAS_STATUS_INTERNAL_ERROR";
	case CUBLAS_STATUS_NOT_SUPPORTED:			return "CUBLAS_STATUS_NOT_SUPPORTED";
	case CUBLAS_STATUS_LICENSE_ERROR:			return "CUBLAS_STATUS_LICENSE_ERROR";
	}
	return "<unknown>";
}
#endif

#ifdef _CUFFT_H_
static const char *_cudaGetErrorEnum(cufftResult error) 
{
	switch (error) 
	{
	case CUFFT_SUCCESS:							return "CUFFT_SUCCESS";
	case CUFFT_INVALID_PLAN:					return "CUFFT_INVALID_PLAN";
	case CUFFT_ALLOC_FAILED:					return "CUFFT_ALLOC_FAILED";
	case CUFFT_INVALID_TYPE:					return "CUFFT_INVALID_TYPE";
	case CUFFT_INVALID_VALUE:					return "CUFFT_INVALID_VALUE";
	case CUFFT_INTERNAL_ERROR:					return "CUFFT_INTERNAL_ERROR";
	case CUFFT_EXEC_FAILED:						return "CUFFT_EXEC_FAILED";
	case CUFFT_SETUP_FAILED:					return "CUFFT_SETUP_FAILED";
	case CUFFT_INVALID_SIZE:					return "CUFFT_INVALID_SIZE";
	case CUFFT_UNALIGNED_DATA:					return "CUFFT_UNALIGNED_DATA";
	case CUFFT_INCOMPLETE_PARAMETER_LIST:		return "CUFFT_INCOMPLETE_PARAMETER_LIST";
	case CUFFT_INVALID_DEVICE:					return "CUFFT_INVALID_DEVICE";
	case CUFFT_PARSE_ERROR:						return "CUFFT_PARSE_ERROR";
	case CUFFT_NO_WORKSPACE:					return "CUFFT_NO_WORKSPACE";
	case CUFFT_NOT_IMPLEMENTED:					return "CUFFT_NOT_IMPLEMENTED";
	case CUFFT_LICENSE_ERROR:					return "CUFFT_LICENSE_ERROR";
	case CUFFT_NOT_SUPPORTED:					return "CUFFT_NOT_SUPPORTED";
	}
	return "<unknown>";
}
#endif

#ifdef CUSPARSEAPI
static const char *_cudaGetErrorEnum(cusparseStatus_t error) 
{
	switch (error) 
	{
	case CUSPARSE_STATUS_SUCCESS:				return "CUSPARSE_STATUS_SUCCESS";
	case CUSPARSE_STATUS_NOT_INITIALIZED:		return "CUSPARSE_STATUS_NOT_INITIALIZED";
	case CUSPARSE_STATUS_ALLOC_FAILED:			return "CUSPARSE_STATUS_ALLOC_FAILED";
	case CUSPARSE_STATUS_INVALID_VALUE:			return "CUSPARSE_STATUS_INVALID_VALUE";
	case CUSPARSE_STATUS_ARCH_MISMATCH:			return "CUSPARSE_STATUS_ARCH_MISMATCH";
	case CUSPARSE_STATUS_MAPPING_ERROR:			return "CUSPARSE_STATUS_MAPPING_ERROR";
	case CUSPARSE_STATUS_EXECUTION_FAILED:		return "CUSPARSE_STATUS_EXECUTION_FAILED";
	case CUSPARSE_STATUS_INTERNAL_ERROR:		return "CUSPARSE_STATUS_INTERNAL_ERROR";
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	}
	return "<unknown>";
}
#endif

#ifdef CUSOLVER_COMMON_H_
static const char *_cudaGetErrorEnum(cusolverStatus_t error) 
{
	switch (error) 
	{
	case CUSOLVER_STATUS_SUCCESS:				return "CUSOLVER_STATUS_SUCCESS";
	case CUSOLVER_STATUS_NOT_INITIALIZED:		return "CUSOLVER_STATUS_NOT_INITIALIZED";
	case CUSOLVER_STATUS_ALLOC_FAILED:			return "CUSOLVER_STATUS_ALLOC_FAILED";
	case CUSOLVER_STATUS_INVALID_VALUE:			return "CUSOLVER_STATUS_INVALID_VALUE";
	case CUSOLVER_STATUS_ARCH_MISMATCH:			return "CUSOLVER_STATUS_ARCH_MISMATCH";
	case CUSOLVER_STATUS_MAPPING_ERROR:			return "CUSOLVER_STATUS_MAPPING_ERROR";
	case CUSOLVER_STATUS_EXECUTION_FAILED:		return "CUSOLVER_STATUS_EXECUTION_FAILED";
	case CUSOLVER_STATUS_INTERNAL_ERROR:		return "CUSOLVER_STATUS_INTERNAL_ERROR";
	case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:		return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	case CUSOLVER_STATUS_NOT_SUPPORTED:			return "CUSOLVER_STATUS_NOT_SUPPORTED ";
	case CUSOLVER_STATUS_ZERO_PIVOT:			return "CUSOLVER_STATUS_ZERO_PIVOT";
	case CUSOLVER_STATUS_INVALID_LICENSE:		return "CUSOLVER_STATUS_INVALID_LICENSE";
	}
	return "<unknown>";
}
#endif

#ifdef CURAND_H_
static const char *_cudaGetErrorEnum(curandStatus_t error) 
{
	switch (error) 
	{
	case CURAND_STATUS_SUCCESS:					return "CURAND_STATUS_SUCCESS";
	case CURAND_STATUS_VERSION_MISMATCH:		return "CURAND_STATUS_VERSION_MISMATCH";
	case CURAND_STATUS_NOT_INITIALIZED:			return "CURAND_STATUS_NOT_INITIALIZED";
	case CURAND_STATUS_ALLOCATION_FAILED:		return "CURAND_STATUS_ALLOCATION_FAILED";
	case CURAND_STATUS_TYPE_ERROR:				return "CURAND_STATUS_TYPE_ERROR";
	case CURAND_STATUS_OUT_OF_RANGE:			return "CURAND_STATUS_OUT_OF_RANGE";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:	return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
	case CURAND_STATUS_LAUNCH_FAILURE:			return "CURAND_STATUS_LAUNCH_FAILURE";
	case CURAND_STATUS_PREEXISTING_FAILURE:		return "CURAND_STATUS_PREEXISTING_FAILURE";
	case CURAND_STATUS_INITIALIZATION_FAILED:	return "CURAND_STATUS_INITIALIZATION_FAILED";
	case CURAND_STATUS_ARCH_MISMATCH:			return "CURAND_STATUS_ARCH_MISMATCH";
	case CURAND_STATUS_INTERNAL_ERROR:			return "CURAND_STATUS_INTERNAL_ERROR";
	}
	return "<unknown>";
}
#endif

#ifdef NVJPEGAPI
// nvJPEG API errors
static const char *_cudaGetErrorEnum(nvjpegStatus_t error) {
	switch (error) {
	case NVJPEG_STATUS_SUCCESS:
		return "NVJPEG_STATUS_SUCCESS";

	case NVJPEG_STATUS_NOT_INITIALIZED:
		return "NVJPEG_STATUS_NOT_INITIALIZED";

	case NVJPEG_STATUS_INVALID_PARAMETER:
		return "NVJPEG_STATUS_INVALID_PARAMETER";

	case NVJPEG_STATUS_BAD_JPEG:
		return "NVJPEG_STATUS_BAD_JPEG";

	case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
		return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

	case NVJPEG_STATUS_ALLOCATOR_FAILURE:
		return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

	case NVJPEG_STATUS_EXECUTION_FAILED:
		return "NVJPEG_STATUS_EXECUTION_FAILED";

	case NVJPEG_STATUS_ARCH_MISMATCH:
		return "NVJPEG_STATUS_ARCH_MISMATCH";

	case NVJPEG_STATUS_INTERNAL_ERROR:
		return "NVJPEG_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}
#endif

#ifdef NV_NPPIDEFS_H
static const char *_cudaGetErrorEnum(NppStatus error) 
{
	switch (error) 
	{
	case NPP_NOT_SUPPORTED_MODE_ERROR:					return "NPP_NOT_SUPPORTED_MODE_ERROR";
	case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:			return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
	case NPP_RESIZE_NO_OPERATION_ERROR:					return "NPP_RESIZE_NO_OPERATION_ERROR";
	case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:			return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000
	case NPP_BAD_ARG_ERROR:								return "NPP_BAD_ARGUMENT_ERROR";
	case NPP_COEFF_ERROR:								return "NPP_COEFFICIENT_ERROR";
	case NPP_RECT_ERROR:								return "NPP_RECTANGLE_ERROR";
	case NPP_QUAD_ERROR:								return "NPP_QUADRANGLE_ERROR";
	case NPP_MEM_ALLOC_ERR:								return "NPP_MEMORY_ALLOCATION_ERROR";
	case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:				return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
	case NPP_INVALID_INPUT:								return "NPP_INVALID_INPUT";
	case NPP_POINTER_ERROR:								return "NPP_POINTER_ERROR";
	case NPP_WARNING:									return "NPP_WARNING";
	case NPP_ODD_ROI_WARNING:							return "NPP_ODD_ROI_WARNING";
#else
	case NPP_BAD_ARGUMENT_ERROR:						return "NPP_BAD_ARGUMENT_ERROR";
	case NPP_COEFFICIENT_ERROR:							return "NPP_COEFFICIENT_ERROR";
	case NPP_RECTANGLE_ERROR:							return "NPP_RECTANGLE_ERROR";
	case NPP_QUADRANGLE_ERROR:							return "NPP_QUADRANGLE_ERROR";
	case NPP_MEMORY_ALLOCATION_ERR:						return "NPP_MEMORY_ALLOCATION_ERROR";
	case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:			return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
	case NPP_INVALID_HOST_POINTER_ERROR:				return "NPP_INVALID_HOST_POINTER_ERROR";
	case NPP_INVALID_DEVICE_POINTER_ERROR:				return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif
	case NPP_LUT_NUMBER_OF_LEVELS_ERROR:				return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
	case NPP_TEXTURE_BIND_ERROR:						return "NPP_TEXTURE_BIND_ERROR";
	case NPP_WRONG_INTERSECTION_ROI_ERROR:				return "NPP_WRONG_INTERSECTION_ROI_ERROR";
	case NPP_NOT_EVEN_STEP_ERROR:						return "NPP_NOT_EVEN_STEP_ERROR";
	case NPP_INTERPOLATION_ERROR:						return "NPP_INTERPOLATION_ERROR";
	case NPP_RESIZE_FACTOR_ERROR:						return "NPP_RESIZE_FACTOR_ERROR";
	case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:			return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000
	case NPP_MEMFREE_ERR:								return "NPP_MEMFREE_ERR";
	case NPP_MEMSET_ERR:								return "NPP_MEMSET_ERR";
	case NPP_MEMCPY_ERR:								return "NPP_MEMCPY_ERROR";
	case NPP_MIRROR_FLIP_ERR:							return "NPP_MIRROR_FLIP_ERR";
#else
	case NPP_MEMFREE_ERROR:								return "NPP_MEMFREE_ERROR";
	case NPP_MEMSET_ERROR:								return "NPP_MEMSET_ERROR";
	case NPP_MEMCPY_ERROR:								return "NPP_MEMCPY_ERROR";
	case NPP_MIRROR_FLIP_ERROR:							return "NPP_MIRROR_FLIP_ERROR";
#endif
	case NPP_ALIGNMENT_ERROR:							return "NPP_ALIGNMENT_ERROR";
	case NPP_STEP_ERROR:								return "NPP_STEP_ERROR";
	case NPP_SIZE_ERROR:								return "NPP_SIZE_ERROR";
	case NPP_NULL_POINTER_ERROR:						return "NPP_NULL_POINTER_ERROR";
	case NPP_CUDA_KERNEL_EXECUTION_ERROR:				return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
	case NPP_NOT_IMPLEMENTED_ERROR:						return "NPP_NOT_IMPLEMENTED_ERROR";
	case NPP_ERROR:										return "NPP_ERROR";
	case NPP_SUCCESS:									return "NPP_SUCCESS";
	case NPP_WRONG_INTERSECTION_QUAD_WARNING:			return "NPP_WRONG_INTERSECTION_QUAD_WARNING";
	case NPP_MISALIGNED_DST_ROI_WARNING:				return "NPP_MISALIGNED_DST_ROI_WARNING";
	case NPP_AFFINE_QUAD_INCORRECT_WARNING:				return "NPP_AFFINE_QUAD_INCORRECT_WARNING";
	case NPP_DOUBLE_SIZE_WARNING:						return "NPP_DOUBLE_SIZE_WARNING";
	case NPP_WRONG_INTERSECTION_ROI_WARNING:			return "NPP_WRONG_INTERSECTION_ROI_WARNING";
#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
	case NPP_LUT_PALETTE_BITSIZE_ERROR:					return "NPP_LUT_PALETTE_BITSIZE_ERROR";
	case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:				return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
	case NPP_QUALITY_INDEX_ERROR:						return "NPP_QUALITY_INDEX_ERROR";
	case NPP_CHANNEL_ORDER_ERROR:						return "NPP_CHANNEL_ORDER_ERROR";
	case NPP_ZERO_MASK_VALUE_ERROR:						return "NPP_ZERO_MASK_VALUE_ERROR";
	case NPP_NUMBER_OF_CHANNELS_ERROR:					return "NPP_NUMBER_OF_CHANNELS_ERROR";
	case NPP_COI_ERROR:									return "NPP_COI_ERROR";
	case NPP_DIVISOR_ERROR:								return "NPP_DIVISOR_ERROR";
	case NPP_CHANNEL_ERROR:								return "NPP_CHANNEL_ERROR";
	case NPP_STRIDE_ERROR:								return "NPP_STRIDE_ERROR";
	case NPP_ANCHOR_ERROR:								return "NPP_ANCHOR_ERROR";
	case NPP_MASK_SIZE_ERROR:							return "NPP_MASK_SIZE_ERROR";
	case NPP_MOMENT_00_ZERO_ERROR:						return "NPP_MOMENT_00_ZERO_ERROR";
	case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:			return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
	case NPP_THRESHOLD_ERROR:							return "NPP_THRESHOLD_ERROR";
	case NPP_CONTEXT_MATCH_ERROR:						return "NPP_CONTEXT_MATCH_ERROR";
	case NPP_FFT_FLAG_ERROR:							return "NPP_FFT_FLAG_ERROR";
	case NPP_FFT_ORDER_ERROR:							return "NPP_FFT_ORDER_ERROR";
	case NPP_SCALE_RANGE_ERROR:							return "NPP_SCALE_RANGE_ERROR";
	case NPP_DATA_TYPE_ERROR:							return "NPP_DATA_TYPE_ERROR";
	case NPP_OUT_OFF_RANGE_ERROR:						return "NPP_OUT_OFF_RANGE_ERROR";
	case NPP_DIVIDE_BY_ZERO_ERROR:						return "NPP_DIVIDE_BY_ZERO_ERROR";
	case NPP_RANGE_ERROR:								return "NPP_RANGE_ERROR";
	case NPP_NO_MEMORY_ERROR:							return "NPP_NO_MEMORY_ERROR";
	case NPP_ERROR_RESERVED:							return "NPP_ERROR_RESERVED";
	case NPP_NO_OPERATION_WARNING:						return "NPP_NO_OPERATION_WARNING";
	case NPP_DIVIDE_BY_ZERO_WARNING:					return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif
#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
	case NPP_OVERFLOW_ERROR:							return "NPP_OVERFLOW_ERROR";
	case NPP_CORRUPTED_DATA_ERROR:						return "NPP_CORRUPTED_DATA_ERROR";
#endif
	}
	return "<unknown>";
}
#endif


// Reset device
#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif	//DEVICE_RESET
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif	DEVICE_RESET
#endif	//__DRIVER_TYPES_H__

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) 
{
	if (result) 
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), 
			_cudaGetErrorEnum(result), func);
		DEVICE_RESET
		exit(EXIT_FAILURE);
	}
}

#ifdef __DRIVER_TYPES_H__
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, static_cast<int>(err), 
			cudaGetErrorString(err));
		DEVICE_RESET
		exit(EXIT_FAILURE);
	}
}
inline void __printLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, static_cast<int>(err), 
			cudaGetErrorString(err));
	}
}
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
// This will only print the proper error string when calling cudaGetLastError
// but not exit program in case error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)
#endif	//__DRIVER_TYPES_H__

#define MAX_BLOCK_SIZE 1024


#endif	//__SANDBOX_CUDA__


