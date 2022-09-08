#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdexcept>
#include "helper_cuda.h"

#define CUDA_CHECK(val) checkCudaErrors(val) 

namespace CUDAHelper
{
	struct DeviceMemory
	{
		void* memory = nullptr;
		size_t size;
		DeviceMemory(size_t size)
			: size(size)
		{
			const auto result = cudaMalloc(&memory, size);
			if (result != cudaSuccess)
				throw std::runtime_error("allocation failed");
		}

		~DeviceMemory()
		{
			if (memory)
			{
				cudaFree(memory);
			}
		}

		DeviceMemory(const DeviceMemory&) = delete;
		DeviceMemory(DeviceMemory&& other)
		{
			*this = std::move(other);
		}

		DeviceMemory& operator=(const DeviceMemory& other) = delete;
		DeviceMemory& operator=(DeviceMemory&& other)
		{
			if (this != &other)
			{
				memory = other.memory;
				size = other.size;
				other.memory = nullptr;
				other.size = 0;
			}
		}
	};

}
