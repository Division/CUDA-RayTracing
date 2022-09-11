#pragma once

#include "helper_cuda.h"

namespace CUDA
{
	template <typename T>
	void check(T result, char const* const func, const char* const file,
		int const line) {
		if (result) {
			fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\". Error text: %s \n", file, line,
				static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func, cudaGetErrorString(result));
#ifdef _DEBUG
			__debugbreak();
#endif
			exit(EXIT_FAILURE);
		}
	}

}

#define CUDA_CHECK(val) CUDA::check((val), #val, __FILE__, __LINE__)

#ifdef __NVCC__
	#define CUDA_HOST __host__
	#define CUDA_DEVICE __device__
    #define CUDA_HOST_DEVICE __host__ __device__
	#define CUDA_ONLY(A) A
	#define HOST_ONLY(A)
	#define CUDA_COMPILER
#else
	#define CUDA_HOST
	#define CUDA_DEVICE
	#define CUDA_HOST_DEVICE __host__ __device__
	#define CUDA_ONLY(A)
	#define HOST_ONLY(A) A
#endif

namespace CUDA
{
	template<typename T>
	class Handle
	{
		typedef cudaError (*DeleterFunc)(T);
		// Can't use std::optional here since cuda compiler doesn't support it
		T object = {};
		bool has_value = false;
		DeleterFunc deleter = nullptr;

	public:
		Handle(std::nullptr_t) : has_value(false) {};
		Handle(T* object, DeleterFunc deleter) : object(*object), has_value(true), deleter(deleter) {}
		Handle() = default;

		~Handle()
		{
			if (has_value && deleter)
				CUDA_CHECK(deleter(object));
		}

		Handle(Handle&& other)
		{
			*this = std::move(other);
		}

		Handle& operator=(Handle&& other)
		{
			if (this != &other)
			{
				object = other.object;
				has_value = other.has_value;
				deleter = other.deleter;
				other.object = {};
				other.deleter = nullptr;
				other.has_value = false;
			}
			
			return *this;
		}

		Handle& operator=(const Handle&) = delete;
		Handle(const Handle&) = delete;

		operator bool() const { return has_value; }
		T Get() const { if (!has_value) throw std::runtime_error("referenced empty handle"); return object; }
		T operator*() const { return Get(); }
	};

	inline Handle<cudaTextureObject_t> CreateTextureObject(const struct cudaResourceDesc* pResDesc, const struct cudaTextureDesc* pTexDesc, const struct cudaResourceViewDesc* pResViewDesc)
	{
		cudaTextureObject_t object;
		CUDA_CHECK(cudaCreateTextureObject(&object, pResDesc, pTexDesc, pResViewDesc));
		return Handle<cudaTextureObject_t>(&object, cudaDestroyTextureObject);
	}

	inline Handle<cudaArray_t> Create3DArray(const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags = 0)
	{
		cudaArray_t object;
		CUDA_CHECK(cudaMalloc3DArray(&object, desc, extent, flags));
		return Handle<cudaArray_t>(&object, cudaFreeArray);
	}

	inline Handle<cudaMipmappedArray_t> CreateMipmappedArray(const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags = 0)
	{
		cudaMipmappedArray_t object;
		CUDA_CHECK(cudaMallocMipmappedArray(&object, desc, extent, numLevels, flags));
		return Handle<cudaMipmappedArray_t>(&object, cudaFreeMipmappedArray);
	}

	class DeviceMemory
	{
		void* memory = nullptr;
		size_t size;
	
	public:
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

		size_t GetSize() const { return size; }
		void* GetMemory() const { return memory; }
	};

}
