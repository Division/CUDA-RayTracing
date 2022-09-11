#pragma once

#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "utils/CUDAHelper.h"
#include "Math.h"
#if defined(CUDA_COMPILER)
	#include "Random.cuh"
#endif

namespace RayTracing
{
	using namespace Math;

	struct Camera
	{
		Camera(vec3 origin, float aspect)
			: origin(origin)
			, aspect(aspect)
		{
			viewport_size.y = 2.0f;
			viewport_size.x = viewport_size.y * aspect;
			horizontal = vec3(viewport_size.x, 0, 0);
			vertical = vec3(0, viewport_size.y, 0);
			lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - vec3(0, 0, focal_length);
		}

		CUDA_HOST_DEVICE Ray GetRay(vec2 uv) const { return Ray(origin, lower_left_corner + uv.x * horizontal + uv.y * vertical - origin); }

		vec3 origin = vec3(0);
		vec2 viewport_size = vec2(0);
		float aspect = 1.0f;
		float focal_length = 1.0f;

	protected:
		vec3 horizontal;
		vec3 vertical;
		vec3 lower_left_corner;
	};


	struct alignas(16) GeometrySphere
	{
		vec3 position;
		float radius;
	};

	struct GPUScene
	{
		char* gpu_memory = nullptr;
		int sphere_count = 0;
		curandState* rng_state = nullptr;
		cudaTextureObject_t environment_cubemap_tex = 0;

		CUDA_DEVICE int GetSphereCount() { return sphere_count; };
		CUDA_DEVICE GeometrySphere* GetSphereData(int index) { return (GeometrySphere*)(gpu_memory + index * sizeof(GeometrySphere)); };
		CUDA_ONLY( CUDA_DEVICE RNG GetRNGForPixel(int screen_width, int x, int y) { return RNG(rng_state + y * screen_width + x); } )
	};

}
