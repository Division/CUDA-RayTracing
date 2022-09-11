#pragma once

#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "Math.h"

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

		__device__ Ray GetRay(vec2 uv) const { return Ray(origin, lower_left_corner + uv.x * horizontal + uv.y * vertical - origin); }

		vec3 origin = vec3(0);
		vec2 viewport_size = vec2(0);
		float aspect = 1.0f;
		float focal_length = 1.0f;

	private:
		vec3 horizontal;
		vec3 vertical;
		vec3 lower_left_corner;
	};


	struct RNG
	{
		__device__ RNG(curandState* state)
			: state(state)
		{}

		__device__ float GetFloat01() { return curand_uniform(state); }
		__device__ float GetFloatNormal() { return curand_normal(state); }
		__device__ vec2 GetVec2Normal() { auto r1 = curand_normal2(state); return vec2(r1.x, r1.y); }
		__device__ vec4 GetVec4Normal() { return vec4(GetVec2Normal(), GetVec2Normal()); }
#ifdef __NVCC__
		__device__ vec3 GetRandomPointOnSphere()
		{
			int counter = 20;
			while (counter--)
			{
				const vec3 p(GetVec4Normal());
				auto sq_len = glm::dot(p, p);
				if (sq_len < 1.0f)
				{
					if (sq_len < 0.0001f) return vec3(1, 0, 0);
					return p * __frsqrt_rn(sq_len);
				}
			}

			return vec3(1, 0, 0);
		}
#endif

		curandState* state;
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

		__host__ __device__
		__device__ int GetSphereCount() { return sphere_count; };
		__device__ GeometrySphere* GetSphereData(int index) { return (GeometrySphere*)(gpu_memory + index * sizeof(GeometrySphere)); };
		__device__ RNG GetRNGForPixel(int screen_width, int x, int y) { return RNG(rng_state + y * screen_width + x); }
	};

}
