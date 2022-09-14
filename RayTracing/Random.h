#pragma once

#include "Math.h"
#include <curand_kernel.h>
#include <math_constants.h>

namespace RayTracing
{
	using namespace Math;

	struct RNG
	{

		__device__ RNG(curandState* state)
			: state(state)
		{}


		__device__ float GetFloat01() { return curand_uniform(state); }
		__device__ float GetFloatNormal() { return curand_normal(state); }
		__device__ vec2 GetVec2Normal() { auto r1 = curand_normal2(state); return vec2(r1.x, r1.y); }
		__device__ vec4 GetVec4Normal() { return vec4(GetVec2Normal(), GetVec2Normal()); }
		__device__ vec3 GetRandomPointOnSphere()
		{
			//return vec3(GetFloatNormal(), GetFloatNormal(), GetFloatNormal());
			float z = GetFloat01() * 2.0f - 1.0f;
			float a = GetFloat01() * CUDART_PI_F * 2.0f;
			float r = sqrt(1.0f - z * z);
			float x = r * cos(a);
			float y = r * sin(a);
			return vec3(x, y, z);
			
			/*int counter = 19;
			while (counter--)
			{
				const vec3 p(GetFloatNormal(), GetFloatNormal(), GetFloatNormal());
				//const vec3 p(GetVec4Normal());
				auto sq_len = glm::dot(p, p);
				if (sq_len > 0.0001f)
				{
					return p / sqrt(sq_len);
				}
			}

			return vec3(1, 0, 0);*/
		}

		curandState* state;
	};
}