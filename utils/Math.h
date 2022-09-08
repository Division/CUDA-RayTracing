#pragma once

#include "glm/glm.hpp"
#include "glm/gtx/compatibility.hpp"
#include "glm/gtx/intersect.hpp"
namespace Math
{
	using vec2 = glm::vec2;
	using vec3 = glm::vec3;
	using vec4 = glm::vec4;
	using ivec2 = glm::ivec2;
	using ivec3 = glm::ivec3;
	using ivec4 = glm::ivec4;
	using uvec2 = glm::uvec2;
	using uvec3 = glm::uvec3;
	using uvec4 = glm::uvec4;
	using mat3 = glm::mat3;
	using mat4 = glm::mat4;

	struct Ray
	{
		vec3 origin = vec3(0);
		vec3 direction = vec3(0);

		__device__ __host__ Ray() = default;
		__device__ __host__ Ray(vec3 origin, vec3 direction) : origin(origin), direction(direction) {}

		__device__ __host__ vec3 at(float t) const { return origin + direction * t; }
	};
}