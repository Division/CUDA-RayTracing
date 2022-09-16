#pragma once

#include "glm/glm.hpp"
#include "glm/gtx/compatibility.hpp"
#include "glm/gtx/intersect.hpp"
#include "glm/gtc/quaternion.hpp"
#include "utils/CUDAHelper.h"

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
	using quat = glm::quat;

	struct AABB
	{
		vec3 min = vec3(1e20f);
		vec3 max = vec3(-1e20f);
	};

	struct Ray
	{
		vec3 origin = vec3(0);
		vec3 direction = vec3(0);

		CUDA_HOST_DEVICE Ray() = default;
		CUDA_HOST_DEVICE Ray(vec3 origin, vec3 direction) : origin(origin), direction(direction) {}

		CUDA_HOST_DEVICE vec3 at(float t) const { return origin + direction * t; }
	};

	CUDA_HOST_DEVICE inline mat4 ComposeMatrix(const vec3& position, const quat& rotation, const vec3& scale)
	{
		mat4 result = glm::identity<mat4>();
		result = glm::translate(result, position);
		result *= glm::mat4_cast(rotation);
		result = glm::scale(result, scale);
		return result;
	}
}