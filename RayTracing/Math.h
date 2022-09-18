#pragma once

#include "glm/glm.hpp"
#include "glm/gtx/compatibility.hpp"
#include "glm/gtx/intersect.hpp"
#include "glm/gtc/quaternion.hpp"
#include "utils/CUDAHelper.h"
#include <limits>

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
		vec3 min = vec3(1e30f);
		vec3 max = vec3(-1e30f);

		CUDA_HOST_DEVICE void Expand(const vec3 p)
		{
			min = glm::min(min, p);
			max = glm::max(max, p);
		}

		CUDA_HOST_DEVICE vec3 GetExtent() const { return max - min; }
	};

	struct Ray
	{
		vec3 origin = vec3(0);
		vec3 direction = vec3(0);

		CUDA_HOST_DEVICE Ray() = default;
		CUDA_HOST_DEVICE Ray(vec3 origin, vec3 direction) : origin(origin), direction(direction) {}

		CUDA_HOST_DEVICE vec3 at(float t) const { return origin + direction * t; }
	};

	CUDA_HOST_DEVICE inline bool IntersectAABB(const Ray& ray, const AABB& aabb, const float ray_length = 1e30f)
	{
		auto& bmin = aabb.min;
		auto& bmax = aabb.max;
		float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
		float tmin = CUDA_MIN(tx1, tx2), tmax = CUDA_MAX(tx1, tx2);
		float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
		tmin = CUDA_MAX(tmin, CUDA_MIN(ty1, ty2)), tmax = CUDA_MIN(tmax, CUDA_MAX(ty1, ty2));
		float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
		tmin = CUDA_MAX(tmin, CUDA_MIN(tz1, tz2)), tmax = CUDA_MIN(tmax, CUDA_MAX(tz1, tz2));
		return tmax >= tmin && tmin < ray_length && tmax > 0;
	}

	CUDA_HOST_DEVICE inline mat4 ComposeMatrix(const vec3& position, const quat& rotation, const vec3& scale)
	{
		mat4 result = glm::identity<mat4>();
		result = glm::translate(result, position);
		result *= glm::mat4_cast(rotation);
		result = glm::scale(result, scale);
		return result;
	}
}