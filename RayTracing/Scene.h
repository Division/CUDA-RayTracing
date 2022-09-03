#pragma once

#include "utils/Math.h"

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
		}

		vec3 origin = vec3(0);
		vec2 viewport_size = vec2(0);
		float aspect = 1.0f;
	};
}