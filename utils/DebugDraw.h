#pragma once

#include "raytracing/Math.h"

namespace DebugDraw
{
	void Flush(const glm::mat4& view, const glm::mat4& proj);
	void DrawLine(glm::vec3 a, glm::vec3 b, uint32_t color = 0xFFFFFFFF, float thickness = 1.0f);
	void DrawLine2D(glm::vec2 a, glm::vec2 b, uint32_t color = 0xFFFFFFFF, float thickness = 1.0f);
	void DrawAABB(const Math::AABB& aabb, uint32_t color = 0xFFFFFFFF, float thickness = 1.0f);
}

