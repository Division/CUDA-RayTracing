#include "DebugDraw.h"
#include "imgui.h"

namespace DebugDraw
{
	namespace
	{
		struct Line
		{
			glm::vec3 a, b;
			float thickness = 1.0f;
			uint32_t color = 0xFFFFFFFF;
		};

		struct Line2D
		{
			glm::vec2 a, b;
			float thickness = 1.0f;
			uint32_t color = 0xFFFFFFFF;
		};

		std::vector<Line> lines;
		std::vector<Line2D> lines_2d;
	}

	void DrawLine(glm::vec3 a, glm::vec3 b, uint32_t color, float thickness)
	{
		lines.emplace_back(a, b, thickness, color);
	}

	void DrawLine2D(glm::vec2 a, glm::vec2 b, uint32_t color, float thickness)
	{
		lines_2d.emplace_back(a, b, thickness, color);
	}

	void DrawAABB(const Math::AABB& aabb, uint32_t color, float thickness)
	{
		const auto& origin = aabb.min;
		const auto dim = aabb.max - aabb.min;
		std::array<glm::vec3, 4> bottom = { origin, origin + glm::vec3(dim.x, 0, 0), origin + glm::vec3(dim.x, 0, dim.z), origin + glm::vec3(0, 0, dim.z) };

		for (int i = 0; i < 4; i++)
		{
			DrawLine(bottom[i], bottom[(i + 1) % 4], color, thickness);
			DrawLine(bottom[i] + glm::vec3(0, dim.y, 0), bottom[(i + 1) % 4] + glm::vec3(0, dim.y, 0), color, thickness);
			DrawLine(bottom[i], bottom[i] + glm::vec3(0, dim.y, 0), color, thickness);
		}
	}

	void Flush(const glm::mat4& view, const glm::mat4& proj)
	{
		const auto view_proj = view * proj;
		ImDrawList* drawList = ImGui::GetWindowDrawList();
		const auto viewport = ImGui::GetMainViewport();
		auto viewport_size = glm::vec4(0, 0, viewport->Size.x, viewport->Size.y);
			
		auto project = [&](const glm::vec4& pos, const glm::mat4& m, glm::vec2 viewport_size)
		{
			return glm::project(glm::vec3(pos), view, proj, glm::vec4(0, 0, viewport_size.x, viewport_size.y));
		};

		for (auto& line : lines)
		{
			auto projected_a = glm::project(line.a, view, proj, viewport_size);
			auto projected_b = glm::project(line.b, view, proj, viewport_size);
			if (projected_a.z < 0 || projected_b.z < 0 || !std::isfinite(projected_a.z) || !std::isfinite(projected_b.z))
				continue;

			drawList->AddLine(glm::vec2(projected_a), glm::vec2(projected_b), line.color, line.thickness);
		}
		lines.clear();

		for (auto& line : lines_2d)
		{
			drawList->AddLine(line.a, line.b, line.color, line.thickness);
		}
		lines_2d.clear();
	}
}