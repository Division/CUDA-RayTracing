#pragma once

#include "Math.h"
#include <span>
#include "GPUScene.h"

namespace Loader
{
	struct AssimpScene;

}
namespace RayTracing
{
	using namespace Math;


	class BVH
	{
		struct Triangle
		{
			vec3 centroid;
			uint32_t index;
		};

		std::vector<GPUBVHNode> nodes;
		std::vector<Triangle> triangles;
		std::span<const GPUVertex> vertices;
		std::span<const GPUFace> faces;
		uint32_t root_node_id = 0;
		uint32_t nodes_used = 1;
		std::vector<uint32_t> face_indices;

	private:
		void UpdateBounds(uint32_t node);
		void Subdivide(uint32_t node);

	public:
		void Calculate(std::span<const GPUVertex> vertices, std::span<const GPUFace> faces);
		void DebugDraw();
		std::span<const GPUBVHNode> GetGPUBVHNodes() const { return std::span<const GPUBVHNode>(nodes.data(), nodes_used); }
		std::span<const uint32_t> GetFaceIndices() const { return face_indices; }
	};
}
