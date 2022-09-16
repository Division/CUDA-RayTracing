#include "BVH.h"
#include <functional>
#include "utils/DebugDraw.h"
#include "utils/AssimpLoader.h"

namespace RayTracing
{
	void BVH::Calculate(std::span<const GPUVertex> vertices, std::span<const GPUFace> faces)
	{
		nodes.resize(faces.size() * 2 - 1);
		triangles.reserve(faces.size());
		this->vertices = vertices;
		this->faces = faces;

		for (uint32_t i = 0; i < faces.size(); i++)
		{
			auto& v0 = vertices[faces[i].v0];
			auto& v1 = vertices[faces[i].v1];
			auto& v2 = vertices[faces[i].v2];
			Triangle t;
			t.centroid = (v0.position + v1.position + v2.position) / 3.0f;
			t.index = i;
			triangles.push_back(t);
		}

		nodes_used = 1;
		auto& root = nodes[root_node_id];
		root.left_child = 0;
		root.first_prim = 0;
		root.prim_count = (uint32_t)faces.size();

		UpdateBounds(root_node_id);
		Subdivide(root_node_id);

		uint32_t total_faces = 0;
		for (uint32_t i = 0; i < nodes_used; i++)
		{
			total_faces += nodes[i].prim_count;
		}
	}

	void BVH::UpdateBounds(uint32_t node_index)
	{
		auto& node = nodes.at(node_index);
		node.bounds = AABB();
		
		for (uint32_t i = 0; i < node.prim_count; i++)
		{
			auto& t = triangles.at(node.first_prim + i);
			node.bounds.Expand(vertices[faces[t.index].v0].position);
			node.bounds.Expand(vertices[faces[t.index].v1].position);
			node.bounds.Expand(vertices[faces[t.index].v2].position);
		}
	}

	void BVH::Subdivide(uint32_t node_index)
	{
		auto& node = nodes.at(node_index);
		auto extent = node.bounds.GetExtent();

		int axis1 = 0;
		if (extent.y > extent.x) axis1 = 1;
		if (extent.z > extent[axis1]) axis1 = 2;
		int axis2 = (axis1 + 1) % 3;
		int axis3 = (axis2 + 1) % 3;
		if (extent[axis3] > extent[axis2])
			std::swap(axis2, axis3);

		const std::array<int, 3> try_axis = { axis1, axis2, axis3 };

		bool left_empty = false;
		bool right_empty = false;
		bool found = false;
		int i = 0;
		int leftCount = 0;
		for (auto axis : try_axis)
		{
			float split_pos = node.bounds.min[axis] + extent[axis] * 0.5f;

			i = node.first_prim;
			int j = i + node.prim_count - 1;
			while (i <= j)
			{
				if (triangles[i].centroid[axis] < split_pos)
					i++;
				else
					std::swap(triangles[i], triangles[j--]);
			}
			leftCount = i - node.first_prim;

			left_empty = leftCount == 0;
			right_empty = leftCount == node.prim_count;
			
			if (!left_empty && !right_empty)
			{
				found = true;
				break;
			}
		}

		if (!found)
		{
			return;
		}

		// create child nodes
		int leftChildIdx = nodes_used++;
		int rightChildIdx = nodes_used++;
		node.left_child = leftChildIdx;
		nodes[leftChildIdx].first_prim = node.first_prim;
		nodes[leftChildIdx].prim_count = leftCount;
		nodes[rightChildIdx].first_prim = i;
		nodes[rightChildIdx].prim_count = node.prim_count - leftCount;
		node.prim_count = 0;

		UpdateBounds(leftChildIdx);
		UpdateBounds(rightChildIdx);

		Subdivide(leftChildIdx);
		Subdivide(rightChildIdx);
	}

	void BVH::DebugDraw()
	{
		uint32_t current_node = root_node_id;
		std::function<void(uint32_t)> draw;

		/*for (auto& t : triangles)
		{
			auto& v0 = vertices[faces[t.index].v0];
			auto& v1 = vertices[faces[t.index].v1];
			auto& v2 = vertices[faces[t.index].v2];
			DebugDraw::DrawLine(v0.position, v1.position);
			DebugDraw::DrawLine(v1.position, v2.position);
			DebugDraw::DrawLine(v2.position, v0.position);
		}
*/
		draw = [&](uint32_t node_index) 
		{
			auto& node = nodes[node_index];

			if (node.IsLeaf())
			{
				DebugDraw::DrawAABB(node.bounds);
				return;
			}

			draw(node.left_child);
			draw(node.left_child + 1);
		};

		draw(current_node);
	}
}
