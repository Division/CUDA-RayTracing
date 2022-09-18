#pragma once

#include "utils/CUDAHelper.h"
#include "Math.h"
#include "Random.h"

namespace RayTracing
{
	using namespace Math;

	struct GPUCamera
	{
		CUDA_HOST_DEVICE Ray GetRay(vec2 uv) const { return Ray(origin, lower_left_corner + uv.x * horizontal + uv.y * vertical - origin); }
		CUDA_HOST_DEVICE const vec2& GetViewportWorldspaceSize() const { return viewport_worldspace_size; }

	protected:
		vec3 origin = vec3(0);
		vec2 viewport_worldspace_size = vec2(1);
		float aspect = 1.0f;
		vec3 horizontal = vec3(0);
		vec3 vertical = vec3(0);
		vec3 lower_left_corner = vec3(0);
	};

	struct alignas(16) GPUVertex
	{
		vec3 position;
		vec3 normal;
		vec2 uv;
	};

	struct alignas(16) GPUFace
	{
		uint32_t v0;
		uint32_t v2;
		uint32_t v1;
		uint32_t material = 0;
	};

	struct alignas(32) GPUBVHNode
	{
		AABB bounds;
		uint32_t first_index = 0; // if IsLeaf() it's index of the first primitive. Otherwise it's the index of the first child node.

		// Number if triangles in the node. When non-zero it's a leaf node with no children.
		// If 0 the node has 2 child nodes at indices (first_index) and (first_index+1)
		uint32_t prim_count = 0; 

		CUDA_HOST_DEVICE bool IsLeaf() const { return prim_count > 0; }
	};

	struct alignas(16) GeometryTriangle
	{
		vec3 va, vb, vc;
		vec3 normal;
		int material = 0;
	};

	struct alignas(16) GeometrySphere
	{
		vec3 position;
		float radius;
		int material = 0;
	};

	struct alignas(16) GPUMaterial
	{
		vec4 albedo = vec4(1);
		vec4 emissive = vec4(0);
		vec4 specular = vec4(0);
		float roughness = 0.9f;
		float specular_percent = 0.0f; 
		float IOR = 1.0f;
	};

	struct GPUScene
	{
		const GeometrySphere* gpu_spheres = nullptr;
		const GPUMaterial* gpu_materials = nullptr;
		const GPUBVHNode* gpu_bvh_nodes = nullptr;
		const uint32_t* gpu_bvh_face_indices = nullptr; // maps GPUBVHNode face indices to gpu_faces
		const GPUVertex* gpu_vertices = nullptr;
		const GPUFace* gpu_faces = nullptr;

		int sphere_count = 0;
		int material_count = 0;
		curandState* rng_state = nullptr;
		cudaTextureObject_t environment_cubemap_tex = 0;

		GPUCamera camera;

		CUDA_DEVICE int GetSphereCount() { return sphere_count; };
		CUDA_DEVICE const GeometrySphere* GetSphereData(int index) { return gpu_spheres + index; };
		CUDA_DEVICE const GPUMaterial* GetMaterial(int index) const { return gpu_materials + index; }
		CUDA_DEVICE RNG GetRNGForPixel(int screen_width, int x, int y) { return RNG(rng_state + y * screen_width + x); }
	};

}
