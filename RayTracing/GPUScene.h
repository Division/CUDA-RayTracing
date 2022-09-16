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
		uint32_t material;
	};

	struct BVHNode
	{
		AABB bounds;
		uint32_t left_child = 0;
		uint32_t first_prim = 0;
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
		const GeometryTriangle* gpu_triangles = nullptr;
		const GPUMaterial* gpu_materials = nullptr;
		int sphere_count = 0;
		int triangle_count = 0;
		int material_count = 0;
		curandState* rng_state = nullptr;
		cudaTextureObject_t environment_cubemap_tex = 0;

		GPUCamera camera;

		CUDA_DEVICE int GetSphereCount() { return sphere_count; };
		CUDA_DEVICE int GetTriangleCount() { return triangle_count; };
		CUDA_DEVICE const GeometrySphere* GetSphereData(int index) { return gpu_spheres + index; };
		CUDA_DEVICE const GeometryTriangle* GetTriangleData(int index) { return gpu_triangles + index; };
		CUDA_DEVICE const GPUMaterial* GetMaterial(int index) { return gpu_materials + index; }
		CUDA_DEVICE RNG GetRNGForPixel(int screen_width, int x, int y) { return RNG(rng_state + y * screen_width + x); }
	};

}
