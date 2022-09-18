#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include "GPUScene.h"

namespace CUDA
{
	class DeviceMemory;
	class Texture;
}

namespace Loader
{
	struct AssimpScene;
}

namespace RayTracing
{
	enum class DirtyFlagValue : uint32_t
	{
		Samples = 1 << 0,
		SceneMemory = 1 << 1,
		BVH = 1 << 2
	};

	typedef uint32_t DirtyFlags;

	struct Scene;
	class BVH;

	struct Camera : public GPUCamera
	{
		Camera(Scene& scene) : scene(scene) {}

		vec2 GetViewportSize() const { return viewport_size; }
		void SetViewportSize(vec2 value) { viewport_size = value; }

		const vec3& GetPosition() const { return origin; }
		void SetPosition(vec3 value) { origin = value; }

		const float GetXAngle() const { return angle_x; }
		void SetXAndle(float value) { angle_x = value; }

		const float GetYAngle() const { return angle_y; }
		void SetYAndle(float value) { angle_y = value; }

		void Update();

		vec3 GetForward() { return -transform[2]; }
		vec3 GetBackward() { return transform[2]; }
		vec3 GetRight() { return transform[0]; }
		vec3 GetLeft() { return -transform[0]; }
		vec3 GetUp() { return transform[1]; }
		vec3 GetDown() { return -transform[1]; }

		const mat4& GetViewMatrix() const { return view; }
		const mat4& GetProjectionMatrix() const { return projection; }

	private:
		Scene& scene;

		vec2 viewport_size = vec2(1);
		float angle_x = 0;
		float angle_y = 0;
		float fov_y = 90;
		mat4 transform = mat4(1);
		mat4 projection = mat4(1);
		mat4 view = mat4(1);
	};


	struct Material : public GPUMaterial
	{
		Material(vec3 albedo, vec3 emissive)
		{
			this->albedo = vec4(albedo, 1);
			this->emissive = vec4(emissive, 1);
		}

		Material(vec3 albedo) : Material(albedo, vec3(0)) {}

		Material() : Material(vec3(0)) {}
	};

	struct Scene : public GPUScene
	{
		Scene();
		~Scene();
		Scene(const Scene&) = delete;
		Scene(Scene&&) = delete;
		Scene& operator=(const Scene&) = delete;
		Scene& operator=(Scene&&) = delete;

		void AddSphere(vec3 position, float radius, int material = 0);
		void AddTriangle(vec3 a, vec3 b, vec3 c, int material = 0);
		void AddQuad(vec3 a, vec3 b, vec3 c, vec3 d, int material = 0) { AddTriangle(a, b, c, material); AddTriangle(c, d, a, material); }
		uint32_t AddMaterial(Material material);
		void Update(float dt);
		void Upload(curandState* rng_state);
		void* GetMemory() const;
		void AddLoadedScene(const Loader::AssimpScene& scene, const mat4& transform = glm::identity<mat4>(), int default_material = 0);

		Camera& GetCamera() { return camera; }

		void AddDirtyFlags(DirtyFlags flags = ~0) { dirty_flags |= flags; }
		void AddDirtyFlag(DirtyFlagValue flag) { AddDirtyFlags(static_cast<DirtyFlags>(flag)); }
		DirtyFlags GetDirtyFlags() const { return dirty_flags; }
		bool IsFlagDirty(DirtyFlagValue flag) const { return dirty_flags & static_cast<DirtyFlags>(flag); }

		void DebugDraw();

	private:
		Camera camera;
		std::vector<GeometrySphere> spheres;
		std::vector<Material> materials;
		std::vector<GPUFace> faces;
		std::vector<GPUVertex> vertices;
		std::unique_ptr<CUDA::DeviceMemory> memory;
		std::unique_ptr<CUDA::DeviceMemory> materials_memory;
		std::unique_ptr<CUDA::DeviceMemory> bvh_memory;
		std::unique_ptr<CUDA::DeviceMemory> bvh_face_index_memory;
		std::unique_ptr<CUDA::DeviceMemory> faces_memory;
		std::unique_ptr<CUDA::DeviceMemory> vertices_memory;
		std::unique_ptr<CUDA::Texture> environment_cubemap;
		std::unique_ptr<BVH> bvh;
		DirtyFlags dirty_flags = ~0;
	};

}