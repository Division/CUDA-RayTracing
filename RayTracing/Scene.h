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

namespace RayTracing
{

	struct Scene;

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

	private:
		Scene& scene;

		vec2 viewport_size = vec2(1);
		float angle_x = 0;
		float angle_y = 0;
		float fov_y = 90;
		mat4 transform = mat4(1);
		mat4 projection = mat4(1);
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

		void AddSphere(glm::vec3 position, float radius, int material = 0);
		void AddTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, int material = 0);
		void AddQuad(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, int material = 0) { AddTriangle(a, b, c, material); AddTriangle(c, d, a, material); }
		uint32_t AddMaterial(Material material);
		void Update(float dt);
		void Upload(curandState* rng_state);
		void* GetMemory() const;

		Camera& GetCamera() { return camera; }

		void SetDirty() { needs_upload = true; }
		bool GetDirty() const { return needs_upload; }

	private:
		Camera camera;
		std::vector<GeometrySphere> spheres;
		std::vector<GeometryTriangle> triangles;
		std::vector<Material> materials;
		std::unique_ptr<CUDA::DeviceMemory> memory;
		std::unique_ptr<CUDA::DeviceMemory> materials_memory;
		std::unique_ptr<CUDA::Texture> environment_cubemap;
		bool needs_upload = false;
	};

}