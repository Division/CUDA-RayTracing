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

	struct Scene : public GPUScene
	{
		Scene();
		~Scene();
		Scene(const Scene&) = delete;
		Scene(Scene&&) = delete;
		Scene& operator=(const Scene&) = delete;
		Scene& operator=(Scene&&) = delete;

		void AddSphere(glm::vec3 position, float radius);
		void Upload(curandState* rng_state);
		void* GetMemory() const;
		//cudaTextureObject_t environment_cubemap_tex = 0;

	private:
		std::vector<GeometrySphere> spheres;
		std::unique_ptr<CUDA::DeviceMemory> memory;
		std::unique_ptr<CUDA::Texture> environment_cubemap;
		bool needs_upload = false;
	};

}