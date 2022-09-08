#include "cuda_runtime.h"
#include "Scene.h"
#include "utils/CUDAHelper.h"

namespace RayTracing
{
	Scene::Scene() = default;
	Scene::~Scene() = default;

	void Scene::AddSphere(vec3 position, float radius)
	{
		needs_upload = true;
		spheres.push_back(GeometrySphere{ .position = position, .radius = radius });
	}

	void Scene::Upload(curandState* rng_state)
	{
		this->rng_state = rng_state;

		if (!needs_upload)
			return;

		size_t total_size = 0;
		total_size += spheres.size() * sizeof(GeometrySphere);

		if (!memory || memory->size < total_size)
			memory = std::make_unique<CUDAHelper::DeviceMemory>(total_size);

		sphere_count = (int)spheres.size();
		auto result = cudaMemcpy(memory->memory, spheres.data(), total_size, cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
			throw std::runtime_error("failed uploading scene");

		gpu_memory = (char*)memory->memory;
		needs_upload = false;
	}

	void* Scene::GetMemory() const
	{
		return memory->memory;
	}

}