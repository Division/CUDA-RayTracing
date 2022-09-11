#include "cuda_runtime.h"
#include "Scene.h"
#include "GPUScene.h"
#include "utils/CUDAHelper.h"
#include "utils/CUDATexture.h"
#include "Math.h"

namespace RayTracing
{
	using namespace Math;

	Scene::Scene()
	{
		environment_cubemap = Loader::LoadDDSFromFile(L"data/sunset_uncompressed.dds");
	}

	Scene::~Scene() = default;

	void Scene::AddSphere(vec3 position, float radius)
	{
		needs_upload = true;
		spheres.push_back(GeometrySphere{ position, radius });
	}

	void Scene::Upload(curandState* rng_state)
	{
		this->rng_state = rng_state;
		environment_cubemap_tex = environment_cubemap->GetTexture();

		if (!needs_upload)
			return;

		size_t total_size = 0;
		total_size += spheres.size() * sizeof(GeometrySphere);

		if (!memory || memory->GetSize() < total_size)
			memory = std::make_unique<CUDA::DeviceMemory>(total_size);

		sphere_count = (int)spheres.size();
		auto result = cudaMemcpy(memory->GetMemory(), spheres.data(), total_size, cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
			throw std::runtime_error("failed uploading scene");

		gpu_memory = (char*)memory->GetMemory();
		needs_upload = false;
	}

	void* Scene::GetMemory() const
	{
		return memory->GetMemory();
	}

}