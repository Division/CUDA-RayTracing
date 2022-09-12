#include "cuda_runtime.h"
#include "Scene.h"
#include "GPUScene.h"
#include "utils/CUDAHelper.h"
#include "utils/CUDATexture.h"
#include "Math.h"
#include "utils/Memory.h"

namespace RayTracing
{
	using namespace Math;

	void Camera::Update()
	{

		aspect = viewport_size.x / viewport_size.y;
		transform = Math::ComposeMatrix(origin, quat(vec3(glm::radians(GetXAngle()), glm::radians(GetYAngle()), 0)), vec3(1));
		projection = glm::perspectiveRH(glm::radians(fov_y), aspect, 10.0f, 1000.0f);

		auto inv_proj = glm::inverse(projection);

		auto lower_left_corner4 = inv_proj * vec4(-1, -1, -1, 1);
		auto upper_right_corner4 = inv_proj * vec4(1, 1, -1, 1);
		
		// lower left corner of front clip rect in the camera space
		lower_left_corner = lower_left_corner4 / lower_left_corner4.w;
		vec3 upper_right_corner = upper_right_corner4 / upper_right_corner4.w;
		viewport_worldspace_size = upper_right_corner - lower_left_corner;

		// Convert to worldspace
		horizontal = transform * vec4(viewport_worldspace_size.x, 0, 0, 0);
		vertical = transform * vec4(0, viewport_worldspace_size.y, 0, 0);
		lower_left_corner = transform * vec4(lower_left_corner, 1);
	}

	Scene::Scene() : camera(*this)
	{
		environment_cubemap = Loader::LoadDDSFromFile(L"data/sunset_uncompressed.dds");
	}

	Scene::~Scene() = default;

	void Scene::AddTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, int material)
	{
		const vec3 normal = glm::normalize(glm::cross(c - b, a - b));
		triangles.push_back({ a, b, c, normal, material });
		SetDirty();
	}

	void Scene::AddSphere(vec3 position, float radius, int material)
	{
		needs_upload = true;
		spheres.push_back(GeometrySphere{ position, radius, material });
		AddMaterial(Material{ vec4(1) });
	}

	uint32_t Scene::AddMaterial(Material material)
	{
		needs_upload = true;
		materials.push_back(material);
		return (uint32_t)materials.size() - 1;
	}

	void Scene::Update(float dt)
	{
		float movement_speed = 10;

		if (GetAsyncKeyState(VK_SHIFT))
			movement_speed *= 4;

		if (GetAsyncKeyState('Q'))
		{
			camera.SetYAndle(camera.GetYAngle() + 90 * dt);
			SetDirty();
		}
		if (GetAsyncKeyState('E'))
		{
			camera.SetYAndle(camera.GetYAngle() - 90 * dt);
			SetDirty();
		}
		if (GetAsyncKeyState('W'))
		{
			camera.SetPosition(camera.GetPosition() + camera.GetForward() * movement_speed * dt);
			SetDirty();
		}
		if (GetAsyncKeyState('S'))
		{
			camera.SetPosition(camera.GetPosition() + camera.GetBackward() * movement_speed * dt);
			SetDirty();
		}
		if (GetAsyncKeyState('A'))
		{
			camera.SetPosition(camera.GetPosition() + camera.GetLeft() * movement_speed * dt);
			SetDirty();
		}
		if (GetAsyncKeyState('D'))
		{
			camera.SetPosition(camera.GetPosition() + camera.GetRight() * movement_speed * dt);
			SetDirty();
		}

		camera.Update();
	}

	void Scene::Upload(curandState* rng_state)
	{
		GPUScene::camera = camera;

		this->rng_state = rng_state;
		environment_cubemap_tex = environment_cubemap->GetTexture();

		if (!needs_upload)
			return;

		size_t total_size = 0;
		total_size += spheres.size() * sizeof(GeometrySphere);

		const auto triangles_offset = Memory::AlignSize(total_size, alignof(GeometryTriangle));
		total_size += alignof(GeometryTriangle) /* for alignment */ + triangles.size() * sizeof(GeometryTriangle);

		const size_t total_materials_size = materials.size() * sizeof(GPUMaterial);

		if (!memory || memory->GetSize() < total_size)
			memory = std::make_unique<CUDA::DeviceMemory>(total_size);

		sphere_count = (int)spheres.size();
		gpu_spheres = (GeometrySphere*)memory->GetMemory();
		CUDA_CHECK(cudaMemcpy(memory->GetMemory(), spheres.data(), spheres.size() * sizeof(GeometrySphere), cudaMemcpyHostToDevice));

		triangle_count = (int)triangles.size();
		gpu_triangles = (GeometryTriangle*)((char*)memory->GetMemory() + triangles_offset);
		CUDA_CHECK(cudaMemcpy((void*)gpu_triangles, triangles.data(), triangles.size() * sizeof(GeometryTriangle), cudaMemcpyHostToDevice));

		if (!materials_memory || materials_memory->GetSize())
			materials_memory = std::make_unique<CUDA::DeviceMemory>(total_materials_size);

		material_count = (int)materials.size();
		CUDA_CHECK(cudaMemcpy(materials_memory->GetMemory(), materials.data(), total_materials_size, cudaMemcpyHostToDevice));

		gpu_materials = (Material*)materials_memory->GetMemory();
		needs_upload = false;
	}

	void* Scene::GetMemory() const
	{
		return memory->GetMemory();
	}

}