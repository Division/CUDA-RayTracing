#include "cuda_runtime.h"
#include "Scene.h"
#include "GPUScene.h"
#include "utils/CUDAHelper.h"
#include "utils/CUDATexture.h"
#include "Math.h"
#include "utils/Memory.h"
#include "utils/AssimpLoader.h"
#include "BVH.h"

namespace RayTracing
{
	using namespace Math;

	void Camera::Update()
	{
		aspect = viewport_size.x / viewport_size.y;
		transform = Math::ComposeMatrix(origin, quat(vec3(glm::radians(GetXAngle()), glm::radians(GetYAngle()), 0)), vec3(1));
		projection = glm::perspectiveRH(glm::radians(fov_y), aspect, 1.0f, 1000.0f);
		view = glm::inverse(transform);

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
		bvh = std::make_unique<BVH>();
	}

	Scene::~Scene() = default;

	void Scene::AddTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, int material)
	{
		const vec3 normal = glm::normalize(glm::cross(c - b, a - b));
		const uint32_t index0 = (uint32_t)vertices.size();
		GPUVertex va;
		va.position = a;
		va.normal = normal;
		va.uv = vec2(0);
		GPUVertex vb;
		vb.position = b;
		vb.normal = normal;
		vb.uv = vec2(0);
		GPUVertex vc;
		vc.position = c;
		vc.normal = normal;
		vc.uv = vec2(0);
		vertices.push_back(va);
		vertices.push_back(vb);
		vertices.push_back(vc);
		faces.push_back(GPUFace{ index0, index0 + 1, index0 + 2, (uint32_t)material });
		AddDirtyFlag(DirtyFlagValue::SceneMemory);
	}

	void Scene::AddSphere(vec3 position, float radius, int material)
	{
		spheres.push_back(GeometrySphere{ position, radius, material });
		AddDirtyFlag(DirtyFlagValue::SceneMemory);
	}

	void Scene::AddLoadedScene(const Loader::AssimpScene& scene, const glm::mat4& transform, int default_material)
	{
		for (auto& mesh : scene.mesh_nodes)
		{
			AddDirtyFlag(DirtyFlagValue::BVH);
			AddDirtyFlag(DirtyFlagValue::SceneMemory);

			//if (!mesh.mesh->HasTextureCoords(0))
			//	throw std::runtime_error("no UV");

			if (!mesh.mesh->HasNormals())
				throw std::runtime_error("no normals");

			const uint32_t index_offset = (uint32_t)vertices.size();
			auto mesh_transform = transform * mesh.transform;
			for (uint32_t v = 0; v < mesh.mesh->mNumVertices; v++)
			{
				auto& pos = mesh.mesh->mVertices[v];
				auto& normal = mesh.mesh->mNormals[v];
				auto uv = mesh.mesh->HasTextureCoords(0) ? mesh.mesh->mTextureCoords[0][v] : aiVector3D(0);

				GPUVertex vertex;
				vertex.position = mesh_transform * glm::vec4(pos.x, pos.y, pos.z, 1.0f);
				vertex.normal = mesh_transform * glm::vec4(normal.x, normal.y, normal.z, 0.0f);
				vertex.uv = glm::vec2(uv.x, uv.y);
				vertices.push_back(vertex);
			}

			for (uint32_t face_index = 0; face_index < mesh.mesh->mNumFaces; face_index++)
			{
				auto mesh_transform = transform * mesh.transform;
				auto face = mesh.mesh->mFaces[face_index];
				if (face.mNumIndices != 3)
					throw std::runtime_error("mesh is not a triangle");

				GPUFace gpu_face;
				gpu_face.v0 = face.mIndices[0] + index_offset;
				gpu_face.v1 = face.mIndices[1] + index_offset;
				gpu_face.v2 = face.mIndices[2] + index_offset;
				gpu_face.material = default_material;
				faces.push_back(gpu_face);

				glm::vec3 v0 = mesh_transform * glm::vec4(
					mesh.mesh->mVertices[face.mIndices[0]].x, mesh.mesh->mVertices[face.mIndices[0]].y, mesh.mesh->mVertices[face.mIndices[0]].z, 1
				);
				glm::vec3 v1 = mesh_transform * glm::vec4(
					mesh.mesh->mVertices[face.mIndices[1]].x, mesh.mesh->mVertices[face.mIndices[1]].y, mesh.mesh->mVertices[face.mIndices[1]].z, 1
				);
				glm::vec3 v2 = mesh_transform * glm::vec4(
					mesh.mesh->mVertices[face.mIndices[2]].x, mesh.mesh->mVertices[face.mIndices[2]].y, mesh.mesh->mVertices[face.mIndices[2]].z, 1
				);

				AddTriangle(v0, v1, v2, default_material);
			}

		}

	}
	
	uint32_t Scene::AddMaterial(Material material)
	{
		AddDirtyFlag(DirtyFlagValue::SceneMemory);
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
			AddDirtyFlag(DirtyFlagValue::Samples);
		}
		if (GetAsyncKeyState('E'))
		{
			camera.SetYAndle(camera.GetYAngle() - 90 * dt);
			AddDirtyFlag(DirtyFlagValue::Samples);
		}
		if (GetAsyncKeyState('W'))
		{
			camera.SetPosition(camera.GetPosition() + camera.GetForward() * movement_speed * dt);
			AddDirtyFlag(DirtyFlagValue::Samples);
		}
		if (GetAsyncKeyState('S'))
		{
			camera.SetPosition(camera.GetPosition() + camera.GetBackward() * movement_speed * dt);
			AddDirtyFlag(DirtyFlagValue::Samples);
		}
		if (GetAsyncKeyState('A'))
		{
			camera.SetPosition(camera.GetPosition() + camera.GetLeft() * movement_speed * dt);
			AddDirtyFlag(DirtyFlagValue::Samples);
		}
		if (GetAsyncKeyState('D'))
		{
			camera.SetPosition(camera.GetPosition() + camera.GetRight() * movement_speed * dt);
			AddDirtyFlag(DirtyFlagValue::Samples);
		}

		camera.Update();
	}

	void Scene::Upload(curandState* rng_state)
	{
		GPUScene::camera = camera;

		this->rng_state = rng_state;
		environment_cubemap_tex = environment_cubemap->GetTexture();

		if (IsFlagDirty(DirtyFlagValue::BVH))
		{
			bvh->Calculate(vertices, faces);

			bvh_memory = std::make_unique<CUDA::DeviceMemory>(bvh->GetGPUBVHNodes().size_bytes());
			gpu_bvh_nodes = (GPUBVHNode*)(bvh_memory->GetMemory());
			CUDA_CHECK(cudaMemcpy((void*)gpu_bvh_nodes, bvh->GetGPUBVHNodes().data(), bvh->GetGPUBVHNodes().size_bytes(), cudaMemcpyHostToDevice));

			bvh_face_index_memory = std::make_unique<CUDA::DeviceMemory>(bvh->GetFaceIndices().size_bytes());
			gpu_bvh_face_indices = (uint32_t*)bvh_face_index_memory->GetMemory();
			CUDA_CHECK(cudaMemcpy((void*)gpu_bvh_face_indices, bvh->GetFaceIndices().data(), bvh->GetFaceIndices().size_bytes(), cudaMemcpyHostToDevice));
		}

		if (IsFlagDirty(DirtyFlagValue::SceneMemory))
		{
			size_t total_size = 0;
			total_size += spheres.size() * sizeof(GeometrySphere);

			const size_t total_materials_size = materials.size() * sizeof(GPUMaterial);

			if (!memory || memory->GetSize() < total_size)
				memory = std::make_unique<CUDA::DeviceMemory>(total_size);

			sphere_count = (int)spheres.size();
			gpu_spheres = (GeometrySphere*)memory->GetMemory();
			CUDA_CHECK(cudaMemcpy(memory->GetMemory(), spheres.data(), spheres.size() * sizeof(GeometrySphere), cudaMemcpyHostToDevice));

			if (!materials_memory || materials_memory->GetSize())
				materials_memory = std::make_unique<CUDA::DeviceMemory>(total_materials_size);

			material_count = (int)materials.size();
			CUDA_CHECK(cudaMemcpy(materials_memory->GetMemory(), materials.data(), total_materials_size, cudaMemcpyHostToDevice));

			gpu_materials = (Material*)materials_memory->GetMemory();

			vertices_memory = std::make_unique<CUDA::DeviceMemory>(vertices.size() * sizeof(GPUVertex));
			gpu_vertices = (GPUVertex*)vertices_memory->GetMemory();
			CUDA_CHECK(cudaMemcpy(vertices_memory->GetMemory(), vertices.data(), vertices.size() * sizeof(GPUVertex), cudaMemcpyHostToDevice));

			faces_memory = std::make_unique<CUDA::DeviceMemory>(faces.size() * sizeof(GPUFace));
			gpu_faces = (GPUFace*)faces_memory->GetMemory();
			CUDA_CHECK(cudaMemcpy(faces_memory->GetMemory(), faces.data(), faces.size() * sizeof(GPUFace), cudaMemcpyHostToDevice));
		}

		dirty_flags = 0;
	}

	void* Scene::GetMemory() const
	{
		return memory->GetMemory();
	}


	void Scene::DebugDraw()
	{
		bvh->DebugDraw();
	}
}