#pragma once

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure

namespace Loader
{
	
	struct AssimpScene
	{
		struct MeshNode
		{
			std::string name;
			const aiMesh* mesh;
			glm::mat4 transform;
		};

		Assimp::Importer importer;
		const aiScene* scene = nullptr;
		std::vector<MeshNode> mesh_nodes;
	};

	std::unique_ptr<AssimpScene> ImportScene(const std::string& pFile);

}
