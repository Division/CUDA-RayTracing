#include "AssimpLoader.h"
#include <assimp/postprocess.h>
#include <assimp/StringUtils.h>

namespace Loader
{

	void CopyNodes(AssimpScene* scene, const aiNode* node, aiMatrix4x4 accTransform) 
    {
        const auto convert_matrix = [](aiMatrix4x4 m) -> glm::mat4 {
            glm::mat4 result = (glm::mat4&)m;
            return glm::transpose(result);
        };

		for (uint32_t i = 0; i < node->mNumMeshes; i++)
		{
			auto& new_mesh = scene->mesh_nodes.emplace_back();
			new_mesh.mesh = scene->scene->mMeshes[node->mMeshes[i]];
			new_mesh.transform = convert_matrix(accTransform * node->mTransformation);
			new_mesh.name = scene->scene->mMeshes[node->mMeshes[i]]->mName.C_Str();
		}

		for (uint32_t i = 0; i < node->mNumChildren; i++)
		{
			CopyNodes(scene, node->mChildren[i], accTransform * node->mTransformation);
		}
	};

    std::unique_ptr<AssimpScene> ImportScene(const std::string& pFile) 
    {
        std::unique_ptr<AssimpScene> scene = std::make_unique<AssimpScene>();

        scene->scene = scene->importer.ReadFile(pFile,
            aiProcess_CalcTangentSpace |
            aiProcess_Triangulate |
            aiProcess_JoinIdenticalVertices |
            aiProcess_SortByPType);

        if (!scene) {
            std::cout << "Error message = " << scene->importer.GetErrorString() << std::endl; 
            return nullptr; 
        } 

		aiMatrix4x4 m;
		aiMatrix4x4::RotationX(-(float)M_PI / 2, m);
        CopyNodes(scene.get(), scene->scene->mRootNode, m);

        return scene;
    }
}