#pragma once

#include "image/DXGIFormat.h"
#include "CUDAHelper.h"
#include <gsl/span>

namespace CUDA
{

	struct CUDAChannelFormat
    {
        cudaChannelFormatDesc desc;
        bool normalized = false;
    };

	struct TextureLevelCopyOp
	{
		int level = 0;
		gsl::span<const uint8_t> data;
	};

	struct TextureDesc
	{
		TextureDesc(int width, int height, CUDAChannelFormat format, int levels = 1)
			: width(width), height(height), format(format), levels(levels)
		{
		}

		TextureDesc& SetCube() { is_cube = true; depth = 6; return *this; }
		TextureDesc& Data(gsl::span<const uint8_t> data) { AddLevelCopy({ 0, data }); return *this; }
		TextureDesc& AddLevelCopy(TextureLevelCopyOp copy) { level_copies.push_back(copy); return *this; }
		TextureDesc& SRGB(bool sRGB) { this->sRGB = sRGB; return *this; }

		int width = 0;
		int height = 0;
		int depth = 0;
		int levels = 1;
		std::vector<TextureLevelCopyOp> level_copies;
		CUDAChannelFormat format;
		bool is_cube = false;
		bool sRGB = true;
		bool normalize_texcoord = true;
	};

	class DeviceMemory;

	class Texture
	{
		std::unique_ptr<DeviceMemory> memory;
		Handle<cudaMipmappedArray_t> array_object;
		Handle<cudaTextureObject_t> texture_object;	
	public:
		Texture(const TextureDesc& desc);
		~Texture();
		cudaTextureObject_t GetTexture() const;
	};

	CUDAChannelFormat DXGIToCUDAFormat(DXGI_FORMAT format);
}

namespace Loader
{
	std::unique_ptr<CUDA::Texture> LoadDDSFromMemory(const gsl::span<const uint8_t> data, bool sRGB = true);
	std::unique_ptr<CUDA::Texture> LoadDDSFromFile(const std::wstring& path, bool sRGB = true);
}