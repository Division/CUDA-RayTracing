#include "CUDATexture.h"
#include "image/DDSLoaderHelpers.h"
#include "image/DDSLoader.h"
#include "FileLoader.h"

namespace CUDA
{

    CUDAChannelFormat DXGIToCUDAFormat(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return { cudaCreateChannelDesc<float4>(), false };
        case DXGI_FORMAT_R32G32B32A32_UINT: return { cudaCreateChannelDesc<uint4>(), false };
        case DXGI_FORMAT_R32G32B32A32_SINT: return { cudaCreateChannelDesc<int4>(), false };
        case DXGI_FORMAT_R32G32B32_FLOAT: return { cudaCreateChannelDesc<float3>() };
        case DXGI_FORMAT_R32G32B32_UINT: return { cudaCreateChannelDesc<uint3>() };
        case DXGI_FORMAT_R32G32B32_SINT: return { cudaCreateChannelDesc<int3>() };
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return { cudaCreateChannelDescHalf4() };
        case DXGI_FORMAT_R16G16B16A16_UNORM: return { cudaCreateChannelDesc<ushort4>(), true };
        case DXGI_FORMAT_R16G16B16A16_UINT: return { cudaCreateChannelDesc<uint4>() };
        case DXGI_FORMAT_R16G16B16A16_SNORM: return { cudaCreateChannelDesc<short4>(), true };
        case DXGI_FORMAT_R16G16B16A16_SINT: return { cudaCreateChannelDesc<short2>() };
        case DXGI_FORMAT_R32G32_FLOAT: return { cudaCreateChannelDesc<float2>() };
        case DXGI_FORMAT_R32G32_UINT: return { cudaCreateChannelDesc<uint2>() };
        case DXGI_FORMAT_R32G32_SINT: return { cudaCreateChannelDesc<int2>() };
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return { cudaCreateChannelDesc<float1>() };
        case DXGI_FORMAT_R10G10B10A2_UNORM: return { cudaChannelFormatDesc{10, 10, 10, 2, cudaChannelFormatKindUnsigned}, true };
        case DXGI_FORMAT_R10G10B10A2_UINT: return { cudaChannelFormatDesc{10, 10, 10, 2, cudaChannelFormatKindUnsigned } };
        case DXGI_FORMAT_R8G8B8A8_UNORM: return { cudaCreateChannelDesc<uchar4>(), true };
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return { cudaCreateChannelDesc<uchar4>(), true };
        case DXGI_FORMAT_R8G8B8A8_UINT: return { cudaCreateChannelDesc<uchar4>() };
        case DXGI_FORMAT_R8G8B8A8_SNORM: return { cudaCreateChannelDesc<char4>(), true };
        case DXGI_FORMAT_R8G8B8A8_SINT: return { cudaCreateChannelDesc<char4>() };
        case DXGI_FORMAT_R16G16_FLOAT: return { cudaCreateChannelDescHalf2() };
        case DXGI_FORMAT_R16G16_UNORM: return { cudaCreateChannelDesc<ushort2>(), true };
        case DXGI_FORMAT_R16G16_UINT: return { cudaCreateChannelDesc<ushort2>() };
        case DXGI_FORMAT_R16G16_SNORM: return { cudaCreateChannelDesc<short2>(), true };
        case DXGI_FORMAT_R16G16_SINT: return { cudaCreateChannelDesc<short2>() };
        case DXGI_FORMAT_R32_FLOAT: return { cudaCreateChannelDesc<float1>() };
        case DXGI_FORMAT_R32_UINT: return { cudaCreateChannelDesc<uint1>() };
        case DXGI_FORMAT_R32_SINT: return { cudaCreateChannelDesc<int1>() };
        case DXGI_FORMAT_R8G8_UNORM: return { cudaCreateChannelDesc<uchar2>(), true };
        case DXGI_FORMAT_R8G8_UINT: return { cudaCreateChannelDesc<uchar2>() };
        case DXGI_FORMAT_R8G8_SNORM: return { cudaCreateChannelDesc<char2>(), true };
        case DXGI_FORMAT_R8G8_SINT: return { cudaCreateChannelDesc<char2>() };
        case DXGI_FORMAT_R16_FLOAT: return { cudaCreateChannelDescHalf() };
        case DXGI_FORMAT_R16_UNORM: return { cudaCreateChannelDesc<ushort1>(), true };
        case DXGI_FORMAT_R16_UINT: return { cudaCreateChannelDesc<ushort1>() };
        case DXGI_FORMAT_R16_SNORM: return { cudaCreateChannelDesc<short1>(), true };
        case DXGI_FORMAT_R16_SINT: return { cudaCreateChannelDesc<short1>() };
        case DXGI_FORMAT_R8_UNORM: return { cudaCreateChannelDesc<uchar1>(), true };
        case DXGI_FORMAT_R8_UINT: return { cudaCreateChannelDesc<uchar1>() };
        case DXGI_FORMAT_R8_SNORM: return { cudaCreateChannelDesc<char1>(), true };
        case DXGI_FORMAT_R8_SINT: return { cudaCreateChannelDesc<char1>() };
        case DXGI_FORMAT_A8_UNORM: return { cudaCreateChannelDesc<uchar1>(), true };
        case DXGI_FORMAT_BC1_UNORM: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>(), true };
        case DXGI_FORMAT_BC1_UNORM_SRGB: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>(), true };
        case DXGI_FORMAT_BC2_UNORM: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed2>(), true };
        case DXGI_FORMAT_BC2_UNORM_SRGB: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed2>(), true };
        case DXGI_FORMAT_BC3_UNORM: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>(), true };
        case DXGI_FORMAT_BC3_UNORM_SRGB: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>(), true };
        case DXGI_FORMAT_BC4_UNORM: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed4>(), true };
        case DXGI_FORMAT_BC4_SNORM: return { cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed4>(), true };
        case DXGI_FORMAT_BC5_UNORM: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>(), true };
        case DXGI_FORMAT_BC5_SNORM: return { cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed5>(), true };
        case DXGI_FORMAT_BC6H_UF16: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed6H>() };
        case DXGI_FORMAT_BC6H_SF16: return { cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed6H>() };
        case DXGI_FORMAT_BC7_UNORM: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>(), true };
        case DXGI_FORMAT_BC7_UNORM_SRGB: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>(), true };

        default:
            throw std::runtime_error("unsupported DXGI format");
        }
    }

    bool IsFormatCompressed(const cudaChannelFormatDesc& desc)
    {
        switch (desc.f)
        {
        case cudaChannelFormatKindUnsignedBlockCompressed1:
        case cudaChannelFormatKindUnsignedBlockCompressed1SRGB:
        case cudaChannelFormatKindUnsignedBlockCompressed2:
        case cudaChannelFormatKindUnsignedBlockCompressed2SRGB:
        case cudaChannelFormatKindUnsignedBlockCompressed3:
        case cudaChannelFormatKindUnsignedBlockCompressed3SRGB:
        case cudaChannelFormatKindUnsignedBlockCompressed4:
        case cudaChannelFormatKindSignedBlockCompressed4:
        case cudaChannelFormatKindUnsignedBlockCompressed5:
        case cudaChannelFormatKindSignedBlockCompressed5:
        case cudaChannelFormatKindUnsignedBlockCompressed6H:
        case cudaChannelFormatKindSignedBlockCompressed6H:
        case cudaChannelFormatKindUnsignedBlockCompressed7:
        case cudaChannelFormatKindUnsignedBlockCompressed7SRGB:
            return true;

        default:
            return false;
        }
    }

    int GetBitsPerPixel(const cudaChannelFormatDesc& desc)
    {
        return desc.x + desc.y + desc.z + desc.w;
    }

    int GetMipDimension(int size, int level)
    {
        return size >> level;
    }

    Texture::Texture(const TextureDesc& desc)
    {
        uint32_t flags = 0;
        if (desc.is_cube)
        {
            if (desc.width != desc.height)
                throw std::runtime_error("width != height");
            flags = cudaArrayCubemap;
        }

        const bool is_compressed = IsFormatCompressed(desc.format.desc);
        if (is_compressed && (desc.width % 4 != 0 || desc.height % 4 != 0))
            throw std::runtime_error("Block compressed format wrong size");

        const int block_size = is_compressed ? 4 : 1;

        const auto& channelDesc = desc.format;
        const auto texture_extent = make_cudaExtent(desc.width / block_size, desc.height / block_size, desc.depth);
        const auto bpp = GetBitsPerPixel(channelDesc.desc);

        array_object = CreateMipmappedArray(&channelDesc.desc, texture_extent, desc.levels, flags);

        for (auto& copy : desc.level_copies)
        {
            cudaArray_t level_arr;
            CUDA_CHECK(cudaGetMipmappedArrayLevel(&level_arr, *array_object, copy.level));

            const int level_width = GetMipDimension((int)texture_extent.width, copy.level);
            const int level_height = GetMipDimension((int)texture_extent.height, copy.level);

            const auto level_size = level_width * level_height * texture_extent.depth * bpp / 8;
            if (copy.data.size_bytes() != level_size)
                throw std::runtime_error("provided data size doesn't match level size");

            cudaMemcpy3DParms myparms = { 0 };
            myparms.srcPos = make_cudaPos(0, 0, 0);
            myparms.dstPos = make_cudaPos(0, 0, 0);
            myparms.srcPtr = make_cudaPitchedPtr((void*)copy.data.data(), level_width * bpp / 8, level_width, level_height);
            myparms.dstArray = level_arr;
            myparms.extent = make_cudaExtent(level_width, level_height, desc.depth);
            myparms.kind = cudaMemcpyHostToDevice;
            CUDA_CHECK(cudaMemcpy3D(&myparms));
        }

        cudaResourceDesc res_desc;
        res_desc.resType = cudaResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = *array_object;

        cudaTextureDesc cuda_tex_desc;
        memset(&cuda_tex_desc, 0, sizeof(cudaTextureDesc));

        cuda_tex_desc.normalizedCoords = desc.normalize_texcoord;
        cuda_tex_desc.filterMode = cudaFilterModeLinear;
        cuda_tex_desc.addressMode[0] = cudaAddressModeWrap;
        cuda_tex_desc.addressMode[1] = cudaAddressModeWrap;
        cuda_tex_desc.addressMode[2] = cudaAddressModeWrap;
        cuda_tex_desc.readMode = channelDesc.normalized ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
        cuda_tex_desc.sRGB = desc.sRGB;
        cuda_tex_desc.seamlessCubemap = desc.is_cube;
        texture_object = CreateTextureObject(&res_desc, &cuda_tex_desc, nullptr);
    };

    Texture::~Texture()
    {
    }

    cudaTextureObject_t Texture::GetTexture() const
    {
        return texture_object.Get();
    }

}

namespace Loader
{
    std::unique_ptr<CUDA::Texture> CreateDDSTexture(const DirectX::DDSSubresourceData& data, bool sRGB)
    {
        CUDA::TextureDesc desc(data.width, data.height, CUDA::DXGIToCUDAFormat(data.format), data.mip_count);
        desc.SRGB(sRGB);

        if (data.is_cubemap)
            desc.SetCube();

        // Cuda texture layout is different so need to rearrange:
        // mip0_level0, mip0_level1, mip1_level0, mip1_level1
        std::deque<std::vector<uint8_t>> copy_src;
        for (UINT miplevel = 0; miplevel < data.mip_count; miplevel++)
        {
            const UINT array_elements_size = data.array_size * data.subresources[miplevel].SysMemSlicePitch;
            const UINT mip_size = data.subresources[miplevel].SysMemSlicePitch;

            // data for all array elements in a particular mip level
            auto& mip_data = copy_src.emplace_back((size_t)array_elements_size);

            for (UINT face = 0; face < data.array_size; face++)
            {
                auto dst = mip_data.data() + mip_size * face;
                auto src_index = face * data.mip_count + miplevel;
                assert(data.subresources.at(src_index).SysMemSlicePitch == mip_size);
                memcpy_s(dst, array_elements_size - mip_size * face, data.subresources.at(src_index).pSysMem, data.subresources.at(src_index).SysMemSlicePitch);
            }
        }

        int current_mip = 0;
        for (auto& copy : copy_src)
            desc.AddLevelCopy({ current_mip++, copy });

        return std::make_unique<CUDA::Texture>(desc);
    }

    std::unique_ptr<CUDA::Texture> LoadDDSFromMemory(const gsl::span<const uint8_t> data, bool sRGB)
    {
        // Validate DDS file in memory
        const DirectX::DDS_HEADER* header = nullptr;
        const uint8_t* bitData = nullptr;
        size_t bitSize = 0;

        HRESULT hr = LoadTextureDataFromMemory(data.data(), data.size(),
            &header,
            &bitData,
            &bitSize
        );

        auto subresources = GetDDSSubresources(header, bitData, bitSize);
        if (!subresources)
            return nullptr;

        return CreateDDSTexture(*subresources, sRGB);
    }

    std::unique_ptr<CUDA::Texture> LoadDDSFromFile(const std::wstring& path, bool sRGB)
    {
        auto dds_data = LoadFile(path);
        if (!dds_data)
            return nullptr;

        return LoadDDSFromMemory(*dds_data, sRGB);
    }

}
