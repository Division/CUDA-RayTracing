#include "DDSLoader.h"
#include "DDSLoaderHelpers.h"

namespace DirectX
{

    enum D3D11_RESOURCE_DIMENSION
    {
        D3D11_RESOURCE_DIMENSION_UNKNOWN = 0,
        D3D11_RESOURCE_DIMENSION_BUFFER = 1,
        D3D11_RESOURCE_DIMENSION_TEXTURE1D = 2,
        D3D11_RESOURCE_DIMENSION_TEXTURE2D = 3,
        D3D11_RESOURCE_DIMENSION_TEXTURE3D = 4
    } 	D3D11_RESOURCE_DIMENSION;

    enum D3D11_RESOURCE_MISC_FLAG
    {
        D3D11_RESOURCE_MISC_GENERATE_MIPS = 0x1L,
        D3D11_RESOURCE_MISC_SHARED = 0x2L,
        D3D11_RESOURCE_MISC_TEXTURECUBE = 0x4L,
        D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS = 0x10L,
        D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS = 0x20L,
        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED = 0x40L,
        D3D11_RESOURCE_MISC_RESOURCE_CLAMP = 0x80L,
        D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX = 0x100L,
        D3D11_RESOURCE_MISC_GDI_COMPATIBLE = 0x200L,
        D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800L,
        D3D11_RESOURCE_MISC_RESTRICTED_CONTENT = 0x1000L,
        D3D11_RESOURCE_MISC_RESTRICT_SHARED_RESOURCE = 0x2000L,
        D3D11_RESOURCE_MISC_RESTRICT_SHARED_RESOURCE_DRIVER = 0x4000L,
        D3D11_RESOURCE_MISC_GUARDED = 0x8000L,
        D3D11_RESOURCE_MISC_TILE_POOL = 0x20000L,
        D3D11_RESOURCE_MISC_TILED = 0x40000L,
        D3D11_RESOURCE_MISC_HW_PROTECTED = 0x80000L
    } 	D3D11_RESOURCE_MISC_FLAG;

    HRESULT FillInitData(
        size_t width,
        size_t height,
        size_t depth,
        size_t mipCount,
        size_t arraySize,
        DXGI_FORMAT format,
        size_t maxsize,
        size_t bitSize,
        const uint8_t* bitData,
        size_t& twidth,
        size_t& theight,
        size_t& tdepth,
        size_t& skipMip,
        SUBRESOURCE_DATA* initData) noexcept
    {
        if (!bitData || !initData)
        {
            return E_POINTER;
        }

        skipMip = 0;
        twidth = 0;
        theight = 0;
        tdepth = 0;

        size_t NumBytes = 0;
        size_t RowBytes = 0;
        const uint8_t* pSrcBits = bitData;
        const uint8_t* pEndBits = bitData + bitSize;

        size_t index = 0;
        for (size_t j = 0; j < arraySize; j++)
        {
            size_t w = width;
            size_t h = height;
            size_t d = depth;
            for (size_t i = 0; i < mipCount; i++)
            {
                HRESULT hr = GetSurfaceInfo(w, h, format, &NumBytes, &RowBytes, nullptr);
                if (FAILED(hr))
                    return hr;

                if (NumBytes > UINT32_MAX || RowBytes > UINT32_MAX)
                    return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);

                if ((mipCount <= 1) || !maxsize || (w <= maxsize && h <= maxsize && d <= maxsize))
                {
                    if (!twidth)
                    {
                        twidth = w;
                        theight = h;
                        tdepth = d;
                    }

                    assert(index < mipCount * arraySize);
                    initData[index].pSysMem = pSrcBits;
                    initData[index].SysMemPitch = static_cast<UINT>(RowBytes);
                    initData[index].SysMemSlicePitch = static_cast<UINT>(NumBytes);
                    initData[index].width = w;
                    initData[index].height = h;
                    initData[index].depth = d;
                    ++index;
                }
                else if (!j)
                {
                    // Count number of skipped mipmaps (first item only)
                    ++skipMip;
                }

                if (pSrcBits + (NumBytes * d) > pEndBits)
                {
                    return HRESULT_FROM_WIN32(ERROR_HANDLE_EOF);
                }

                pSrcBits += NumBytes * d;

                w = w >> 1;
                h = h >> 1;
                d = d >> 1;
                if (w == 0)
                {
                    w = 1;
                }
                if (h == 0)
                {
                    h = 1;
                }
                if (d == 0)
                {
                    d = 1;
                }
            }
        }

        return (index > 0) ? S_OK : E_FAIL;
    }

    std::optional<DDSSubresourceData> GetDDSSubresources(
        _In_ const DDS_HEADER* header,
        _In_reads_bytes_(bitSize) const uint8_t* bitData,
        _In_ size_t bitSize
    )
    {
        HRESULT hr = S_OK;

        UINT width = header->width;
        UINT height = header->height;
        UINT depth = header->depth;

        uint32_t resDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
        UINT arraySize = 1;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
        bool isCubeMap = false;

        size_t mipCount = header->mipMapCount;
        if (0 == mipCount)
        {
            mipCount = 1;
        }

        if ((header->ddspf.flags & DDS_FOURCC) &&
            (MAKEFOURCC('D', 'X', '1', '0') == header->ddspf.fourCC))
        {
            auto d3d10ext = reinterpret_cast<const DDS_HEADER_DXT10*>(reinterpret_cast<const char*>(header) + sizeof(DDS_HEADER));

            arraySize = d3d10ext->arraySize;
            if (arraySize == 0)
            {
                return std::nullopt;
            }

            switch (d3d10ext->dxgiFormat)
            {
            case DXGI_FORMAT_AI44:
            case DXGI_FORMAT_IA44:
            case DXGI_FORMAT_P8:
            case DXGI_FORMAT_A8P8:
                std::cout << "ERROR: DDSTextureLoader does not support video textures. Consider using DirectXTex instead";
                return std::nullopt;

            default:
                if (BitsPerPixel(d3d10ext->dxgiFormat) == 0)
                {
				    std::cout << "ERROR: Unknown DXGI format " << static_cast<uint32_t>(d3d10ext->dxgiFormat) << std::endl;
                    return std::nullopt;
                }
            }

            format = d3d10ext->dxgiFormat;

            switch (d3d10ext->resourceDimension)
            {
            case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
                // D3DX writes 1D textures with a fixed Height of 1
                if ((header->flags & DDS_HEIGHT) && height != 1)
                {
                    return std::nullopt;
                }
                height = depth = 1;
                break;

            case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
                if (d3d10ext->miscFlag & D3D11_RESOURCE_MISC_TEXTURECUBE)
                {
                    arraySize *= 6;
                    isCubeMap = true;
                }
                depth = 1;
                break;

            case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
                if (!(header->flags & DDS_HEADER_FLAGS_VOLUME))
                {
                    return std::nullopt;
                }

                if (arraySize > 1)
                {
                     std::cout << ("ERROR: Volume textures are not texture arrays\n");
                    return std::nullopt;
                }
                break;

            case D3D11_RESOURCE_DIMENSION_BUFFER:
                std::cout << "ERROR: Resource dimension buffer type not supported for textures\n";
                return std::nullopt;

            case D3D11_RESOURCE_DIMENSION_UNKNOWN:
            default:
                std::cout << "ERROR: Unknown resource dimension" << static_cast<uint32_t>(d3d10ext->resourceDimension) << std::endl;
                return std::nullopt;
            }

            resDim = d3d10ext->resourceDimension;
        }
        else
        {
            format = GetDXGIFormat(header->ddspf);

            if (format == DXGI_FORMAT_UNKNOWN)
            {
                std::cout << "ERROR: DDSTextureLoader does not support all legacy DDS formats. Consider using DirectXTex.\n";
                return std::nullopt;
            }

            if (header->flags & DDS_HEADER_FLAGS_VOLUME)
            {
                resDim = D3D11_RESOURCE_DIMENSION_TEXTURE3D;
            }
            else
            {
                if (header->caps2 & DDS_CUBEMAP)
                {
                    // We require all six faces to be defined
                    if ((header->caps2 & DDS_CUBEMAP_ALLFACES) != DDS_CUBEMAP_ALLFACES)
                    {
                        std::cout << "ERROR: DirectX 11 does not support partial cubemaps\n";
                        return std::nullopt;
                    }

                    arraySize = 6;
                    isCubeMap = true;
                }

                depth = 1;
                resDim = D3D11_RESOURCE_DIMENSION_TEXTURE2D;

                // Note there's no way for a legacy Direct3D 9 DDS to express a '1D' texture
            }

            assert(BitsPerPixel(format) != 0);
        }

        // Bound sizes (for security purposes we don't trust DDS file metadata larger than the Direct3D hardware requirements)
        if (mipCount > 16)
        {
            std::cout << "ERROR: Too many mipmap levels defined for DirectX 11 (" << mipCount << ")\n";
            return std::nullopt;
        }

        int num_dimensions = 0;

        switch (resDim)
        {
        case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
            num_dimensions = 1;

            if ((arraySize > 2048) ||
                (width > 16384))
            {
                std::cout << "ERROR: Resource dimensions too large for DirectX 11 (1D: array " << arraySize << " size " << width << std::endl;
                return std::nullopt;
            }
            break;

        case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
            num_dimensions = 2;

            if (isCubeMap)
            {
                // This is the right bound because we set arraySize to (NumCubes*6) above
                if ((arraySize > 2048) ||
                    (width > 16384) ||
                    (height > 16384))
                {
                    std::cout << "ERROR: Resource dimensions too large for DirectX 11 (2D cubemap: array " << arraySize << ", size " << width << " by " << height << " )\n";
                    return std::nullopt;
                }
            }
            else if ((arraySize > 2048) ||
                (width > 16384) ||
                (height > 16384))
            {
                std::cout << "ERROR: Resource dimensions too large for DirectX 11 (2D: array " << arraySize << ", size " << width << " by " << height << " )\n";
                return std::nullopt;
            }
            break;

        case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
            num_dimensions = 3;

            if ((arraySize > 1) ||
                (width > 2048) ||
                (height > 2048) ||
                (depth > 2048))
            {
                std::cout << "ERROR: Resource dimensions too large for DirectX 11 (3D: array " << arraySize << ", size " << width << " by " << height << " depth = " << depth << " )\n";
                return std::nullopt;
            }
            break;

        case D3D11_RESOURCE_DIMENSION_BUFFER:
            std::cout << "ERROR: Resourc]e dimension buffer type not supported for textures\n";
            return std::nullopt;

        case D3D11_RESOURCE_DIMENSION_UNKNOWN:
        default:
            std::cout << "ERROR: Unknown resource dimension " << static_cast<uint32_t>(resDim) << std::endl;
            return std::nullopt;
        }

        if (!num_dimensions)
        {
            std::cout << "ERROR: Texture dimensions unknown";
            return std::nullopt;
        }

        {
            // Create the texture
            std::vector<SUBRESOURCE_DATA> initData(mipCount * arraySize);
            size_t skipMip = 0;
            size_t twidth = 0;
            size_t theight = 0;
            size_t tdepth = 0;
            hr = FillInitData(width, height, depth, mipCount, arraySize, format, 0, bitSize, bitData, twidth, theight, tdepth, skipMip, initData.data());

            DDSSubresourceData result;
            result.subresources = std::move(initData);
            result.header = header;
			result.mip_count = (UINT)(mipCount - skipMip);
            result.width = width;
            result.height = height;
            result.depth = depth;
            result.array_size = arraySize;
            result.is_cubemap = isCubeMap;
            result.num_dimensions = num_dimensions;
            result.format = format;

            if (SUCCEEDED(hr))
                return result;
        }

        return std::nullopt;
    }

}