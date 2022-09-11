#pragma once

#include "DDS.h"
#include "DDSLoaderHelpers.h"
#include <optional>

namespace DirectX 
{
    struct SUBRESOURCE_DATA
    {
        size_t width;
        size_t height;
        size_t depth;
        const void* pSysMem;
        UINT SysMemPitch;
        UINT SysMemSlicePitch;
    };

    struct DDSSubresourceData
    {
        std::vector<SUBRESOURCE_DATA> subresources;
        const DDS_HEADER* header;
        UINT width;
        UINT height;
        UINT depth;
        UINT mip_count;
        UINT array_size;
        UINT num_dimensions;
        DXGI_FORMAT format;
        bool is_cubemap;
    };

    std::optional<DDSSubresourceData> GetDDSSubresources(
        _In_ const DDS_HEADER* header,
        _In_reads_bytes_(bitSize) const uint8_t* bitData,
        _In_ size_t bitSize
    );
}