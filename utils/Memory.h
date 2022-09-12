#pragma once

namespace Memory
{
    inline uint32_t NextPowerOfTwo(uint32_t n)
    {
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }

    inline size_t AlignSize(size_t ptr, size_t alignment)
    {
        return (((ptr)+((alignment)-1)) & ~((alignment)-1));
    }
}
