#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Scene.h"
#include "utils/Math.h"

using namespace Math;
using namespace RayTracing;

#define PI 3.1415926536f

struct RGBA
{
    float r, g, b, a;
};

struct RenderData
{
    unsigned char* surface;
    int width; 
    int height;
    size_t pitch;
    Camera camera;
};

/*
 * Paint a 2D texture with a moving red/green hatch pattern on a
 * strobing blue background.  Note that this kernel reads to and
 * writes from the texture, hence why this texture was not mapped
 * as WriteDiscard.
 */
__global__ void raytracing_kernel_main(RenderData render_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    RGBA* pixel;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= render_data.width || y >= render_data.height) return;

    // get a pointer to the pixel at (x,y)
    pixel = (RGBA*)(render_data.surface + y * render_data.pitch) + x;

    // populate it
    const vec2 pixel_coords(x, y);
    vec4 result_color(pixel_coords, 0, 0);
    result_color /= vec4(render_data.width - 1.0f, render_data.height - 1.0f, 1, 1);
    result_color.a = 1;

    pixel->r = result_color.r;  // red
    pixel->g = result_color.g;  // green
    pixel->b = result_color.b;  // blue
    pixel->a = result_color.a;  // alpha
}

extern "C" void raytracing_process(void* surface, int width, int height, size_t pitch) {
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

    Camera camera(vec3(0), (float)width / (float)height);
    RenderData render_data{ static_cast<unsigned char*>(surface), width, height, pitch, camera };

    raytracing_kernel_main<<<Dg, Db>>>(render_data);

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
    }
}
