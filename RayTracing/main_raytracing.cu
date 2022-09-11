#include "Math.h"
#include "GPUScene.h"

using namespace Math;
using namespace RayTracing;

#define PI 3.1415926536f

typedef vec4 RGBA;

struct RenderData
{
    unsigned char* surface;
    unsigned char* surface_last_frame;
    int width; 
    int height;
    size_t pitch;
    int frame_index;
    Camera camera;
    GPUScene scene;
    cudaTextureObject_t env_tex;
};

struct HitData
{
    vec3 position;
    vec3 normal;
    float distance;
};

__device__ bool GetRayHit(const Ray& ray, GPUScene& scene, HitData& result)
{
    const int sphere_count = scene.GetSphereCount();
    const auto normalized_ray_direction = glm::normalize(ray.direction);
    float lowest_distance = INFINITY;
    for (int i = 0; i < sphere_count; i++)
    {
        auto sphere = scene.GetSphereData(i);
        float distance;
        if (glm::intersectRaySphere(ray.origin, normalized_ray_direction, sphere->position, sphere->radius * sphere->radius, distance))
        {
            if (distance >= lowest_distance) continue;
            result.distance = distance;
            result.position = ray.origin + normalized_ray_direction * distance;
            result.normal = (result.position - sphere->position) / sphere->radius;
            lowest_distance = distance;
        }
    }

    return lowest_distance < INFINITY;
}

__device__ RGBA ray_color(const Ray& r, GPUScene& scene, RNG rng, int depth) 
{
    if (depth <= 0)
        return RGBA(0);

    HitData hit;
    if (GetRayHit(r, scene, hit))
    {
        vec3 target = hit.position + hit.normal + rng.GetRandomPointOnSphere();
        Ray new_ray(hit.position, target);
        return 0.5f * ray_color(new_ray, scene, rng, depth - 1);
        //return RGBA((hit.normal + 1.0f) * 0.5f, 1.0f);
    }

    float4 cubemap = texCubemapLod<float4>(scene.environment_cubemap_tex, r.direction.x, r.direction.y, r.direction.z, 0);
    return RGBA(cubemap.x, cubemap.y, cubemap.z, 1.0f);

    vec3 unit_direction = glm::normalize(r.direction);
    auto t = 0.5f*(unit_direction.y + 1.0f);
    
    return glm::lerp(RGBA(1.0f, 1.0f, 1.0f, 1.0f), RGBA(0.5f, 0.7f, 1.0f, 1.0f), t);
}

__global__ void raytracing_kernel_main(RenderData render_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr int sample_count = 1;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= render_data.width || y >= render_data.height) return;

    //float4 cubemap = texCubemapLod<float4>(t, -1.0f, 0.0f, 0.0f, 1);
    // get a pointer to the pixel at (x,y)
    RGBA* pixel = (RGBA*)(render_data.surface + y * render_data.pitch) + x;
    RGBA* pixel_last_frame = (RGBA*)(render_data.surface_last_frame + y * render_data.pitch) + x;

    // populate it
    const vec2 pixel_coords(x, y);

    RNG rng = render_data.scene.GetRNGForPixel(render_data.camera.viewport_size.x, x, y);
    vec4 result_color(0);
    for (int i = 0; i < sample_count; i++)
    {
		const auto uv = (pixel_coords + vec2(rng.GetFloat01(), rng.GetFloat01())) / vec2(render_data.width - 1.0f, render_data.height - 1.0f);
        Ray ray = render_data.camera.GetRay(uv);
        result_color += ray_color(ray, render_data.scene, rng, 2);
	}

    result_color /= (float)sample_count;
    float lerp_value = render_data.frame_index > 0 ? 1.0f / (float)(render_data.frame_index + 1) : 1.0f;
    *pixel = glm::lerp(*pixel_last_frame, result_color, lerp_value);
    pixel->a = 1.0f;
}

extern "C" void raytracing_process(void* surface, void* surface_last_frame, int width, int height, size_t pitch, int frame_index, GPUScene* scene) {
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

    Camera camera(vec3(0), (float)width / (float)height);
    RenderData render_data{
        static_cast<unsigned char*>(surface), static_cast<unsigned char*>(surface_last_frame), width, height, pitch, 
        frame_index, camera, *scene
    };

    raytracing_kernel_main<<<Dg, Db>>>(render_data);

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("(raytracing_kernel_main) failed to launch error = %d\n", error);
    }
}

