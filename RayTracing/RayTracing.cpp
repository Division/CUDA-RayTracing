#include "RayTracing.h"
#include "Scene.h"
#include "utils/CUDAHelper.h"

using namespace std::chrono;

extern "C" 
{
	void raytracing_process(void* surface, void* surface_last_frame, int width, int height, size_t pitch, int frame_index, RayTracing::Scene* scene, cudaTextureObject_t t);
	void init_rng(uint32_t thread_block_count, uint32_t thread_block_size, curandState* const rngStates, const unsigned int seed, cudaTextureObject_t t);
}


namespace RayTracing
{
	CUDARayTracer::CUDARayTracer(SurfaceData surface)
		: surface(surface)
	{
		scene = std::make_unique<Scene>();
		scene->AddSphere(vec3(0, 0, -1), 0.5f);
		scene->AddSphere(vec3(0, -100.5f, -1), 100);
	}

	CUDARayTracer::~CUDARayTracer() = default;
	
	void CUDARayTracer::Process()
	{
		if (!rng_state)
		{
			const auto pixel_count = surface.width * surface.height;
			const uint32_t thread_block_size = 128;
			const uint32_t thread_block_count = (pixel_count + thread_block_size - 1) / thread_block_size;
			rng_state = std::make_unique<CUDA::DeviceMemory>(sizeof(curandState) * thread_block_count * thread_block_size); 
			const uint32_t ms = (uint32_t)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
			init_rng(thread_block_count, thread_block_size, static_cast<curandState*>(rng_state->GetMemory()), 0xDEADBEEFu * ms, scene->environment_cubemap_tex);
		}

		scene->Upload(static_cast<curandState*>(rng_state->GetMemory()));
		raytracing_process(surface.surface, surface.last_frame_surface, surface.width, surface.height, surface.pitch, frame_index, scene.get(), scene->environment_cubemap_tex);
		CUDA_CHECK(cudaMemcpy(surface.last_frame_surface, surface.surface, surface.pitch * surface.height, cudaMemcpyDeviceToDevice));
		frame_index++;
	}
}