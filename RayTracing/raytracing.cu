#include "RayTracing.h"
#include "Scene.h"
#include <curand_kernel.h>
#include <chrono>
#include "utils/CUDAHelper.h"

using namespace std::chrono;

extern "C" 
{
	void raytracing_process(void* surface, void* surface_last_frame, int width, int height, size_t pitch, int frame_index, RayTracing::Scene* scene);
}


namespace RayTracing
{

	__global__ void initRNG(curandState *const rngStates, const unsigned int seed) {
	  // Determine thread ID
	  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	  // Initialise the RNG
	  curand_init(seed, tid, 0, &rngStates[tid]);
	}

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
			const dim3 thread_block_size(128);
			const dim3 thread_block_count((pixel_count + thread_block_size.x - 1) / thread_block_size.x);
			rng_state = std::make_unique<CUDAHelper::DeviceMemory>(sizeof(curandState) * thread_block_count.x * thread_block_size.x); 
			const uint32_t ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
			initRNG<<<thread_block_count, thread_block_size>>>(static_cast<curandState*>(rng_state->memory), 0xDEADBEEFu * ms);
		}

		scene->Upload(static_cast<curandState*>(rng_state->memory));
		raytracing_process(surface.surface, surface.last_frame_surface, surface.width, surface.height, surface.pitch, frame_index, scene.get());
		CUDA_CHECK(cudaMemcpy(surface.last_frame_surface, surface.surface, surface.pitch * surface.height, cudaMemcpyDeviceToDevice));
		frame_index++;
	}
}