#include "RayTracing.h"
#include "Scene.h"
#include "utils/CUDAHelper.h"

using namespace std::chrono;

extern "C" 
{
	void raytracing_process(void* surface, void* surface_last_frame, int width, int height, size_t pitch, int frame_index, RayTracing::Scene* scene);
	void init_rng(uint32_t thread_block_count, uint32_t thread_block_size, curandState* const rngStates, const unsigned int seed);
}


namespace RayTracing
{
	CUDARayTracer::CUDARayTracer(SurfaceData surface)
		: surface(surface)
	{
		scene = std::make_unique<Scene>();

		const auto red = scene->AddMaterial(Material{ vec4(1, 0, 0, 1) });

		scene->AddSphere(vec3(0, 0, -1), 0.5f);
		scene->AddSphere(vec3(0, -100.5f, -1), 100);


		scene->AddQuad(vec3(-1, -1, -2), vec3(1, -1, -2), vec3(1, 1, -2), vec3(-1, 1, -2));

		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&last_time);
	}

	CUDARayTracer::~CUDARayTracer() = default;
	
	void CUDARayTracer::Process()
	{
		LARGE_INTEGER current_time, elapsed_time;
		QueryPerformanceCounter(&current_time);
		elapsed_time.QuadPart = current_time.QuadPart - last_time.QuadPart;
		last_time = current_time;
		double dt = (double)elapsed_time.QuadPart / (double)frequency.QuadPart;

		if (!rng_state)
		{
			const auto pixel_count = surface.width * surface.height;
			const uint32_t thread_block_size = 128;
			const uint32_t thread_block_count = (pixel_count + thread_block_size - 1) / thread_block_size;
			rng_state = std::make_unique<CUDA::DeviceMemory>(sizeof(curandState) * thread_block_count * thread_block_size); 
			const uint32_t ms = (uint32_t)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
			init_rng(thread_block_count, thread_block_size, static_cast<curandState*>(rng_state->GetMemory()), 0xDEADBEEFu * ms);
		}

		scene->GetCamera().SetViewportSize(vec2(surface.width, surface.height));
		scene->Update(static_cast<float>(dt));

		if (scene->GetDirty())
		{
			frame_index = 0;
		}

		scene->Upload(static_cast<curandState*>(rng_state->GetMemory()));
		raytracing_process(surface.surface, surface.last_frame_surface, surface.width, surface.height, surface.pitch, frame_index, scene.get());
		CUDA_CHECK(cudaMemcpy(surface.last_frame_surface, surface.surface, surface.pitch * surface.height, cudaMemcpyDeviceToDevice));
		frame_index++;
	}
}