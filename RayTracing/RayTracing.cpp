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

		SetupCornellBox();

		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&last_time);
	}

	CUDARayTracer::~CUDARayTracer() = default;

	void CUDARayTracer::SetupCornellBox()
	{
		// Back wall
		{
			vec3 A = vec3(-12.6f, -12.6f, 25.0f);
			vec3 B = vec3(12.6f, -12.6f, 25.0f);
			vec3 C = vec3(12.6f, 12.6f, 25.0f);
			vec3 D = vec3(-12.6f, 12.6f, 25.0f);
			auto material = scene->AddMaterial({ vec3(0.7f, 0.7f, 0.7f), vec3(0.0f, 0.0f, 0.0f) });
			scene->AddQuad(A, B, C, D, material);
		}

        // floor
        {
            vec3 A = vec3(-12.6f, -12.45f, 25.0f);
            vec3 B = vec3(12.6f, -12.45f, 25.0f);
            vec3 C = vec3(12.6f, -12.45f, 15.0f);
            vec3 D = vec3(-12.6f, -12.45f, 15.0f);
			auto material = scene->AddMaterial({ vec3(0.7f, 0.7f, 0.7f), vec3(0.0f, 0.0f, 0.0f) });
            scene->AddQuad(A, B, C, D, material);
        }

        // cieling
        {
            vec3 A = vec3(-12.6f, 12.5f, 25.0f);
            vec3 B = vec3(12.6f, 12.5f, 25.0f);
            vec3 C = vec3(12.6f, 12.5f, 15.0f);
            vec3 D = vec3(-12.6f, 12.5f, 15.0f);
			auto material = scene->AddMaterial({ vec3(0.7f, 0.7f, 0.7f), vec3(0.0f, 0.0f, 0.0f) });
            scene->AddQuad(A, B, C, D, material);
        }

        // left wall
        {
            vec3 A = vec3(-12.5f, -12.6f, 25.0f);
            vec3 B = vec3(-12.5f, -12.6f, 15.0f);
            vec3 C = vec3(-12.5f, 12.6f, 15.0f);
            vec3 D = vec3(-12.5f, 12.6f, 25.0f);
			auto material = scene->AddMaterial({ vec3(0.1f, 0.7f, 0.1f), vec3(0.0f, 0.0f, 0.0f) });
            scene->AddQuad(A, B, C, D, material);
        }

        // right wall 
        {
            vec3 A = vec3(12.5f, -12.6f, 25.0f);
            vec3 B = vec3(12.5f, -12.6f, 15.0f);
            vec3 C = vec3(12.5f, 12.6f, 15.0f);
            vec3 D = vec3(12.5f, 12.6f, 25.0f);
			auto material = scene->AddMaterial({ vec3(0.7f, 0.1f, 0.1f), vec3(0.0f, 0.0f, 0.0f) });
            scene->AddQuad(A, B, C, D, material);
        }

        // light
        {
            vec3 A = vec3(-5.0f, 12.4f, 22.5f);
            vec3 B = vec3(5.0f, 12.4f, 22.5f);
            vec3 C = vec3(5.0f, 12.4f, 17.5f);
            vec3 D = vec3(-5.0f, 12.4f, 17.5f);
			auto material = scene->AddMaterial({ vec3(0), vec3(1.0f, 0.9f, 0.7f) * 20.0f });
            scene->AddQuad(A, B, C, D, material);
        }

		scene->AddSphere(vec3(-9.0f, -9.5f, 20.0f), 3, scene->AddMaterial(Material(vec3(0.9f, 0.9f, 0.50f))));
		scene->AddSphere(vec3(0.0f, -9.5f, 20.0f), 3, scene->AddMaterial(Material(vec3(0.9f, 0.50f, 0.9f))));
		scene->AddSphere(vec3(9.0f, -9.5f, 20.0f), 3, scene->AddMaterial(Material(vec3(0.50f, 0.9f, 0.9f))));
		scene->GetCamera().SetYAndle(180);
	}
	
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