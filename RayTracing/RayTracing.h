#include <memory>

namespace CUDA
{
	class DeviceMemory;
}

namespace RayTracing
{
	struct Scene;

	class CUDARayTracer
	{
	public:
		struct SurfaceData
		{
			void* surface = nullptr;
			void* last_frame_surface = nullptr;
			int width = 0;
			int height = 0;
			size_t pitch = 0;
		};

		CUDARayTracer(SurfaceData surface);
		~CUDARayTracer();

		void OnResize(SurfaceData surface)
		{
			this->surface = surface;
		}

		Scene& GetScene() { return *scene; }
		void Process();

	private:
		LARGE_INTEGER frequency;
		LARGE_INTEGER last_time;

		int frame_index = 0;
		std::unique_ptr<Scene> scene;
		std::unique_ptr<CUDA::DeviceMemory> rng_state;
		SurfaceData surface;
	};

}