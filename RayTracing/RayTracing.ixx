export module RayTracing;

export namespace RayTracing
{

	class CUDARayTracer
	{
	public:
		struct SurfaceData
		{
			void* surface = nullptr;
			int width = 0;
			int height = 0;
			size_t pitch = 0;
		};

		CUDARayTracer(SurfaceData surface)
			: surface(surface)
		{

		}

		void OnResize(SurfaceData surface)
		{
			this->surface = surface;
		}

		void Process();

		SurfaceData surface;
	};

}