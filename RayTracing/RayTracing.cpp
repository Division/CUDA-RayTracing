module RayTracing;

extern "C" 
{
	void raytracing_process(void* surface, int width, int height, size_t pitch);
}


namespace RayTracing
{
	void CUDARayTracer::Process()
	{
		raytracing_process(surface.surface, surface.width, surface.height, surface.pitch);
	}
}