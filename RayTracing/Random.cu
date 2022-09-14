#include "Random.h"

__global__ void initRNG(curandState *const rngStates, const unsigned int seed) {
  // Determine thread ID
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Initialise the RNG
  curand_init(seed, tid, 0, &rngStates[tid]);
}

extern "C" void init_rng(uint32_t thread_block_count, uint32_t thread_block_size, curandState* const rngStates, const unsigned int seed)
{
	initRNG<<<dim3(thread_block_count), dim3(thread_block_size)>>>(rngStates, seed);
}
