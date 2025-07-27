#include <stdio.h>
#include <stdlib.h>

// Interleaved Addressing & Divergent Branching
__global__ void reduce0 (int *g_input, int *g_output) {

	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_input[i];
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		
		if ((tid % (2*s)) == 0)
			sdata[tid] += sdata[tid + s];

		__syncthreads();
	}
	
	if (tid == 0)
		g_output[blockIdx.x] = sdata[0];

}
	

// Solved: Divergent Branching 
__global__ void reduce1 (int *g_input, int *g_output) {

	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_input[i];
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		
		int index = 2 * s * tid;

		if (index < blockDim.x)
			sdata[index] += sdata[index + s];

		__syncthreads();
	}
	
	if (tid == 0)
		g_output[blockIdx.x] = sdata[0];

}
	

// Solved: Bank Conflict (Sequential Addressing) 
__global__ void reduce2 (int *g_input, int *g_output) {

	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_input[i];
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {

		if (tid < s)
			sdata[tid] += sdata[tid + s];

		__syncthreads();
	}
	
	if (tid == 0)
		g_output[blockIdx.x] = sdata[0];

}
	

// Solved: First element's copy overhead 
__global__ void reduce3 (int *g_input, int *g_output) {

	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x*2 + threadIdx.x;

	sdata[tid] = g_input[i] + g_input[i+blockDim.x];
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {

		if (tid < s)
			sdata[tid] += sdata[tid + s];

		__syncthreads();
	}
	
	if (tid == 0)
		g_output[blockIdx.x] = sdata[0];

}


// Solved: Last warp unroll
__device__ void warpReduce (volatile int *sdata, int tid) {

	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce4 (int *g_input, int *g_output) {

	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_input[i] + g_input[i+blockDim.x];
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {

		if (tid < s)
			sdata[tid] += sdata[tid + s];

		__syncthreads();
	}

	if (tid < 32) 
		warpReduce(sdata, tid);
	
	if (tid == 0)
		g_output[blockIdx.x] = sdata[0];

}

// Solved: All warp unroll
template <unsigned int blockSize>
__device__ void allWarpReduce (volatile int *sdata, int tid) {

	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce5 (int *g_input, int *g_output) {

	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_input[i] + g_input[i+blockDim.x];
	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256)
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}

	if (tid < 32) 
		allWarpReduce<blockSize> (sdata, tid);
	
	if (tid == 0)
		g_output[blockIdx.x] = sdata[0];

}

template <unsigned int blockSize>
__global__ void reduce6 (int *g_input, int *g_output, unsigned int n) {

	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[0] = 0;
	while (i < n) {
		sdata[tid] += g_input[i] + g_input[i+blockSize];
		i += blockSize;
		//i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256)
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}

	if (tid < 32) 
		allWarpReduce<blockSize> (sdata, tid);
	
	if (tid == 0)
		g_output[blockIdx.x] = sdata[0];

}
