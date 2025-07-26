#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void vector_add (int *a, int *b, int *c) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	c[idx] = a[idx] * b[idx];

}

int main() {

	int i = 0;
	int nBlocks = 1024;
	int nThreads = 512;
	int size = nBlocks * nThreads;
	size_t BufferSize = size * sizeof(int);

	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	cudaHostAlloc ((void**)&h_a, BufferSize, cudaHostAllocMapped);
	cudaHostAlloc ((void**)&h_b, BufferSize, cudaHostAllocMapped);
	cudaHostAlloc ((void**)&h_c, BufferSize, cudaHostAllocMapped);

	for (i = 0; i < size; i++) {
		h_a[i] = i;
		h_b[i] = i;
	}

	cudaHostGetDevicePointer ((void**) &d_a, (void*)h_a, 0);
	cudaHostGetDevicePointer ((void**) &d_b, (void*)h_b, 0);
	cudaHostGetDevicePointer ((void**) &d_c, (void*)h_c, 0);

	vector_add <<<nBlocks, nThreads>>> (d_a, d_b, d_c);
	cudaThreadSynchronize();

	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);

	return 0;

}
