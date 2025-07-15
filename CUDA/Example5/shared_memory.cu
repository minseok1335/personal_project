#include <stdio.h>
#include <stdlib.h>

#define MAX_SHARED_SIZE 2048

__global__ void LoadStoreSharedMemory(int *in, int *out) {

	int LoadStoreSize = MAX_SHARED_SIZE/blockDim.x;

	int begin = LoadStoreSize * threadIdx.x;
	int end = begin + LoadStoreSize;

	__shared__ int SharedMemory[MAX_SHARED_SIZE];

	for (int i = begin; i < end; i++)
		SharedMemory[i] = in[i];

	__syncthreads();

	for (int i = begin; i < end; i++)
		out[i] = SharedMemory[i];

	__syncthreads();

}

int main () {
	
	const int size = MAX_SHARED_SIZE;
	const int BufferSize = size * sizeof(int);

	int *input;
	int *output;

	input = (int*)malloc(BufferSize);
	output = (int*)malloc(BufferSize);

	for (int i = 0; i < size; i++) {
		input[i] = i;
		output[i] = 0;
	}

	int *dev_in;
	int *dev_out;

	cudaMalloc ((void**)&dev_in, size*sizeof(int));
	cudaMalloc ((void**)&dev_out, size*sizeof(int));

	cudaMemcpy (dev_in, input, size * sizeof(int), cudaMemcpyHostToDevice);
	
	LoadStoreSharedMemory <<<32, 512>>> (dev_in, dev_out);

	cudaMemcpy (output, dev_out, size * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 5; i++)
		printf("Output[%d]: %d\n", i, output[i]);

	printf("......\n");

	for (int i = size - 5; i < size; i++)
		printf("Output[%d]: %d\n", i, output[i]);

	cudaFree(dev_in);
	cudaFree(dev_out);

	free(input);
	free(output);

	return 0;
}
