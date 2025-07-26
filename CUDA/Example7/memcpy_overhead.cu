#include <stdio.h>
#include <stdlib.h>


int main() {

	const int size = 1024 * 1024 * 200;
	const int BufferSize = size * sizeof(int);

	int *page_able_memory_in;
	int *page_able_memory_out;

	page_able_memory_in = (int*)malloc(BufferSize);
	page_able_memory_out = (int*)malloc(BufferSize);

	for (int i = 0; i < size; i++) {
		page_able_memory_in[i] = i;
		page_able_memory_out[i] = 0;
	}

	int *DeviceMemory;

	cudaMalloc ((void**)&DeviceMemory, BufferSize);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	cudaMemcpy(DeviceMemory, page_able_memory_in, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(page_able_memory_out, DeviceMemory, BufferSize, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsed_time;

	cudaEventElapsedTime (&elapsed_time, start, stop);

	printf("transfer time: %lf msec\n",elapsed_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	free(page_able_memory_in);
	free(page_able_memory_out);

	
	int *pinned_memory_in;
	int *pinned_memory_out;

	cudaMallocHost ((void**)&pinned_memory_in, BufferSize);
	cudaMallocHost ((void**)&pinned_memory_out, BufferSize);

	for (int i = 0; i < size; i++) {
		pinned_memory_in[i] = i;
		pinned_memory_out[i] = 0;
	}

	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);


	cudaMemcpy(DeviceMemory, pinned_memory_in, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(pinned_memory_out, DeviceMemory, BufferSize, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime (&elapsed_time, start, stop);
	printf("transfer time(pinned memory): %lf msec\n",elapsed_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFreeHost(pinned_memory_in);
	cudaFreeHost(pinned_memory_out);

	cudaFree(DeviceMemory);

	return 0;
}




