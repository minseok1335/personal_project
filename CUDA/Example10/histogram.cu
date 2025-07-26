#include <stdio.h>
#include <stdlib.h>

__global__ void HistogramGlobal (unsigned char *buffer, int *histogram, int buffer_size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	for (; index < buffer_size; index += offset)
		atomicAdd (&(histogram[buffer[index]]), 1);
}

__global__ void HistogramShared (unsigned char *buffer, int *histogram, int buffer_size) {

	__shared__ int sHisto[256];
	if (threadIdx.x < 256)
		sHisto[threadIdx.x] = 0;
	__syncthreads();

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	for (; index < buffer_size; index += offset)
		atomicAdd (&(sHisto[buffer[index]]), 1);
	__syncthreads();

	if(threadIdx.x < 256)
		atomicAdd(&(histogram[threadIdx.x]), sHisto[threadIdx.x]);
}

int main () {

	const int nblocks = 512;
	const int nthreads = 512;
	const int size = 1024 * 1024;

	unsigned char *host_buffer;
	int *host_histogram;

	cudaMallocHost((void**)&host_buffer, size);
	cudaMallocHost((void**)&host_histogram, 256 * sizeof(int));

	for (int i = 0; i < size; i++)
		host_buffer[i] = i % 256;

	unsigned char* dev_buffer;
	int *dev_histogram;
	
	cudaMalloc ((void**)&dev_buffer, size);
	cudaMalloc ((void**)&dev_histogram, 256*sizeof(int));

	cudaMemset (dev_buffer, 0, size);
	cudaMemset (dev_histogram, 0, 256 * sizeof(int));

	cudaMemcpy (dev_buffer, host_buffer, size, cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	cudaEventRecord (start, 0);
	HistogramGlobal <<<nblocks, nthreads>>> (dev_buffer, dev_histogram, size);
	cudaEventRecord (stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy (host_histogram, dev_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i =0; i < 10; i++)
			printf("Histogram[%d]: %d\n", i, host_histogram[i]);

	printf("...................\n");

	for (int i = 250; i < 256; i++)
		printf("Histogram[%d]: %d\n", i, host_histogram[i]);
	
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Elapsed Time (global): %f\n", elapsed_time);

	cudaEventDestroy (start);
	cudaEventDestroy (stop);


	memset (host_histogram, 0, 256 * sizeof(int));

	cudaMemset (dev_buffer, 0, size);
	cudaMemset (dev_histogram, 0, 256 * sizeof(int));

	cudaMemcpy (dev_buffer, host_buffer, size, cudaMemcpyHostToDevice);
			
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	cudaEventRecord (start, 0);
	HistogramShared <<<nblocks, nthreads>>> (dev_buffer, dev_histogram, size);
	cudaEventRecord (stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy (host_histogram, dev_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i =0; i < 10; i++)
			printf("Histogram[%d]: %d\n", i, host_histogram[i]);

	printf("...................\n");

	for (int i = 250; i < 256; i++)
		printf("Histogram[%d]: %d\n", i, host_histogram[i]);

	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Elapsed Time (shared): %f\n", elapsed_time);

	cudaEventDestroy (start);
	cudaEventDestroy (stop);
	
	cudaFreeHost(host_buffer);
	cudaFreeHost(host_histogram);

	cudaFree(dev_buffer);
	cudaFree(dev_histogram);

	return 0;
}
