#include <stdio.h>
#include <stdlib.h>

__global__ void CountAtomic (int *count) {
	atomicAdd(count, 1);
}

__global__ void CountShared (int *count) {

	__shared__ int nCount;
	
	if(threadIdx.x == 0)
		nCount = 0;
	__syncthreads();

	atomicAdd(&nCount, 1);
	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd (count, nCount);
}

int main() {
	const int nBlocks = 10000;
	const int nThreads = 512;

	int host_nThreadCount = 0;
	int *dev_nThreadCount;

	cudaMalloc((void**)&dev_nThreadCount, sizeof(int));
	cudaMemset(dev_nThreadCount, 0, sizeof(int));

	cudaEvent_t start, stop;
	float Elapsed_time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	CountAtomic <<<nBlocks, nThreads>>> (dev_nThreadCount);

	cudaMemcpy(&host_nThreadCount, dev_nThreadCount, sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&Elapsed_time, start, stop);

	printf("global memory\n threads: %d\n elapsed time: %f\n", host_nThreadCount, Elapsed_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	host_nThreadCount = 0;
	cudaMemset(dev_nThreadCount, 0, sizeof(int));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	CountShared <<<nBlocks, nThreads>>> (dev_nThreadCount);

	cudaMemcpy(&host_nThreadCount, dev_nThreadCount, sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&Elapsed_time, start, stop);

	printf("shared memory\n threads: %d\n elapsed time: %f\n", host_nThreadCount, Elapsed_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dev_nThreadCount);
	return 0;
}
