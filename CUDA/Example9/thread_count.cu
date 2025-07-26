#include <stdio.h>
#include <stdlib.h>

__global__ void ThreadRace (int *count) {
	(*count)++;
}

__global__ void ThreadAtomic (int *count) {
	atomicAdd(count, 1);
}

int main () {
	const int nBlocks = 10000;
	const int nThreads = 512;

	int count = 0;
	int *dev_count;

	cudaMalloc ((void**)&dev_count, sizeof(int));
	cudaMemset (dev_count, 0, sizeof(int));

	ThreadRace <<<nBlocks, nThreads>>> (dev_count);

	cudaMemcpy (&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);

	printf("race count: %d\n", count);


	count = 0;
	cudaMemset (dev_count, 0, sizeof(int));

	ThreadAtomic <<<nBlocks, nThreads>>> (dev_count);

	cudaMemcpy (&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);

	printf("atomic count: %d\n", count);

	cudaFree(dev_count);
	return 0;
}
