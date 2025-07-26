#include <stdio.h>
#include <stdlib.h>

__global__ void kernel (int *in, int *out) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < 5; i++)
		out[tid] += in[tid];
}

int main () {

	const int nStreams = 15;
	const int nBlocks = 65535;
	const int nThreads = 512;
	const int N = 512 * 65535;
	const int size = N * sizeof(int);


	int *h_in;
	int *h_out;

	cudaMallocHost ((void**)&h_in, size);
	cudaMallocHost ((void**)&h_out, size);

	for (int i = 0; i < N; i++) {
		h_in[i] = i;
		h_out[i] = 0;
	}

	int *d_in;
	int *d_out;

	cudaMalloc ((void**)&d_in, size);
	cudaMalloc ((void**)&d_out, size);

	cudaMemset(d_in, 0, size);
	cudaMemset(d_out, 0, size);

	cudaEvent_t SyncStart, SyncStop;

	float SyncTime;

	cudaEventCreate(&SyncStart);
	cudaEventCreate(&SyncStop);

	cudaEventRecord(SyncStart, 0);

	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

	kernel <<<nBlocks, nThreads>>> (d_in, d_out);

	cudaMemcpy(d_out, h_out, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(SyncStop, 0);
	cudaEventSynchronize(SyncStop);

	cudaEventElapsedTime(&SyncTime, SyncStart, SyncStop);

	printf("synchronization version: %f msec\n", SyncTime);

	for (int i = 0; i < N; i++) {
		h_in[i] = i;
		h_out[i] = 0;
	}

	cudaMemset(d_in, 0, size);
	cudaMemset(d_out, 0, size);

	cudaStream_t *streams = (cudaStream_t*)malloc(nStreams * sizeof(cudaStream_t));

	for (int i = 0; i < nStreams; i++) 
		cudaStreamCreate(&(streams[i]));
	
	cudaEvent_t StreamStart, StreamStop;
	float StreamTime;

	cudaEventCreate(&StreamStart);
	cudaEventCreate(&StreamStop);

	int offset = 0;
	int chunck_size = size / nStreams;

	cudaEventRecord(StreamStart, 0);

	for (int i = 0; i < nStreams; i++) {
		offset = i * N / nStreams;
		cudaMemcpyAsync(d_in + offset, h_in + offset, chunck_size, cudaMemcpyHostToDevice, streams[i]);
	}

	for (int i = 0; i < nStreams; i++) {
		offset = i * N / nStreams;
		kernel <<<nBlocks/nStreams, nThreads, 0, streams[i]>>> (d_in + offset, d_out + offset);
	}

	for (int i = 0; i < nStreams; i++) {
		offset = i * N / nStreams;
		cudaMemcpyAsync(h_in + offset, d_out + offset, chunck_size, cudaMemcpyDeviceToHost, streams[i]);
	}

	cudaEventRecord(StreamStop, 0);
	cudaEventSynchronize(StreamStop);

	cudaEventElapsedTime (&StreamTime, StreamStart, StreamStop);

	printf("stream version: %f\n", StreamTime);

	cudaEventDestroy(SyncStart);
	cudaEventDestroy(SyncStop);
	cudaEventDestroy(StreamStart);
	cudaEventDestroy(StreamStop);

	for (int i = 0; i < nStreams; i++)
		cudaStreamDestroy(streams[i]);

	cudaFree (d_in);
	cudaFree (d_out);

	cudaFreeHost (h_in);
	cudaFreeHost (h_out);

	return 0;
}
