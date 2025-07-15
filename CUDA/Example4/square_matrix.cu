#include <stdio.h>
#include <stdlib.h>


void MatrixMulC (int *M, int *N, int *P, int width) {
	int col = 0;
	int row = 0;
	int index = 0;
	int Destindex = 0;
	for (col = 0; col < width; col++) {
		for (row = 0; row < width; row++) {
			Destindex = col * width + row;
			for (index = 0; index < width; index++) {
				P[Destindex] += M[col *width + index] * N[index*width + row];
			}
		}
	}
}


__global__ void MatrixMul (int *M, int *N, int *P, int width) {
	int tid, tx, ty;

	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;
	tid = width * ty + tx;

	int Value = 0;
	int Mval = 0;
	int Nval = 0;

	for (int i = 0; i < width; i++) {
		Mval = M[ty * width + i];
		Nval = N[i * width + tx];
		Value += Mval * Nval;
	}

	P[tid] = Value;
}

int main() {
	const int MatrixWidth = 12;
	const int MatrixHeight = 12;
	const int MatrixSize = MatrixWidth * MatrixHeight;
	const int BufferSize = MatrixSize * sizeof(int);

	int *M;
	int *N;
	int *P_cuda;
	int *P_C;

	M = (int*)malloc(BufferSize);
	N = (int*)malloc(BufferSize);
	P_cuda = (int*)malloc(BufferSize);
	P_C = (int*)malloc(BufferSize);

	for (int i = 0; i < MatrixSize; i++) {
		
		M[i] = i;
		N[i] = i;
		P_cuda[i] = 0;
		P_C[i] = 0;

	}

	int *dev_M;
	int *dev_N;
	int *dev_P;

	cudaMalloc ((void**)&dev_M, BufferSize);
	cudaMalloc ((void**)&dev_N, BufferSize);
	cudaMalloc ((void**)&dev_P, BufferSize);
	
	cudaMemcpy (dev_M, M, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy (dev_N, N, BufferSize, cudaMemcpyHostToDevice);

	dim3 Dg(3, 4, 1);
	dim3 Db(4, 3, 1);

	MatrixMul <<<Dg, Db>>> (dev_M, dev_N, dev_P, 12);

	cudaMemcpy (P_cuda, dev_P, BufferSize, cudaMemcpyDeviceToHost);

	MatrixMulC (M, N, P_C, 12);

	bool ResultFlag = true;

	for (int i = 0; i < MatrixSize; i++) {
		if (P_cuda[i] != P_C[i]) ResultFlag = false;
	}

	if (ResultFlag)
		printf("MatrixMul Result OK\n");
	else
		printf("MatrixMul Result Error\n");

	cudaFree (dev_M);
	cudaFree (dev_N);
	cudaFree (dev_P);

	free (M);
	free (N);
	free (P_cuda);
	free (P_C);

	return 0;
}

