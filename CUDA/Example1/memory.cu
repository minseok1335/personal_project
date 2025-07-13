#include <stdio.h>

int main() {
    int InputData[5] = {1,2,3,4,5};
    int OutputData[5] = {0};

    int* GraphicsCard_Mem;

    cudaMalloc ((void**)&GraphicsCard_Mem, 5 * sizeof(int));
    cudaMemcpy (GraphicsCard_Mem, InputData, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (OutputData, GraphicsCard_Mem, 5 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i<5; i++) {
        printf("Output[%d]: %d\n", i, OutputData[i]);
    }

    cudaFree (GraphicsCard_Mem);
    return 0;
}