#include <stdio.h>


__global__ void KernelFunction(int a, int b, int c) {
    int sum = a+b+c;
}

int main() {
    KernelFunction <<<6, 6>>> (1,2,3);

    printf("Complete\n");
    return 0;
}