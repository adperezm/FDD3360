#include <stdio.h>
#define N 1
#define TPB 256



__global__ void printHelloWorld()
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	__syncthreads();
	printf("Hello world! My threadId is  %2d \n",i);
}

int main()
{
  printHelloWorld<<<N, TPB>>>();
  cudaDeviceSynchronize();
  return 0;
}
