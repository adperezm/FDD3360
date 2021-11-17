#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand.h>
#include <sys/time.h>

#define SINGLE_PRECISION


double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void gpu_random(float *res,curandState *states,unsigned int blockSize,unsigned long long numIter,unsigned int n_blocks) {

  int id = blockIdx.x*blockDim.x+threadIdx.x;
#ifdef SINGLE_PRECISION
  float x, y, z;
#else
  double x, y, z;
#endif
  float count=0;

  int seed=id; //different seed per thread
  curand_init(seed,id,0,&states[id]); //initialize curand

  for (int i=0;i < (numIter/(n_blocks*blockSize)+1); i++){ //+1 because might give 0 that division
    //Generate a random point
#ifdef SINGLE_PRECISION
    x=curand_uniform(&states[id]);
    y=curand_uniform(&states[id]);
#else
    x=curand_uniform_double(&states[id]);
    y=curand_uniform_double(&states[id]);
#endif
    // Check if point is in unit circle
    z = sqrt((x*x) + (y*y));
    if (z <= 1.0){
      count=count+1;
    }
  }
  //Store the values
  res[id]=count;

}



int main(int argc, char* argv[])
{
    float count=0;
    double pi;
    unsigned long long numIter;
    unsigned int blockSize= atoi(argv[1]);
    unsigned int n_blocks= 1000000/blockSize; // So that there are always a million threads
    //Define an array of size (num of threads) to hold the results in GPU
    float *d_x=0;
    scanf("%llu", &numIter);
    numIter=(((unsigned long long)1000000)*numIter);
    cudaMalloc(&d_x,n_blocks*blockSize*sizeof(float));
    printf("Num iter %llu, Block size: %d, N blocks: %d\n", numIter, blockSize, n_blocks);
    // Hold the results in CPU
    float *cpu_x= (float *)malloc(n_blocks*blockSize*sizeof(float));



    //Initialize randon numbers in GPU
    curandState *dev_random;
    cudaMalloc((void**)&dev_random,n_blocks*blockSize*sizeof(curandState));

    double iStart=cpuSecond();
    //run the kernel
    gpu_random <<<n_blocks,blockSize>>> (d_x,dev_random,blockSize,numIter,n_blocks);
    cudaDeviceSynchronize();
 
    //copy back into CPU
    cudaMemcpy(cpu_x,d_x,n_blocks*blockSize*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_x);

    for (int i=0;i< n_blocks*blockSize; i++){
    count = count + cpu_x[i];
    }
    
    // Estimate Pi and display the result
    pi = ((double)count / (double)numIter) * 4.0;
    
    double iElaps=cpuSecond() - iStart;

    //for (int i=0;i< NB*TPB; i++){
    //  printf("x = %f\n", cpu_x[i]);
    //}


    printf("The result is %f\n", pi);
    printf("time elapsed =  %f seconds\n", iElaps);


    return 0;
}


