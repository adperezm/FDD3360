#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand.h>
#include <sys/time.h>

#define NUM_ITER 1000000000
#define TPB 128
#define NB 1

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void gpu_random(float *res,curandState *states) {

  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double x, y, z;
  float count=0;

  int seed=id; //different seed per thread
  curand_init(seed,id,0,&states[id]); //initialize curand

  for (int i=0;i < NUM_ITER/(NB*TPB); i++){
    //Generate a random point
    x=curand_uniform_double(&states[id]);
    y=curand_uniform_double(&states[id]);
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

    //Define an array of size (num of threads) to hold the results in GPU
    float *d_x=0;
    cudaMalloc(&d_x,NB*TPB*sizeof(float));

    // Hold the results in CPU
    float *cpu_x= (float *)malloc(NB*TPB*sizeof(float));

    //Initialize randon numbers in GPU
    curandState *dev_random;
    cudaMalloc((void**)&dev_random,NB*TPB*sizeof(curandState));

    double iStart=cpuSecond();
    //run the kernel
    gpu_random <<<NB,TPB>>> (d_x,dev_random);
    cudaDeviceSynchronize();
 
    //copy back into CPU
    cudaMemcpy(cpu_x,d_x,NB*TPB*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_x);

    for (int i=0;i< NB*TPB; i++){
    count = count + cpu_x[i];
    }
    
    // Estimate Pi and display the result
    pi = ((double)count / (double)NUM_ITER) * 4.0;
    
    double iElaps=cpuSecond() - iStart;

    //for (int i=0;i< NB*TPB; i++){
    //  printf("x = %f\n", cpu_x[i]);
    //}


    printf("The result is %f\n", pi);
    printf("time elapsed =  %f seconds\n", iElaps);


    return 0;
}


