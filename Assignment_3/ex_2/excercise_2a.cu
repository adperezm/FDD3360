#include <stdio.h>
#include <stdlib.h>

// Question 1 defines:
#define FIXED_DT 1
#define BLOCK_SIZE 64
#define VEL 1
typedef float3 Particle;


//function declarations
__device__ inline float3& operator +=(float3 &a, const float3 &b);
__global__ void GPU_update(Particle *all_particles_ptr, float v,unsigned int num_particles); //Computes the update of only one update for all particles.
void particle_computation_launcher(bool use_cuda_malloc, unsigned int num_particles);
//function main
int main(int argc, char *argv[])
{
    printf("Running! \n");   
    unsigned int mode = atoi(argv[1]); //1 indicates cuda Malloc host, the other the simpler malloc.
    unsigned int num_particles = atoi(argv[2]);
    particle_computation_launcher(mode, num_particles); 

    return 0;
}

//function definitions
void particle_computation_launcher(bool use_cuda_malloc, unsigned int num_particles)
{
    float v=VEL; //Could be a random value.
    Particle *d_all_particles_ptr;
    Particle *all_particles_ptr;
    cudaMalloc(&d_all_particles_ptr, num_particles*sizeof(Particle)); 
    if (use_cuda_malloc) //Allocation method
        all_particles_ptr = (Particle *) malloc(num_particles*sizeof(Particle));
    else
    {    
        //printf("Got here"); //cudamallochost is the same as cudahostalloc in this case https://stackoverflow.com/questions/35535831/is-there-any-difference-between-cudamallochost-and-cudahostalloc-without-spe#:~:text=cudaMallocHost()%20%3A%20%22Allocates%20page%2D,locked%20memory%20on%20the%20host.%22
        if (cudaMallocHost(&all_particles_ptr, num_particles*sizeof(Particle)) != cudaSuccess)
	    {printf("Could not allocate pag memory!");}
    }
    //printf("What happened");
    for(unsigned int i=0; i< num_particles;i++)
    {
	all_particles_ptr[i].x=0;
	all_particles_ptr[i].y=0;
	all_particles_ptr[i].z=0;
    }
    cudaMemcpy(d_all_particles_ptr,all_particles_ptr,num_particles*sizeof(Particle),cudaMemcpyHostToDevice);
    GPU_update<<<num_particles/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_all_particles_ptr,v,num_particles);
    cudaDeviceSynchronize();
    cudaMemcpy(all_particles_ptr,d_all_particles_ptr,num_particles*sizeof(Particle),cudaMemcpyDeviceToHost);
    cudaFree(d_all_particles_ptr);
    if (use_cuda_malloc)
        free(all_particles_ptr);
    else
        cudaFreeHost(all_particles_ptr);
}


__global__ void GPU_update(Particle *all_particles_ptr, float v,unsigned int num_particles)
{
        const int i = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int curr_particle=i; //Check boundaries
        if(i<num_particles)
        {
            float update_value=v*FIXED_DT;
            float3 update = make_float3(update_value, update_value, update_value);
            all_particles_ptr[curr_particle]+=update;
        }
}
__device__ inline float3& operator +=(float3 &a, const float3 &b) { //https://forums.developer.nvidia.com/t/operator-overloading-for-float4/27228

  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
