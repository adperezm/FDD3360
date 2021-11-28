#include <stdio.h>
#include <stdlib.h>

// Question 1 defines:
#define FIXED_DT 1
#define BLOCK_SIZE 64
#define VEL 1
typedef float3 Particle;


//function declarations
__device__ inline float3& operator +=(float3 &a, const float3 &b);
__global__ void GPU_update(Particle *all_particles_ptr, float v,unsigned long long offset,unsigned int streamsize,unsigned int num_iterations); //Computes the update of only one update for all particles.
void particle_computation_launcher(unsigned int n_streams, unsigned long long num_particles,unsigned int batch_size,unsigned int num_iterations);
//function main
int main(int argc, char *argv[])
{
    printf("Running! \n");   
    unsigned int n_streams = atoi(argv[1]); //1 indicates the number of strings to use
    unsigned long long num_particles = ((unsigned long long)atoi(argv[2]))*1000; //The number of thousands of particles to use
    //unsigned int batch_size= std::max(1000,num_particles/2); //We use 2 batches if num particles is short. Else 1000 threads per batch, so we use most of the cores. I should try with different values of this to see optimal
    unsigned int batch_size= atoi(argv[3]);
    unsigned int num_iterations=atoi(argv[4]);
    particle_computation_launcher(n_streams, num_particles,batch_size,num_iterations);  

    return 0;
}

//function definitions
void particle_computation_launcher(unsigned int n_streams, unsigned long long num_particles,unsigned int batch_size, unsigned int num_iterations)
{
    float v=VEL; //Could be a random value.
    Particle *d_all_particles_ptr;
    Particle *all_particles_ptr;
    cudaStream_t *streams;
    const unsigned int streamSize=batch_size;
    const unsigned int streamBytes=streamSize*sizeof(Particle);
    if(num_particles%(batch_size) !=0)
    {
        fprintf(stderr, "Only programmed for num particles a multiple of n streams");
        exit(1);
    }
    //Allocates memory in host, device, and sets the values to 0.
    cudaMalloc(&d_all_particles_ptr, num_particles*sizeof(Particle)); 
    if (cudaMallocHost(&all_particles_ptr, num_particles*sizeof(Particle)) != cudaSuccess)
	    {printf("Could not allocate pag memory!");};
    for(unsigned int i=0; i< num_particles;i++)
    {
        all_particles_ptr[i].x=0;
        all_particles_ptr[i].y=0;
        all_particles_ptr[i].z=0;
    }
    streams=(cudaStream_t *) malloc(n_streams*sizeof(cudaStream_t)); //Allocates streams.
    for(unsigned int i=0; i< n_streams;i++)
        cudaStreamCreate(&streams[i]);
    // Batch processing with streams
    unsigned int curr_str=0; //Stream to send each batch
    for (unsigned int i=0; i< num_particles/batch_size ;i++)
    {
        unsigned long long offset= i*streamSize;
        //Some division might give problems where num particles%nstreams != 0 . 
        cudaMemcpyAsync(&d_all_particles_ptr[offset],&all_particles_ptr[offset], streamBytes, cudaMemcpyHostToDevice, streams[curr_str]);
        GPU_update<<<streamSize/BLOCK_SIZE+1,BLOCK_SIZE,0,streams[curr_str]>>>(d_all_particles_ptr,v,offset,streamSize,num_iterations);
        cudaMemcpyAsync(&all_particles_ptr[offset],&d_all_particles_ptr[offset], streamBytes, cudaMemcpyDeviceToHost, streams[curr_str]);
        curr_str=(curr_str+1)%n_streams; //After sending one batch, uses next stream for the next one. 
    }
    cudaDeviceSynchronize();
    //Checks result 
    for (long i = 0; i < num_particles; i++) 
    {
		if (fabs(all_particles_ptr[i].x - 1) > 0.01 ||fabs(all_particles_ptr[i].y - 1) > 0.01 || fabs(all_particles_ptr[i].z - 1) > 0.01  ) {
			fprintf(stderr, "Computation failed at index %ld", i);
			fprintf(stderr, "Values here: %f %f %f %f ", all_particles_ptr[i].x,all_particles_ptr[i].y,all_particles_ptr[i].z,all_particles_ptr[i+1].x);
			exit(1);
		}
    }
    printf("Output without errors!");
    //Destroys
    cudaFree(d_all_particles_ptr);
    for(unsigned int i=0; i< n_streams;i++)
        cudaStreamDestroy(streams[i]);
}


__global__ void GPU_update(Particle *d_all_particles_ptr, float v, unsigned long long offset, unsigned int streamSize,unsigned int num_iterations)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < streamSize) //Stream size may not be a multiple of the block size used.
    {
        for(unsigned int j=0; j<num_iterations;j++)
	{
        unsigned long long curr_particle=i+offset; 
        float update_value=v*FIXED_DT;
        float3 update = make_float3(update_value, update_value, update_value);
        d_all_particles_ptr[curr_particle]+=update;
	}
    }

}
__device__ inline float3& operator +=(float3 &a, const float3 &b) { //https://forums.developer.nvidia.com/t/operator-overloading-for-float4/27228

    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
