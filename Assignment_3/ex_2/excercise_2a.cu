

// Question 1 defines:
#define FIXED_DT 1
#define BLOCK_SIZE 64
#define VEL 1
typedef float3 Particle;


//function declarations
__global__ void GPU_update(Particle *all_particles_ptr, float v,unsigned int num_particles); //Computes the update of only one update for all particles.
void particle_computation_launcher(bool use_cuda_malloc, unsigned int num_particles);
//function main
int main(int argc, char *argv[])
{
    printf("TBD");   
    unsigned int mode = atoi(argv[1]); //1 indicates cuda Malloc host, the other the simpler malloc.
    unsigned int num_particles = atoi(argv[2]);
    return 0;
}

//function definitions
void particle_computation_launcher(bool use_cuda_malloc, unsigned int num_particles)
{
    float v=VEL; //Could be a random value.
    Particle *all_particles_ptr = (Particle *) calloc(num_particles, sizeof(Particle)); 
    Particle *d_all_particles_ptr;
    if (use_cuda_malloc) //Allocation method
        cudaMalloc(&d_all_particles_ptr, num_particles*sizeof(Particle));
    else
        cudaMallocHost(&d_all_particles_ptr, num_particles*sizeof(Particle));
    GPU_update<<<num_particles/block_size+1,block_size>>>(d_all_particles_ptr,v,num_iterations,num_particles);
    cudaDeviceSynchronize();
    cudaMemcpy(all_particles_ptr,d_all_particles_ptr,num_particles*sizeof(Particle),cudaMemcpyDeviceToHost);
    cudaFree(d_all_particles_ptr);

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