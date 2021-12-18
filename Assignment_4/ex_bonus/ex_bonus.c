#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <assert.h>


//Particle info vector will have the information needed to compute the particles. Complicated to do structs in OpenCL, so we Have to do it like that:
//Particle= Float[6] where Particle[0],Particle[1],Particle[2]= Vx, Vy, Vz and Particle[3],Particle[4],Particle[5]= Px, Py, Pz
#define FLOATS_PER_PARTICLE 6 
#define CHECK_OUTPUTS 
#define DT_STEP 1
typedef float* ParticleInfo

//Kernel declaration;
const char * kernel=
"__kernel \n"
"void update_particles{__global float *PartInfo,__global unsigned long long part_n, __global unsigned int iterations } { \n"
"   int index = get_global_id(0); \n"
"   if(index<part_n){\n"
"       float* currParticle=ParticlesInfo+index*6;\n" 
"       for(unsigned int i=0;i<iterations;i++)\n"
"       {\n"
"           currParticle[0]=1;currParticle[1]=1;currParticle[2]=1; //Update of velocity (setting to 1 so then its easier to check that output is expected)\n"
"           currParticle[3]+=currParticle[0]*1;currParticle[4]+=currParticle[1]*1;currParticle[5]+=currParticle[2]*1; //Update of position\n"
"       }\n"
"    }\n"
"}\n";


//function declarations
void CPU_compute_particle_simulation(unsigned long long num_particles,unsigned int num_iterations);
void GPU_OpenCL_compute_particle_simulation(unsigned long long num_particles,unsigned int num_iterations,unsigned int batch_size);
//function main
int main(int argc, char *argv[])
{
    unsigned long long num_particles = ((unsigned long long)atoi(argv[1]))*1000; //The number of thousands of particles to use
    unsigned int num_iterations=atoi(argv[2]);
    unsigned int batch_size=atoi(argv[3]);
    printf("Runing particle updater with N particles : %llu  and N iterations: %d \n", num_particles, num_iterations)
    double iStart = cpuSecond();
    CPU_compute_particle_simulation(num_particles, num_iterations);
    double time_elapsed_CPU=cpuSecond() - iStart;
    iStart = cpuSecond();
    GPU_OpenCL_compute_particle_simulation(num_particles, num_iterations,batch_size);
    double time_elapsed_GPU_OpenCL=cpuSecond() - iStart;
    printf("CPU processing time: %f  \nGPU processing time: %f  \n", time_elapsed_CPU,time_elapsed_GPU_OpenCL);
    return 0;
}



//CPU functions
void CPU_compute_particle_simulation(unsigned long long num_particles,unsigned int num_iterations)
{
    ParticleInfo ParticlesInfo=(ParticleInfo) calloc(num_particles,sizeof(float)*FLOATS_PER_PARTICLE);

    for(unsigned long long part=0;part<num_particles;part++)
    {
        ParticleInfo currParticle=ParticlesInfo+part*FLOATS_PER_PARTICLE;
        for(unsigned int i=0;i<num_iterations;i++)
        {
            currParticle[0]=1;currParticle[1]=1;currParticle[2]=1; //Update of velocity (setting to 1 so then its easier to check that output is expected)
            currParticle[3]+=currParticle[0]*DT_STEP;currParticle[4]+=currParticle[1]*DT_STEP;currParticle[5]+=currParticle[2]*DT_STEP; //Update of position
        }
    }
    #ifdef CHECK_OUTPUTS
    for(unsigned long long part=0;part<num_particles;part++)
    {
        ParticleInfo currParticle=ParticlesInfo+part*FLOATS_PER_PARTICLE;
        if (fabs(currParticle[3] - num_iterations) > 0.01 ||fabs(currParticle[4] - num_iterations) > 0.01 || fabs(currParticle[5] - num_iterations) > 0.01  ) {
			fprintf(stderr, "Computation failed at particle %ld", part);
			fprintf(stderr, "Values here: %f %f %f ", currParticle[3],currParticle[4],currParticle[5]);
			exit(1);
		}
    }
    #endif
    free(ParticlesInfo);
}

// GPU With Open CL
void GPU_OpenCL_compute_particle_simulation(unsigned long long num_particles,unsigned int num_iterations,unsigned int batch_size)
{
    cl_platform_id * platforms; cl_uint n_platform;
    //Platforms in the system 
    cl_int err = clGetPlatformIDs(0,NULL,&n_platform); assert(err!=CL_SUCCESS);
    platforms= (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
    err=clGetPlatformIDs(n_platform,platforms,NULL); assert(err!=CL_SUCCESS);

    //Devices inside platforms
    cl_device_id * device_list; cl_uint n_devices;
    err=clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU,0,NULL,&n_devices); assert(err!=CL_SUCCESS);
    device_list= (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
    err=clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU,n_devices,device_list,NULL); assert(err!=CL_SUCCESS);


    cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err);assert(err!=CL_SUCCESS);

    cl_command_queue cmd_queue= clCreateCommandQueue(context, device_list[0],0,&err);assert(err!=CL_SUCCESS);

    ParticleInfo particlesInfo=(ParticleInfo) calloc(num_particles,sizeof(float)*FLOATS_PER_PARTICLE);
    unsigned int array_size=FLOATS_PER_PARTICLE*num_particles;

    cl_mem particlesInfo_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size,NULL, &err);assert(err!=CL_SUCCESS);
    cl_mem num_particles_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, 1,NULL, &err);assert(err!=CL_SUCCESS);
    cl_mem num_iterations_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, 1,NULL, &err);assert(err!=CL_SUCCESS);

    err = clEnqueueWriteBuffer(cmd_queue,particlesInfo_dev,CL_TRUE,0,array_size,particlesInfo,0,NULL);assert(err!=CL_SUCCESS);
    err = clEnqueueWriteBuffer(cmd_queue,num_particles_dev,CL_TRUE,0,1,num_particles,0,NULL);assert(err!=CL_SUCCESS);
    err = clEnqueueWriteBuffer(cmd_queue,num_iterations_dev,CL_TRUE,0,1,num_iterations,0,NULL);assert(err!=CL_SUCCESS);

    cl_program program= clCreateProgramWithSource(context,1,(const char **) &kernel, NULL, &err);assert(err!=CL_SUCCESS);
    //Build kernel
    err = clBuildProgram(program,1,device_list, NULL, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        size_t len; char buffer[2048];
        clGetProgramBuildInfo(program,device_list[0],CL_PROGRAM_BUILD_LOG,sizeof(buffer),biffer,&len);
        fprintf(stderr,"Build error: %s\n",buffer);exit(1);
    }
    cl_kernel=clCreateKernel(program,"update_particles", &err);
    //Set arguments kernel
    err=clSetKernelArg(kernel,0,sizeof(cl_mem),(void * ) &particlesInfo_dev );assert(err!=CL_SUCCESS);
    err=clSetKernelArg(kernel,1,sizeof(cl_mem),(void * ) &num_particles_dev );assert(err!=CL_SUCCESS);
    err=clSetKernelArg(kernel,2,sizeof(cl_mem),(void * ) &num_iterations_dev );assert(err!=CL_SUCCESS);

    size_t n_workitem=num_particles;
    size_t workgroup_size= batch_size;
    //Launch kernel
    err = clEnqueueNDRangeKernel(cmd_queue,kernel,1,NUL, &n_workitem,&workgroup_size,0,NULL,NULL);assert(err!=CL_SUCCESS);
    //Transfer data from C back
    err = clEnqueueReadBuffer(cmd_queue,particlesInfo_dev,CL_TRUE,0,array_size,particlesInfo,0,NULL,NULL);assert(err!=CL_SUCCESS);

    err = clFlush(cmd_queue);
    err = clFinish(cmd_queue);
    #ifdef CHECK_OUTPUTS
    for(unsigned long long part=0;part<num_particles;part++)
    {
        ParticleInfo currParticle=ParticlesInfo+part*FLOATS_PER_PARTICLE;
        if (fabs(currParticle[3] - num_iterations) > 0.01 ||fabs(currParticle[4] - num_iterations) > 0.01 || fabs(currParticle[5] - num_iterations) > 0.01  ) {
			fprintf(stderr, "Computation failed at particle %ld", part);
			fprintf(stderr, "Values here: %f %f %f ", currParticle[3],currParticle[4],currParticle[5]);
			exit(1);
		}
    }
    #endif 
    free(particlesInfo)

}