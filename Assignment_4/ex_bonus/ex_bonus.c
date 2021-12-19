#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

//Particle info vector will have the information needed to compute the particles. Complicated to do structs in OpenCL, so we Have to do it like that:
//Particle= Float[6] where Particle[0],Particle[1],Particle[2]= Vx, Vy, Vz and Particle[3],Particle[4],Particle[5]= Px, Py, Pz
#define FLOATS_PER_PARTICLE 6 
//#define CHECK_OUTPUTS 
#define DT_STEP 1
typedef float* ParticleInfo;

//Kernel to test
const char *vadd_program=
"__kernel \n"
"void vadd(__global float *A, __global float *B, __global float* C) \n"
"{ int index = get_global_id(0); \n"
"C[index]=A[index] + B[index]; } \n";

//Kernel declaration;
const char *kernel_char=
"__kernel \n"
"void update_particles(__global float *PartInfo,__global float* part_n_thousands, __global float* iterations ) { \n"
"   int index = get_global_id(0);float currPart[6]; \n"
"   for(unsigned int i=0;i<6;i++)\n"
"      currPart[i]=PartInfo[index*6+i]; \n"
"   //float* currParticle=PartInfo+index*6;\n" 
"   for(unsigned int i=0;i<*iterations;i++)\n"
"   {\n"
"     currPart[0]=1;currPart[1]=1;currPart[2]=1; //Update of velocity (setting to 1 so then its easier to check that output is expected)\n"
"     currPart[3]+=currPart[0]*1;currPart[4]+=currPart[1]*1;currPart[5]+=currPart[2]*1; //Update of position\n"
"   }\n"
"   for(unsigned int i=0;i<6;i++)\n"
"      PartInfo[index*6+i]=currPart[i]; \n"
"}\n";



double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void myAssert(cl_int err, const char * msg)
{
	if (err != CL_SUCCESS)
{
	printf("Assert message: %s. Error number: %d \n",msg, err);
	assert(0);}
}

//function declarations
void CPU_compute_particle_simulation(unsigned int num_particles,unsigned int num_iterations);
void GPU_OpenCL_compute_particle_simulation(unsigned int num_particles,unsigned int num_iterations,unsigned int batch_size);
//function main
int main(int argc, char *argv[])
{
    unsigned int num_particles_thou = atoi(argv[1]); //The number of thousands of particles to use
    unsigned int num_iterations=atoi(argv[2]);
    unsigned int batch_size=atoi(argv[3]);
    printf("Runing particle updater with N particles : %d (thousands)  and N iterations: %d Batch size: %d \n", num_particles_thou, num_iterations, batch_size);
    double iStart = cpuSecond();
    CPU_compute_particle_simulation(num_particles_thou, num_iterations);
    double time_elapsed_CPU=cpuSecond() - iStart;
    double iStart2 = cpuSecond();
    GPU_OpenCL_compute_particle_simulation(num_particles_thou, num_iterations,batch_size);
    double time_elapsed_GPU_OpenCL=cpuSecond() - iStart2;
    printf("CPU processing time: %f  \nGPU processing time: %f  \n", time_elapsed_CPU,time_elapsed_GPU_OpenCL);
    return 0;
}



//CPU functions
void CPU_compute_particle_simulation(unsigned int num_particles,unsigned int num_iterations)
{
    ParticleInfo ParticlesInfo=(ParticleInfo) calloc((unsigned long long) (num_particles*1000.0),sizeof(float)*FLOATS_PER_PARTICLE);

    for(unsigned long long part=0;part<num_particles*1000;part++)
    {
        ParticleInfo currParticle=ParticlesInfo+part*FLOATS_PER_PARTICLE;
        for(unsigned int i=0;i<num_iterations;i++)
        {
            currParticle[0]=1;currParticle[1]=1;currParticle[2]=1; //Update of velocity (setting to 1 so then its easier to check that output is expected)
            currParticle[3]+=currParticle[0]*DT_STEP;currParticle[4]+=currParticle[1]*DT_STEP;currParticle[5]+=currParticle[2]*DT_STEP; //Update of position
        }
    }
    #ifdef CHECK_OUTPUTS
    for(unsigned long long part=0;part<num_particles*1000;part++)
    {
        ParticleInfo currParticle=ParticlesInfo+part*FLOATS_PER_PARTICLE;
        if (fabs(currParticle[3] - num_iterations) > 0.01 ||fabs(currParticle[4] - num_iterations) > 0.01 || fabs(currParticle[5] - num_iterations) > 0.01  ) {
			fprintf(stderr, "CPU Computation failed at particle %ld", part);
			fprintf(stderr, "Values here: %f %f %f ", currParticle[3],currParticle[4],currParticle[5]);
			exit(1);
		}
    }
    #endif
    free(ParticlesInfo);
}

// GPU With Open CL
void GPU_OpenCL_compute_particle_simulation(unsigned int num_particles,unsigned int num_iterations,unsigned int batch_size)
{

    num_particles=((unsigned int)((float)num_particles*1000)/batch_size)*batch_size; //So can be divided by batch size


    cl_platform_id * platforms; cl_uint n_platform;
    //Platforms in the system 
    cl_int err = clGetPlatformIDs(0,NULL,&n_platform); myAssert(err,"Platforms 1");
    platforms= (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
    err=clGetPlatformIDs(n_platform,platforms,NULL); myAssert(err, "Platforms 2");

    //Devices inside platforms
    cl_device_id * device_list; cl_uint n_devices;
    err=clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU,0,NULL,&n_devices); myAssert(err,"Devices");
    device_list= (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
    err=clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU,n_devices,device_list,NULL); myAssert(err,"Devices 2");


    cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err);myAssert(err, "Context");

    cl_command_queue cmd_queue= clCreateCommandQueue(context, device_list[0],0,&err);myAssert(err,"CommandQueue");

    ParticleInfo particlesInfo=(ParticleInfo) calloc(num_particles,sizeof(float)*FLOATS_PER_PARTICLE);
    unsigned int array_size=FLOATS_PER_PARTICLE*num_particles*sizeof(float);
    unsigned int size_float = sizeof(float);

    cl_mem particlesInfo_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size,NULL, &err);myAssert(err,"Buffer 1");
    cl_mem num_particles_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, size_float,NULL, &err);myAssert(err,"Buffer 2");
    cl_mem num_iterations_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, size_float,NULL, &err);

    float num_particles_f=(float) num_particles;   
    float num_iterations_f=(float) num_iterations;
    printf("Number of iterations to send: %f, Number of particles to send: %f \n",num_iterations_f,num_particles_f); 

    err = clEnqueueWriteBuffer(cmd_queue,particlesInfo_dev,CL_TRUE,0,array_size,particlesInfo,0,NULL,NULL);myAssert(err, "Enqueue write buffer 1");
    err = clEnqueueWriteBuffer(cmd_queue,num_particles_dev,CL_TRUE,0,size_float,(void *)&num_particles_f,0,NULL,NULL);myAssert(err,"Enqueue 2");
    err = clEnqueueWriteBuffer(cmd_queue,num_iterations_dev,CL_TRUE,0,size_float,(void *)&num_iterations_f,0,NULL,NULL);

    cl_program program= clCreateProgramWithSource(context,1,(const char **) &kernel_char, NULL, &err);myAssert(err,"Program creation");
    //Build kernel
    err = clBuildProgram(program,1,device_list, NULL, NULL, NULL);
    if(err != CL_SUCCESS)
    { 
        size_t len; char buffer[2048];
	cl_int ret=clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        printf("Err: %d Len : %d Ret: %d \n",err,len,ret);
	//buffer = (char *) malloc(len*sizeof(char));
        clGetProgramBuildInfo(program,device_list[0],CL_PROGRAM_BUILD_LOG,sizeof(buffer),buffer,&len);
	//printf("Build error: %s \n",buffer);
        fprintf(stderr,"Build error: %s\n",buffer);
	exit(1);
    }
    cl_kernel kernel =clCreateKernel(program,"update_particles", &err);myAssert(err,"Kernel creation");
    //Set arguments kernel
    err=clSetKernelArg(kernel,0,sizeof(cl_mem),(void * ) &particlesInfo_dev );myAssert(err,"Kernel Arguments");
    err=clSetKernelArg(kernel,1,sizeof(cl_mem),(void * ) &num_particles_dev );myAssert(err,"Kernel Arguments 2");
    err=clSetKernelArg(kernel,2,sizeof(cl_mem),(void * ) &num_iterations_dev );

    size_t n_workitem=num_particles;
    size_t workgroup_size= batch_size;
    //print("Max group size: %d \n",
    //Launch kernel
    err = clEnqueueNDRangeKernel(cmd_queue,kernel,1,NULL, &n_workitem,&workgroup_size,0,NULL,NULL);myAssert(err,"Enqueue1");
    //Transfer data from C back
    err = clEnqueueReadBuffer(cmd_queue,particlesInfo_dev,CL_TRUE,0,array_size,particlesInfo,0,NULL,NULL);myAssert(err,"Transfer Data");

    err = clFlush(cmd_queue);
    err = clFinish(cmd_queue);myAssert(err,"Problem in Finish");
    #ifdef CHECK_OUTPUTS
    printf("Lets print some values %f %f %f and %f ", particlesInfo[0],particlesInfo[3],particlesInfo[6], particlesInfo[9]);
    for(unsigned long long part=0;part<num_particles;part++)
    {
        ParticleInfo currParticle=particlesInfo+part*FLOATS_PER_PARTICLE;
        if (fabs(currParticle[3] - num_iterations) > 0.01 ||fabs(currParticle[4] - num_iterations) > 0.01 || fabs(currParticle[5] - num_iterations) > 0.01  ) {
			fprintf(stderr, "GPU Computation failed at particle %ld", part);
			fprintf(stderr, "Values here: %f %f %f and %f ", currParticle[3],currParticle[4],currParticle[5], currParticle[1]);
			exit(1);
		}
    }
    #endif 
    free(particlesInfo);

}
