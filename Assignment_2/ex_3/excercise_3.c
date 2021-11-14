#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
/* Definitions */
typedef  struct{
    float x,y,z;
}float3; // Comment when it is compiled in cuda 

typedef float3 Particle;

// Question 1 defines:
#define QUESTION_1_NUM_ITERATIONS 1000.0
#define QUESTION_1_NUM_PARTICLES_MIN 10.0
#define QUESTION_1_NUM_PARTICLES_MAX 10000.0
#define QUESTION_1_NUM_RUNS 10
#define QUESTION_1_DT 1.0
#define QUESTION_1_VEL 1.0

// Question 2 defines:
#define QUESTION_2_NUM_ITERATIONS 1000.0
#define QUESTION_2_NUM_PARTICLES_MIN 10.0
#define QUESTION_2_NUM_PARTICLES_MAX 10000.0
#define QUESTION_2_NUM_RUNS 10
#define QUESTION_2_BLOCKS {16,32,64,128,256}
#define QUESTION_2_NUM_BLOCKS_RUN 5
#define QUESTION_2_DT 1.0
#define QUESTION_2_VEL 1.0
/* Function declarations */
void question1_loop(void);
void question2_loop(void);
void save_float_array(char * filename,char * x_label,char * y_label, float * x_farray,float * y_farray, unsigned int n_elements);
float CPU_compute_particle_simulation(unsigned int num_particles, unsigned int num_iterations);
float GPU_compute_particle_simulation(unsigned int num_particles, unsigned int num_iterations, unsigned int block_size);//TBD
double cpuSecond(void);
/*Main*/
int main(int argc, char *argv[])
{
    printf("Test");   
    question1_loop();
    question2_loop();
    
    return 0;
}

/*Loops of the excercise */
void question1_loop()
{
    float CPU_proc_times[QUESTION_1_NUM_RUNS],num_particles[QUESTION_1_NUM_RUNS];
    for(unsigned int i=0;i<QUESTION_1_NUM_RUNS; i++) 
    {
        num_particles[i]=(unsigned int) (QUESTION_1_NUM_PARTICLES_MIN + (float)i*((QUESTION_1_NUM_PARTICLES_MAX-QUESTION_1_NUM_PARTICLES_MIN)/(float)QUESTION_1_NUM_RUNS));
        CPU_proc_times[i]=CPU_compute_particle_simulation(num_particles[i], QUESTION_1_NUM_ITERATIONS);
    }
    save_float_array("Question1.txt", "Num Particles", "CPU Processing time",num_particles,CPU_proc_times, QUESTION_1_NUM_RUNS);
}

void question2_loop()
{
    float GPU_proc_times[QUESTION_2_NUM_BLOCKS_RUN][QUESTION_2_NUM_RUNS],num_particles[QUESTION_2_NUM_BLOCKS_RUN][QUESTION_2_NUM_RUNS];
    unsigned int block_size[QUESTION_2_NUM_BLOCKS_RUN]=QUESTION_2_BLOCKS;
    for(unsigned int curr_block_index=0;curr_block_index<QUESTION_2_NUM_BLOCKS_RUN;curr_block_index++)
    {    
        for(unsigned int curr_n_particles=0;curr_n_particles<QUESTION_2_NUM_RUNS; curr_n_particles++) //Generates num particles array
        {
            num_particles[curr_block_index][curr_n_particles]=(unsigned int) (QUESTION_2_NUM_PARTICLES_MIN + (float)curr_n_particles*((QUESTION_2_NUM_PARTICLES_MAX-QUESTION_2_NUM_PARTICLES_MIN)/(float)QUESTION_2_NUM_RUNS));
            GPU_proc_times[curr_block_index][curr_n_particles]=GPU_compute_particle_simulation(num_particles[curr_block_index][curr_n_particles], QUESTION_2_NUM_ITERATIONS,block_size[curr_block_index]);
        }
    }
    /*Code to save output */
    FILE *fp;
    fp = fopen("Question2.txt", "w");
    fprintf(fp,"Block number\tNumber of particles\tTime elapsed\n");
    for (unsigned i = 0; i < QUESTION_2_NUM_BLOCKS_RUN; i++) {
        for (unsigned j = 0; j < QUESTION_2_NUM_RUNS; j++) 
            fprintf(fp, "%d\t%f\t%f\n",block_size[i],num_particles[i][j],GPU_proc_times[i][j]);
    }
    fclose(fp);
}


/* CPU computation */
float CPU_compute_particle_simulation(unsigned int num_particles, unsigned int num_iterations)
{
    
    float v=QUESTION_1_VEL; //Could be a random value.
    double start_time=cpuSecond();
    Particle *all_particles_ptr = (Particle *) calloc(num_particles, sizeof(Particle));
    for(unsigned int it_n=0; it_n<num_iterations; it_n++)
    {
        for(unsigned int curr_particle=0; curr_particle<num_particles; curr_particle++)
        {
            all_particles_ptr[curr_particle].x+=v*QUESTION_1_DT;
            all_particles_ptr[curr_particle].y+=v*QUESTION_1_DT;
            all_particles_ptr[curr_particle].z+=v*QUESTION_1_DT;
        }
    }
    double time_elapsed=cpuSecond() - start_time;
    printf("Num particles: %d, Time elapsed CPU: %f \n",num_particles,time_elapsed);
    return time_elapsed; // change
}
/* GPU computation */ 
float GPU_compute_particle_simulation(unsigned int num_particles, unsigned int num_iterations, unsigned int block_size)//TBD
{
    return num_particles;
}



/*Auxiliar functions */

void save_float_array(char * filename,char * x_label,char * y_label, float * x_farray,float * y_farray, unsigned int n_elements)
{
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp,x_label);
    fprintf(fp,"\t");
    fprintf(fp,y_label);
    fprintf(fp,"\n");
    for (unsigned i = 0; i < n_elements; i++) {
        fprintf(fp, "%f\t%f\n",x_farray[i],y_farray[i]);
    }
    fclose(fp);
}
void save_float_array_3(char * filename,char * x_label,char * y_label,char * z_label, float * x_farray,float * y_farray,float * z_farray, unsigned int n_elements) //Could have done a general case, but this was faster to debug.
{
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp,x_label);
    fprintf(fp,"\t");
    fprintf(fp,y_label);
    fprintf(fp,"\t");
    fprintf(fp,z_label);
    fprintf(fp,"\n");
    for (unsigned i = 0; i < n_elements; i++) {
        fprintf(fp, "%f\t%f\t%f\n",x_farray[i],y_farray[i],z_farray[i]);
    }
    fclose(fp);
}

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}