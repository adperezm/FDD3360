#include <stdio.h>
#include <math.h>
#include <sys/time.h>


#define TPB 256


double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void SAXPY(float *out, float a, float * x, float * y,unsigned int arraySize)
{
        #pragma acc parallel loop copyin(x[0:arraySize]) copyin(y[0:arraySize]) copyin(a) copyout(out[0:arraySize]) num_gangs(arraySize/256) num_workers(256) 
        for(int j = 0; j < arraySize; j++)
        {
                out[j]=x[j]*a+y[j];
		}
        
}
void SAXPYLauncher(float *out, float a, float * x, float * y,unsigned int arraySize) {

  double iStart = cpuSecond();
  SAXPY(out, a, x, y,arraySize);
  double calc_time = cpuSecond() - iStart;

  printf("GPU calculation time: %f  ", calc_time);
}

void CPU_saxpy(float *out, float a, float * x, float * y,unsigned int arraySize) {
        for(int j = 0; j < arraySize; j++)
        {
                out[j]=x[j]*a+y[j];
		}
}

int main(int argc, char *argv[])
{
        unsigned int arraySize= atoi(argv[1]);
        float *x = (float *)malloc(arraySize*sizeof(float));
        float *y = (float *)malloc(arraySize*sizeof(float));
        float *out_gpu = (float *)malloc(arraySize*sizeof(float));
        float *out_cpu = (float *)malloc(arraySize*sizeof(float));
        float a=3;
        float den=0;
        for(int i=0; i< arraySize;i++)
		{
                x[i]=1;
                y[i]=3;
        }
        printf("Array size: %d\n", arraySize);
        printf("Computing SAXPY on the CPU…");
        double iStart = cpuSecond();
        CPU_saxpy(out_cpu, a, x, y,arraySize);
        double iElaps_CPU = cpuSecond() - iStart;
        printf("Done! \n");
        printf("Computing SAXPY on the GPU…");
        iStart = cpuSecond();
        SAXPYLauncher(out_gpu, a, x, y,arraySize);
        double iElaps_GPU = cpuSecond() - iStart;
        printf("Done! \n");
        printf("Time elapsed CPU: %f \n Time elapsed GPU: %f \n",iElaps_CPU,iElaps_GPU);
        printf("First elements out array CPU: %f, %f, %f",out_cpu[0],out_cpu[1],out_cpu[2]);
        printf("First elements out array GPU: %f, %f, %f",out_gpu[0],out_gpu[1],out_gpu[2]);
        for(int i = 0; i < arraySize; i++)
        {
                if (abs(out_cpu[i])< 1e-6)
                {       den=1e-6; }
                else
                {       den = out_cpu[i]; }

                if(abs(out_gpu[i]-out_cpu[i])/den> 1e-3) {
                        printf("Difference encountered! \n");
                        break;
                }
        }

        return 0;
}

		
