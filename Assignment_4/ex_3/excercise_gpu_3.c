#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <math.h>

#define arraySize 12800000

float x[arraySize];
float y[arraySize];
float out_cpu[arraySize];
float out_gpu[arraySize];

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char *argv[])
{

        float a=3;
        float den=0;
        printf("Initializing arrays: %d\n", arraySize);
        for(int i=0; i< arraySize;i++)
		{
                x[i]=1;
                y[i]=3;
        }
        printf("Array size: %d\n", arraySize);
        // SAXPY IN THE CPU
        printf("Computing SAXPY on the CPU…");
        double iStart = cpuSecond();
        for(int j = 0; j < arraySize; j++)
        {
                out_cpu[j]=x[j]*a+y[j];
		}
        double iElaps_CPU = cpuSecond() - iStart;
        printf("Done! \n");
        // SAXPY IN THE GPU
        printf("Computing SAXPY on the GPU…");
        iStart = cpuSecond();
        #pragma acc data copyin(x[0:arraySize]) copyin(y[0:arraySize]) copyin(a) copy(out_gpu[0:arraySize])
        {
            #pragma acc parallel loop
            for(int j = 0; j < arraySize; j++)
            {
                out_gpu[j]=x[j]*a+y[j];
            }
        }
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

                //if(abs(out_gpu[i]-out_cpu[i])/den> 1e-3) {
                //        printf("Difference encountered! \n");
                //        break;
                //}
        }


        return 0;
}

		
