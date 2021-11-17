T = readtable('Outputs.csv')
Arr_size=T.ArraySize;
CPU_proc_t=T.CPUComputingTime;
GPU_proc_t=T.GPULauncherTime;
GPU_proc_t_without_mem_transf=T.GPURoutine_withoutMemoryTransferences_;


loglog(Arr_size,CPU_proc_t,Arr_size,GPU_proc_t,Arr_size,GPU_proc_t_without_mem_transf)
legend('CPU', 'GPU', 'GPU w/o memory transfer','Location','southeast')
xlabel('Array Size [#floats]')
ylabel('Processing time [s]')
title('Evaluation of SAXPY performance')