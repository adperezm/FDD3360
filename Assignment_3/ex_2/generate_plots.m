%Creates plots
% fid = fopen('Ex2a output for plots.txt');
% txt = textscan(fid,'%s','delimiter','\n'); 
% all_text_lines=txt{1};
% fclose(fid)
% num_particles_list=[];
% for i=1:len(all_text_lines)
%     curr_line=all_text_lines{i};
%     command_str="command: ./excercise_2a.out 1" ;
%     k=strfind(curr_line,command_str);
%     if len(k)>0
%        idx_num_part=len( command_str)+k(1);
%        num_particles_list.append(double(curr_line(idx_num_part:end)));
%     end
%     
% end
%La verdad que es mas facil copiarlo a mano del txt dios
malloc_num_particles=[1e3,   1e4,       1e5,      1e6,        1e7,     3.16e7,    1e8      ,  3.16e8  , 1e9 ];
malloc_time_HtoD=[2.3040e-6,11.904e-6, 100.96e-6,1.1559e-3, 11.827e-3, 34.715e-3, 109.73e-3, 346.78e-3,1.10940 ];

cuda_malloc_num_particles=[1e3,   1e4,       1e5,      1e6,        1e7,     3.16e7,    1e8      ,  3.16e8  , 1e9 ];
cuda_malloc_time_HtoD=[2.0800e-6, 11.712e-6, 98.335e-6, 967.29e-6, 9.6561e-3, 30.511e-3, 96.556e-3, 305.09e-3, 974.24e-3];

loglog(malloc_num_particles, malloc_time_HtoD, 'DisplayName', 'Malloc')
hold on
loglog(cuda_malloc_num_particles, cuda_malloc_time_HtoD, 'DisplayName', 'Cuda Malloc')
grid()
xlabel("Number of particles")
ylabel("CUDA memcpy HtoD [s]")
title("Comparison of memory allocators")
legend()

%%Now I should plot (cudamemmanagedmalloc+cudafree) vs
%%(HtoD,DtoH,pinnedmalloc, free) 
cuda_malloc_time_DtoH=[1.6320e-6,              9.7920e-6,        91.391e-6   ,       911.96e-6,          9.1135e-3,      28.793e-3    ,    91.107e-3,        287.90e-3   ,     912.93e-3       ];
cuda_malloc_time_free=[315.73e-6+78.795e-6, 77.600e-6+315.21e-6,  315.42e-6+80.858e-6,1.6404e-3+108.61e-6, 15.859e-3+263.78e-6,42.484e-3+459.00e-6,122.23e-3+1.1159e-3,372.50e-3+ 3.1904e-3,1.16619+9.7976e-3];%Free host + free del device
cuda_malloc_time_alloc=[159.80e-3+565.14e-6,147.76e-3+541.05e-6, 150.29e-3+551.20e-6, 147.82e-3+2.7027e-3,153.99e-3+26.109e-3,149.63e-3+80.887e-3 ,255.13e-3+44.55e-3, 806.30e-3+ 151.85e-3,2.57175+161.31e-3];  %alloc del host + device
%Una observacion es que el alloc en la gpu medio que siempre tarda lo
%mismo, pero el host va cambiando el tiempo. Los frees tmb toman mucho
%tiempo en el host
managed_num_part=[1e3,1e4,1e5, 1e6,1e7,3.16e7,1e8,3.16e8,1e9];
managed_time_alloc=[189.45e-3,185.63e-3,188.96e-3, 185.66e-3,177.81e-3,163.90e-3,172.77e-3,179.17e-3,173.01e-3];
managed_time_free=[92.274e-6,88.039e-6, 141.39e-6, 576.97e-6,5.3581e-3,16.798e-3,48.510e-3,153.78e-3,484.79e-3];

hold off
loglog(cuda_malloc_num_particles, cuda_malloc_time_HtoD+cuda_malloc_time_DtoH+cuda_malloc_time_free+cuda_malloc_time_alloc, 'DisplayName', 'Cuda Malloc Host')
hold on
loglog(managed_num_part,managed_time_alloc+ managed_time_free, 'DisplayName', 'Managed memory')
grid()
xlabel("Number of particles")
ylabel("Total time spent on memory routines [s]")
title("Perofrmance of managed memory")
legend()

