CPU_proc_time=[0.022405 0.042172 0.090746 0.153184 0.276771 0.557631 1.094109 1.989788 4.060093 8.544947 18.419793 39.357227];
GPU_proc_time=[0.114105 0.115623 0.113173 0.081825 0.109776 0.088334 0.082486 0.096524 0.159872 0.227850 0.440376 0.837959 ];
N_part = [1e3 2e3 5e3 10e3 21e3 46e3 100e3 210e3 460e3 1000e3 2100e3 4600e3];

loglog(N_part, CPU_proc_time, 'DisplayName', 'CPU')
hold on
loglog(N_part, GPU_proc_time, 'DisplayName', 'GPU')
grid()
xlabel("Number of particles")
ylabel("Processing time [s]")
title("Comparison of processing time")
legend('Location','northwest')
