// wrapper.h

// #include <cuda.h>
// #include <cuda_runtime.h>

const int threadsPerBlock = 512;

extern int getGPU_Info(void);

extern void cuda_function(int *route, int *distance, int N, float *crap);