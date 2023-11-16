#ifndef KERNELABEL_H
#define KERNELABEL_H

#include <cuda_runtime.h>

extern "C" 
{
    __global__ void kgetinfo(unsigned int* d_labels, unsigned int** output, unsigned int i_current_label, int rows_number, int cols_number);
}

#endif
