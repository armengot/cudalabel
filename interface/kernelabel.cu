#include <cuda_runtime.h>

__global__ void kgetinfo(unsigned int* d_labels, unsigned int** output, unsigned int i_current_label, int rows_number, int cols_number) 
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int current_label = output[i_current_label][0];    
    if (tid_x < cols_number && tid_y < rows_number) 
    {
        for (int BX = 0; BX < blockDim.x; BX++)
        {
            for (int BY = 0; BY < blockDim.y; ++BY)
            {
                int pixelX = blockIdx.x * blockDim.x + BX;
                int pixelY = blockIdx.y * blockDim.y + BY;

                // Check the boundary condition
                if (pixelX < cols_number && pixelY < rows_number)
                {
                    int index = pixelY * cols_number + pixelX;
                    if (d_labels[index]==current_label)                                         
                    {   
                        atomicMin(&output[i_current_label][1], pixelX);
                        atomicMax(&output[i_current_label][2], pixelX);
                        atomicMin(&output[i_current_label][3], pixelY);
                        atomicMax(&output[i_current_label][4], pixelY);                                                                                                      
                    }                                   
                }
            }
        }
    }    
}