/* estandar/external headers*/
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <set>

/* project headers */
#include <cudalabel.h>
#include <CCL.cuh>
#include <utils.hpp>

/* as in CCL.cu */
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4

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

// modified from folkev <= D. P. Playne and K. Hawick (2018)
unsigned int cudalabel_countool(unsigned int* img, size_t N, std::set<unsigned int>* realabels)
{
    unsigned int components = 0;
    for (int i = 0; i < N; i++)
    {
        // Each new component will have its root+1 as label
        if (img[i] == i+1)
        {
            components ++;
            realabels->insert(img[i]);
        }
    }
    return(components);
}

cudalabel::cudalabel() 
{
}

cudalabel::~cudalabel() 
{
    if (d_img)
	    cudaFree(d_img);
    if (d_labels)
	    cudaFree(d_labels);
    if (output)
        output->release();
    if (gpuinfo) 
    {
        for (int i = 0; i < nlabels; ++i) 
        {
            cudaFree(gpuinfo[i]);            
        }
        cudaFree(gpuinfo);        
    }
}

void cudalabel::setimg(const cv::Mat input) 
{
    ncols = input.cols;
    nrows = input.rows;
    npixel = nrows*ncols;
    
    cudaMallocManaged(&d_labels, npixel * sizeof(int));
    cudaMallocManaged(&d_img, npixel * sizeof(char));
    image = input.clone();
}

void cudalabel::setgpuimg(const cv::cuda::GpuMat input)
{
    ncols = input.cols;
    nrows = input.rows;
    npixel = nrows*ncols;
    
    cudaMallocManaged(&d_labels, npixel * sizeof(int));
    cudaMallocManaged(&d_img, npixel * sizeof(char));
    gpuimage = input.clone();
}

void cudalabel::preprocess()
{
    double imin, imax;
    cv::Point minloc, maxloc;   
    if (!gpuimage.empty())
    {           
        cv::cuda::GpuMat localthres;
        cv::cuda::minMax(gpuimage,&imin,&imax);        
        imean = static_cast<unsigned int>(imin+((imax-imin)/2));    
        cv::cuda::threshold(gpuimage, localthres, imean, imax, cv::THRESH_BINARY);        
        localthres.copyTo(cv::cuda::GpuMat(gpuimage.size(), gpuimage.type(), d_img));
    }
    else if (!image.empty())    
    {        
        cv::minMaxLoc(image,&imin,&imax,&minloc,&maxloc);
        imean = static_cast<unsigned int>(imin+((imax-imin)/2));    
	    util::threshold(d_img, image.data, imean, npixel);                
    }
    else
    {
        std::cerr << "No data available." << std::endl;
    }
}

void cudalabel::labelize()
{
    connectedComponentLabeling(d_labels, d_img, ncols, nrows);
    nlabels = cudalabel_countool(d_labels, npixel, &finalabels);
    /* check if required 
	cv::Mat finalImage = util::postProc(d_labels, ncols, nrows);
	cv::imshow("Labelled image", finalImage);
	cv::waitKey(); */
}

void cudalabel::imgen()
{
    output = new cv::cuda::GpuMat(nrows, ncols, CV_32SC1, d_labels);
}

void cudalabel::lsave(std::string outputname)
{
    cv::Mat result;
    output->download(result);
    cv::imwrite(outputname,result);
}

unsigned int cudalabel::lnumber()
{
    return(nlabels);
}

unsigned int cudalabel::lmean()
{
    return(imean);
}

unsigned int** cudalabel::getinfo() 
{
    // Create Grid/Block
	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid((ncols+BLOCK_SIZE_X-1)/BLOCK_SIZE_X,(nrows+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y);    

    /*
    std::cout << "BLOCK: " << block.x << " " << block.y << " " << block.z << std::endl;
    std::cout << "GRID: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    */

    // output data    
    cudaMallocManaged(&gpuinfo, finalabels.size() * sizeof(unsigned int*));
    int i = 0;
    for (const auto& ilabel : finalabels)
    {
        cudaMallocManaged(&gpuinfo[i], 5 * sizeof(unsigned int));     
        gpuinfo[i][0] = ilabel;
        gpuinfo[i][1] = ncols;
        gpuinfo[i][2] = 0;
        gpuinfo[i][3] = nrows;
        gpuinfo[i][4] = 0;        
        i++;
    }    
    // kernel call
    for(unsigned int i=1; i<nlabels; i++)
    {
        kgetinfo<<<grid, block>>>(d_labels, gpuinfo, i, nrows, ncols);                        
        // gpu sync
        //cudaDeviceSynchronize();
    }        
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }        
    return(gpuinfo);
}
