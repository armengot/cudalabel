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
#include <unistd.h>

/* project headers */
#include <cudalabel.h>
#include <CCL.cuh>
#include <utils.hpp>

/* as in CCL.cu */
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4

__global__ void kgetinfo(unsigned int* d_labels, unsigned int** outinfo, unsigned int Nlabel, int rows_number, int cols_number) 
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

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
                    if (d_labels[index]!=0)
                    {   
                        int i;
                        bool protect = false;
                        for(i=0;i<Nlabel;i++)
                        {
                            if (d_labels[index] == outinfo[i][0])
                            {
                                protect = true;
                                break;                            
                            }
                        }
                        if (protect)
                        {
                            atomicMin(&outinfo[i][1], pixelX);
                            atomicMax(&outinfo[i][2], pixelX);
                            atomicMin(&outinfo[i][3], pixelY);
                            atomicMax(&outinfo[i][4], pixelY);
                        }
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

/* builder */
cudalabel::cudalabel() 
{
}

/* destructor */
void cudalabel::reset()
{
    if (d_img)
    {
	    cudaFree(d_img);
        d_img = nullptr;
    }
    if (d_labels)
    {
	    cudaFree(d_labels);    
        d_labels = nullptr;
    }
    if (gpuinfo) 
    {
        for (int i = 0; i < nlabels; ++i) 
        {
            cudaFree(gpuinfo[i]);            
        }
        cudaFree(gpuinfo);     
        gpuinfo = nullptr;   
    }
    if (cpu_output) 
    {        
        delete(cpu_output);
        cpu_output = nullptr;
    }
    if (!image.empty())
        image.release();
    if (!gpuimage.empty())
        gpuimage.release();
    if (!finalabels.empty())
        finalabels.clear();
    nlabels = 0;        
}

cudalabel::~cudalabel() 
{
    reset();
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
    //cv::Point minloc, maxloc;   
    if (!gpuimage.empty())
    {           
        cv::cuda::GpuMat localthres;
        localthres.create(gpuimage.size(), gpuimage.type());
        cv::cuda::minMax(gpuimage,&imin,&imax);        
        //imean = static_cast<unsigned int>(imin+((imax-imin)/2));    
        cv::cuda::threshold(gpuimage, localthres, 0, imax, cv::THRESH_BINARY);        
        cudaDeviceSynchronize();        
        localthres.copyTo(cv::cuda::GpuMat(gpuimage.size(), gpuimage.type(), d_img));
    }
    else if (!image.empty())    
    {        
        //cv::minMaxLoc(image,&imin,&imax,&minloc,&maxloc);
        //imean = static_cast<unsigned int>(imin+((imax-imin)/2));    
	    util::threshold(d_img, image.data, 0, npixel);                        
    }
    else
    {
        std::cerr << "No data available." << std::endl;
    }
}

void cudalabel::labelize()
{
    connectedComponentLabeling(d_labels, d_img, ncols, nrows);
    cudaDeviceSynchronize();
    nlabels = cudalabel_countool(d_labels, npixel, &finalabels);
    /* check if required 
	cv::Mat check(nrows, ncols, CV_8UC1, cv::Scalar::all(0));

	for (int i = 0; i < nrows; i++)
    {
		for(int j = 0; j< ncols; j++)
        {
			size_t idx = i * ncols + j;			
			check.at<uchar>(i,j) = d_labels[idx];			
		}
	}    
    cv::imshow("CHECKING",check);
    cv::waitKey(0);
    */
}

bool cudalabel::imgen()
{
    if (!gpuimage.empty())
    {
        cpu_output = new cv::Mat(gpuimage.size(),gpuimage.type());
        gpuimage.download(*cpu_output);        
        return(true);
    }
    else if (!image.empty())
    {
        cpu_output = new cv::Mat(image.size(),image.type());
        *cpu_output = image.clone();
        return(true);
    }
    else
    {
        return(false);
    }
}

void cudalabel::lsave(std::string outputname)
{    
    unsigned int totalarea = ncols*nrows;
    if (cpu_output->channels() == 1) 
    {    
        cv::Mat tmp;
        cv::cvtColor(*cpu_output, tmp, cv::COLOR_GRAY2BGR);
        *cpu_output = tmp.clone();
    }            
    for (unsigned int i = 0; i < nlabels; ++i) 
    {
        if (gpuinfo[i])
        {
            int x_min = gpuinfo[i][1];
            int x_max = gpuinfo[i][2];
            int y_min = gpuinfo[i][3];
            int y_max = gpuinfo[i][4];
            unsigned int area = (y_max-y_min)*(x_max-x_min);            
            if (area<0.8*totalarea)
                cv::rectangle(*cpu_output, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 255, 0), 2);            
        }        
    }    
    cv::imwrite(outputname,*cpu_output);        
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
    kgetinfo<<<grid, block>>>(d_labels, gpuinfo, nlabels, nrows, ncols);        
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }        
    return(gpuinfo);
}

unsigned int** cudalabel::get_clean_includes()
{
    std::vector<std::vector<unsigned int>> tmp; // Temporary storage for valid ROIs
    std::vector<bool> is_valid(nlabels, true);   // Boolean list to mark valid ROIs

    if (gpuinfo)
    {
        for (int i = 0; i < nlabels; ++i)
        {
            unsigned int roi_x0 = gpuinfo[i][1];
            unsigned int roi_x1 = gpuinfo[i][2];
            unsigned int roi_y0 = gpuinfo[i][3];
            unsigned int roi_y1 = gpuinfo[i][4];

            cv::Rect r_roi(roi_x0, roi_y0, roi_x1 - roi_x0 + 1, roi_y1 - roi_y0 + 1);

            // Check if r_roi is contained in any other ROI or contains another ROI
            for (int j = 0; j < nlabels; ++j)
            {
                if (i == j)
                    continue; // Skip comparison with itself

                unsigned int other_x0 = gpuinfo[j][1];
                unsigned int other_x1 = gpuinfo[j][2];
                unsigned int other_y0 = gpuinfo[j][3];
                unsigned int other_y1 = gpuinfo[j][4];

                cv::Rect r_other(other_x0, other_y0, other_x1 - other_x0 + 1, other_y1 - other_y0 + 1);

                // Check if r_roi is contained in r_other or if r_other is contained in r_roi
                bool r_roi_contains_other = r_roi.contains(cv::Point(other_x0, other_y0)) &&
                                            r_roi.contains(cv::Point(other_x1, other_y0)) &&
                                            r_roi.contains(cv::Point(other_x0, other_y1)) &&
                                            r_roi.contains(cv::Point(other_x1, other_y1));

                bool r_other_contains_roi = r_other.contains(cv::Point(roi_x0, roi_y0)) &&
                                             r_other.contains(cv::Point(roi_x1, roi_y0)) &&
                                             r_other.contains(cv::Point(roi_x0, roi_y1)) &&
                                             r_other.contains(cv::Point(roi_x1, roi_y1));

                // Mark as invalid if one contains the other
                if (r_roi_contains_other)
                {
                    is_valid[j] = false; // Mark the other as invalid if it is contained in r_roi
                    break; // No need to check further for this ROI
                }
                else if (r_other_contains_roi)
                {
                    is_valid[i] = false; // Mark r_roi as invalid if it contains the other
                    break; // No need to check further for this ROI
                }                
            }
        }

        // Now, insert only the valid ROIs into tmp
        for (int i = 0; i < nlabels; ++i)
        {
            if (is_valid[i])
            {
                unsigned int roi_x0 = gpuinfo[i][1];
                unsigned int roi_y0 = gpuinfo[i][2];
                unsigned int roi_x1 = gpuinfo[i][3];
                unsigned int roi_y1 = gpuinfo[i][4];

                tmp.push_back({gpuinfo[i][0], roi_x0, roi_y0, roi_x1, roi_y1});
            }
        }

        // Free the old gpuinfo memory
        for (int i = 0; i < nlabels; ++i)
        {
            cudaFree(gpuinfo[i]);
        }
        cudaFree(gpuinfo);
        gpuinfo = nullptr;

        // Allocate new gpuinfo based on filtered ROIs
        nlabels = tmp.size(); // Update label count
        if (nlabels > 0)
        {
            gpuinfo = (unsigned int**)malloc(nlabels * sizeof(unsigned int*));
            for (int i = 0; i < nlabels; ++i)
            {
                gpuinfo[i] = (unsigned int*)malloc(5 * sizeof(unsigned int));
                for (int j = 0; j < 5; ++j)
                {
                    gpuinfo[i][j] = tmp[i][j];
                }
            }
        }
        else
        {
            gpuinfo = nullptr;
        }
        return gpuinfo;
    }
    else
    {
        return nullptr;
    }
}