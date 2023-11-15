#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>


#include <cudalabel.h>
#include <CCL.cuh>
#include <utils.hpp>

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
        delete(output);
}

void cudalabel::setimg(const cv::Mat& input) 
{
    ncols = input.cols;
    nrows = input.rows;
    npixel = nrows*ncols;
    
    cudaMallocManaged(&d_labels, npixel * sizeof(int));
    cudaMallocManaged(&d_img, npixel * sizeof(char));
    image = input.clone();
}

void cudalabel::preprocess()
{
    imean = util::mean(image.data, npixel);
	util::threshold(d_img, image.data, imean, npixel);
}

void cudalabel::labelize()
{
    connectedComponentLabeling(d_labels, d_img, ncols, nrows);
    nlabels = util::countComponents(d_labels, npixel);
    output = new cv::cuda::GpuMat(nrows, ncols, CV_32SC1, d_labels);
}

void cudalabel::lsave(std::string outputname)
{
    cv::Mat result;
    output->download(result);
    cv::imwrite(outputname,result);
}
