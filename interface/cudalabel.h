#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class cudalabel
{
    private:
        unsigned char *d_img;
        unsigned int *d_labels;
        int ncols,nrows,npixel;
        cv::Mat image;
        unsigned int imean;
        unsigned int nlabels;        
        cv::cuda::GpuMat *output;

    public:
        cudalabel();
        ~cudalabel();

        void setimg(const cv::Mat& input);
        void preprocess();
        void labelize();
        void lsave(std::string outputname);       
        
};
