#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>

class cudalabel
{
    private:
        // needed from folkev sources
        unsigned char* d_img = nullptr;
        unsigned int* d_labels = nullptr;
        // image features
        int ncols,nrows,npixel;
        // alternative input CPU/GPU (dev/test)
        cv::Mat image;
        cv::cuda::GpuMat gpuimage;
        // output data
        unsigned int imean;
        unsigned int nlabels;                
        unsigned int** gpuinfo = nullptr;
        std::set<unsigned int> finalabels;

    public:
        cudalabel();
        ~cudalabel();
        /* (1) */
        void setimg(const cv::Mat input);
        void setgpuimg(const cv::cuda::GpuMat input);
        /* (2) */
        void preprocess();
        /* (3) */
        void labelize();
        /* (4) */
        unsigned int** getinfo();
        /* optional tools */
        void reset();
        bool imgen();
        void lsave(std::string outputname);
        unsigned int lnumber();
        unsigned int lmean();        
        cv::Mat *cpu_output = nullptr;
        
        
};
