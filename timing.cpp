#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

#include <cudalabel.h>

int main(int argc, char **argv) 
{
    std::vector<std::string> filenames;
    std::vector<std::string> outnames;
    const char flag[2][4] = {"CPU", "GPU"};
    int twice = 0;

    for (int i = 0; i < 10; ++i)
    {
        std::string filename = "../samples/sample" + std::to_string(i) + ".jpg";
        std::string outname = "output" + std::to_string(i) + ".png";
        filenames.push_back(filename);
        outnames.push_back(outname);
    }
   
    cudalabel labels;

    while (twice<2)
    {
        for(unsigned int i=0;i<filenames.size();i++)
        {   
            labels.reset();

            cv::Mat image = cv::imread(filenames[i], cv::IMREAD_GRAYSCALE);
            cv::cuda::GpuMat gpuimg;

            if (!image.empty())
                gpuimg.upload(image);
            
            auto start_time = std::chrono::high_resolution_clock::now();        
            /* load */
            if (!twice)
                labels.setimg(image);       // CPU
            else
                labels.setgpuimg(gpuimg);   // GPU
            /* steps */
            labels.preprocess();
            labels.labelize();
            labels.getinfo();
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            
            if (labels.imgen())
                labels.lsave(outnames[i]);       

            std::cout << "[" << flag[twice] << "] processing image [" << image.cols << "x" << image.rows << "] with " << filenames[i] << " took " << duration << " milliseconds." << std::endl;
            
        }
        std::cout << std::endl;
        twice++;
    }

    return 0;
}
