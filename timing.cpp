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

    for (int i = 0; i < 10; ++i)
    {
        std::string filename = "../samples/sample" + std::to_string(i) + ".jpg";
        std::string outname = "output" + std::to_string(i) + ".png";
        filenames.push_back(filename);
        outnames.push_back(outname);
    }

    cudalabel labels;

    for(unsigned int i=0;i<filenames.size();i++)
    {
        cv::Mat image = cv::imread(filenames[i], cv::IMREAD_GRAYSCALE);
        
        // upload -> GPU
        labels.setimg(image);

        // time in GPU
        auto start_time = std::chrono::high_resolution_clock::now();        
        labels.preprocess();
        labels.labelize();
        labels.getinfo();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // download
        labels.imgen();
        labels.lsave(outnames[i]);
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cout << "Processing image " << filenames[i] << " took " << duration << " milliseconds." << std::endl;
    }

    return 0;
}
