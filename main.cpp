#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cudalabel.h>

int main(int argc, char **argv) 
{
    std::string filename;
    cudalabel labels;
    
    if (argc < 2) 
    {
        std::cout << "Usage: " << argv[0] << " <image file>" << std::endl;
        return (-1);
    }
    filename = argv[1];

    // Read image
    cv::Mat image;
    image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (!image.data) 
    {
        std::cerr << "Couldn't open file" << std::endl;
        return (-1);
    }
    if (!image.isContinuous()) 
    {
        std::cerr << "Image is not allocated with continuous data. Exiting..." << std::endl;
        return (-1);
    }
    labels.setimg(image);
    labels.preprocess();
    labels.labelize();
    labels.lsave("output.png");
    
    return(0);
}
