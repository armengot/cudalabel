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
    unsigned int** labelinfo;
    unsigned int n;
    
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
    n = labels.lnumber();
    std::cout << "Num of labels: " << n << std::endl;
    labelinfo = labels.getinfo();
    
    for (unsigned int i = 1; i < n; ++i) 
    {
        if (labelinfo[i])
        {
            int x_min = labelinfo[i][1];
            int x_max = labelinfo[i][2];
            int y_min = labelinfo[i][3];
            int y_max = labelinfo[i][4];    
            cv::rectangle(image, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 0, 255), 2);    
            std::cout << "[main] LABEL[" << i << "] = [" << labelinfo[i][0] << "," << labelinfo[i][1] << "," << labelinfo[i][2] << "," << labelinfo[i][3] << "," << labelinfo[i][4] << "]" << std::endl;
                
        }
    }
    cv::imshow("labels", image);
    cv::waitKey(0);  

    labels.imgen();
    labels.lsave("output.png");
    
    return(0);
}
