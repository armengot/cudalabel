#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cudalabel.h>

void show_result(cv::Mat image, unsigned int** info, unsigned int n)
{
    // Convert grayscale to color if necessary
    if (image.channels() == 1)
    {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    // Iterate through all ROIs
    for (unsigned int i = 0; i < n; ++i)
    {
        if (info[i])
        {
            int x_min = info[i][1];
            int x_max = info[i][2];
            int y_min = info[i][3];
            int y_max = info[i][4];

            // Draw the rectangle on the image
            cv::rectangle(image, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 255, 0), 2);

            // Print the label information
            std::cout << "[main] LABEL[" << i << "] = [" << info[i][0] << "," << info[i][1] << "," << info[i][2] << "," << info[i][3] << "," << info[i][4] << "]" << std::endl;

            // Draw the index number below and to the right of (x_min, y_min)
            cv::Point text_pos(x_min + 10, y_min + 30); // Position slightly below and to the right of the top-left corner
            cv::putText(image, std::to_string(i), text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
    }

    // Display the image with drawn rectangles and indices
    cv::imshow("labels", image);
    cv::waitKey(0);    
}

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
    image = cv::imread(filename, cv::IMREAD_UNCHANGED);
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

    double input_min, input_max;
    cv::minMaxLoc(image, &input_min, &input_max);
    std::cout << "[main] Image loaded   " << filename << " type: " << image.type() << " channels number " << image.channels() << std::endl;
    std::cout << "[main] Image loaded   " << filename << " size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "[main] Image loaded   " << filename << "  min: " << input_min << " max: " << input_max << std::endl;
    if (image.channels() > 1) 
    {
        //cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);        
        std::vector<cv::Mat> channels;
        cv::split(image, channels);        
        image = channels[0].clone();
    }    
    if (image.type() != CV_8U) 
    {
        image.convertTo(image, CV_8U);
    }    
    std::cout << "[main] Image prepared " << filename << " type: " << image.type() << " channels number " << image.channels() << std::endl;
    std::cout << "[main] Image prepared " << filename << " size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "[main] Image prepared " << filename << "  min: " << input_min << " max: " << input_max << std::endl;    

    cv::cuda::GpuMat imgpu;
    imgpu.upload(image);

    //labels.setimg(image); // (or CPU)
    labels.setgpuimg(imgpu); // or GPU    
    labels.preprocess();    
    labels.labelize();
    n = labels.lnumber();
    std::cout << "[main] Num of labels: " << n << " computed with threshold = " << labels.lmean() << std::endl;
    
    // first
    labelinfo = labels.getinfo();
    show_result(image,labelinfo,n);
    
    // clean includes
    labelinfo = labels.get_clean_includes();
    n = labels.lnumber();    
    show_result(image,labelinfo,n);   

    // output
    labels.imgen();
    labels.lsave("output.png");    
    return(0);
}
