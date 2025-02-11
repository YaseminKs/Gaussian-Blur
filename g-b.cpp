// using opencv

#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    cv::Mat image = cv::imread( "input.jpg" );
    if( image.empty() ){
        std::cout << "Error: Could not load image!" << std::endl;
        return -1;
    }
    
    cv::Mat blurred;
    cv::GaussianBlur( image, blurred, cv::Size( 3, 3 ), 0 );
    cv::imwrite( "output.jpg", blurred );
    
    std::cout << "Gaussian blur applied!" << std::endl;
    return 0;
}
