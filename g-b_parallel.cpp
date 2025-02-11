#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

void applyGaussianBlurOpenMP( cv::Mat &image, cv::Mat &output ){
    int kernel[3][3] = { { 1, 2, 1 }, { 2, 4, 2 }, { 1, 2, 1 } };
    int kernelSum = 16;

    #pragma omp parallel for collapse( 2)
    for( int x = 1 ; x < image.rows - 1 ; x++ ){
        for( int y = 1 ; y < image.cols - 1 ; y++ ){
            cv::Vec3b pixel( 0, 0, 0 );
            for( int i = -1 ; i <= 1 ; i++ ){
                for( int j = -1 ; j <= 1 ; j++ ){
                    cv::Vec3b temp = image.at<cv::Vec3b>( x + i, y + j );
                    pixel[0] += temp[0] * kernel[i + 1][j + 1];
                    pixel[1] += temp[1] * kernel[i + 1][j + 1];
                    pixel[2] += temp[2] * kernel[i + 1][j + 1];
                }
            }
            output.at<cv::Vec3b>( x, y ) = pixel / kernelSum;
        }
    }
}

int main(){
    cv::Mat image = cv::imread( "input.jpg" );
    if( image.empty() ){
        std::cout << "Error loading image!" << std::endl;
        return -1;
    }

    cv::Mat output(image.size(), image.type() );
    applyGaussianBlurOpenMP( image, output );
    cv::imwrite( "output.jpg", output );

    std::cout << "Gaussian blur applied using OpenMP!" << std::endl;
    return 0;
}
