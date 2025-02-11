#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define BLOCK_SIZE 16
__constant__ float d_kernel[9] = { 1/16.0f, 2/16.0f, 1/16.0f, 2/16.0f, 4/16.0f, 2/16.0f, 1/16.0f, 2/16.0f, 1/16.0f };

__global__ void gaussianBlurKernel( unsigned char *input, unsigned char *output, int width, int height, int channels ){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= 1 && y >= 1 && x < width - 1 && y < height - 1 ){
        for( int c = 0 ; c < channels ; c++ ){
            float sum = 0.0f;
            for( int i = -1 ; i <= 1 ; i++ ){
                for( int j = -1 ; j <= 1 ; j++ ){
                    int idx = ( ( y + i ) * width + ( x + j ) ) * channels + c;
                    sum += input[idx] * d_kernel[( i + 1 ) * 3 + ( j + 1 )];
                }
            }
            output[( y * width + x ) * channels + c] = ( unsigned char )sum;
        }
    }
}

void applyGaussianBlurCUDA( cv::Mat &image, cv::Mat &output ){
    int imgSize = image.rows * image.cols * image.channels();
    unsigned char *d_input, *d_output;

    cudaMalloc( ( void** )&d_input, imgSize );
    cudaMalloc( ( void** )&d_output, imgSize );
    cudaMemcpy( d_input, image.data, imgSize, cudaMemcpyHostToDevice );

    dim3 blockSize( BLOCK_SIZE, BLOCK_SIZE );
    dim3 gridSize( ( image.cols + BLOCK_SIZE - 1 ) / BLOCK_SIZE, ( image.rows + BLOCK_SIZE - 1 ) / BLOCK_SIZE );
    
    gaussianBlurKernel<<<gridSize, blockSize>>>( d_input, d_output, image.cols, image.rows, image.channels() );

    cudaMemcpy( output.data, d_output, imgSize, cudaMemcpyDeviceToHost );

    cudaFree( d_input );
    cudaFree( d_output );
}

int main(){
    cv::Mat image = cv::imread( "input.jpg" );
    if( image.empty() ){
        std::cout << "Error loading image!" << std::endl;
        return -1;
    }

    cv::Mat output( image.size(), image.type() );
    applyGaussianBlurCUDA( image, output );
    cv::imwrite( "output.jpg", output );

    std::cout << "Gaussian blur applied using CUDA!" << std::endl;
    return 0;
}
