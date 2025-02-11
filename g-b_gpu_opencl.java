// firstly, download jocl from jogamp.org and include the JAR and native libraries in your Java project

import org.jocl.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.jocl.CL.*;

public class OpenCLGaussianBlur{

    private static final String KERNEL_SOURCE =
        "__kernel void gaussian_blur( __global uchar *input, __global uchar *output, int width, int height ){"
        + "    int x = get_global_id( 0 );"
        + "    int y = get_global_id( 1 );"
        + "    if( x < 1 || y < 1 || x >= width - 1 || y >= height - 1 )"
        + "        return;"
        + "    float kernel[3][3] = {"
        + "        { 1/16.0, 2/16.0, 1/16.0 },"
        + "        { 2/16.0, 4/16.0, 2/16.0 },"
        + "        { 1/16.0, 2/16.0, 1/16.0 }"
        + "    };"
        + "    float r = 0, g = 0, b = 0;"
        + "    for( int i = -1 ; i <= 1 ; i++ ){"
        + "        for( int j = -1 ; j <= 1 ; j++ ){"
        + "            int index = ( ( y + j ) * width + ( x + i ) ) * 3;"
        + "            r += input[index] * kernel[i + 1][j + 1];"
        + "            g += input[index + 1] * kernel[i + 1][j + 1];"
        + "            b += input[index + 2] * kernel[i + 1][j + 1];"
        + "        }"
        + "    }"
        + "    int idx = ( y * width + x ) * 3;"
        + "    output[idx] = ( uchar ) r;"
        + "    output[idx + 1] = ( uchar ) g;"
        + "    output[idx + 2] = ( uchar ) b;"
        + "}";

    public static void main( String[] args ) throws Exception{
        // Load image
        BufferedImage image = ImageIO.read( new File( "input.jpg" ) );
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] imageData = extractRGB( image );

        // Initialize OpenCL
        CL.setExceptionsEnabled( true );
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs( platforms.length, platforms, null );
        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, devices.length, devices, null );
        cl_context context = clCreateContext( null, devices.length, devices, null, null, null );
        cl_command_queue commandQueue = clCreateCommandQueue( context, devices[0], 0, null );

        // Create memory buffers
        cl_mem inputBuffer = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_uchar * imageData.length, Pointer.to( imageData ), null );
        cl_mem outputBuffer = clCreateBuffer( context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_uchar * imageData.length, null, null );

        // Compile OpenCL kernel
        cl_program program = clCreateProgramWithSource( context, 1, new String[]{ KERNEL_SOURCE }, null, null );
        clBuildProgram( program, 0, null, null, null, null );
        cl_kernel kernel = clCreateKernel( program, "gaussian_blur", null );

        // Set kernel arguments
        clSetKernelArg( kernel, 0, Sizeof.cl_mem, Pointer.to( inputBuffer ) );
        clSetKernelArg( kernel, 1, Sizeof.cl_mem, Pointer.to( outputBuffer ) );
        clSetKernelArg( kernel, 2, Sizeof.cl_int, Pointer.to( new int[]{ width } ) );
        clSetKernelArg( kernel, 3, Sizeof.cl_int, Pointer.to( new int[]{ height } ) );

        // Execute kernel
        long[] globalWorkSize = new long[]{ width, height };
        clEnqueueNDRangeKernel( commandQueue, kernel, 2, null, globalWorkSize, null, 0, null, null );

        // Read the results
        byte[] outputData = new byte[imageData.length];
        clEnqueueReadBuffer( commandQueue, outputBuffer, CL_TRUE, 0,
                outputData.length * Sizeof.cl_uchar, Pointer.to( outputData ), 0, null, null );

        // Save the output image
        BufferedImage outputImage = createImage( outputData, width, height );
        ImageIO.write( outputImage, "jpg", new File( "output.jpg" ) );

        // Cleanup
        clReleaseMemObject( inputBuffer );
        clReleaseMemObject( outputBuffer );
        clReleaseKernel( kernel );
        clReleaseProgram( program );
        clReleaseCommandQueue( commandQueue );
        clReleaseContext( context );

        System.out.println( "GPU Gaussian Blur applied with OpenCL!" );
    }

    private static byte[] extractRGB( BufferedImage image ){
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] pixels = new byte[width * height * 3];

        for( int y = 0 ; y < height ; y++ ){
            for( int x = 0 ; x < width ; x++ ){
                int rgb = image.getRGB( x, y );
                int idx = ( y * width + x ) * 3;
                pixels[idx] = ( byte ) ( ( rgb >> 16 ) & 0xFF );
                pixels[idx + 1] = ( byte ) ( ( rgb >> 8 ) & 0xFF );
                pixels[idx + 2] = ( byte ) ( rgb & 0xFF );
            }
        }
        return pixels;
    }

    private static BufferedImage createImage( byte[] pixels, int width, int height ){
        BufferedImage image = new BufferedImage( width, height, BufferedImage.TYPE_INT_RGB );
        for( int y = 0 ; y < height ; y++ ){
            for( int x = 0 ; x < width ; x++ ){
                int idx = ( y * width + x ) * 3;
                int rgb = ( ( pixels[idx] & 0xFF ) << 16 ) | ( ( pixels[idx + 1] & 0xFF ) << 8 ) | ( pixels[idx + 2] & 0xFF );
                image.setRGB( x, y, rgb );
            }
        }
        return image;
    }
}


// Runs on GPU
// Much faster than CPU
// Uses OpenCL (cross-platform GPU support)
