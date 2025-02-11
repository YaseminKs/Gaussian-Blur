// Utilizes all available CPU cores using Java’s ExecutorService.
// Divides image rows among threads for efficient parallel execution.
// Ensures proper synchronization to avoid conflicts.
// Scales with CPU performance—more cores mean faster execution.

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.imageio.ImageIO;

public class ParallelGaussianBlur{
    private static final float[][] KERNEL = {
        { 1/16f, 2/16f, 1/16f },
        { 2/16f, 4/16f, 2/16f },
        { 1/16f, 2/16f, 1/16f }
    };

    private static final int THREADS = Runtime.getRuntime().availableProcessors();

    public static BufferedImage applyGaussianBlur( BufferedImage image ){
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage result = new BufferedImage( width, height, BufferedImage.TYPE_INT_RGB );

        ExecutorService executor = Executors.newFixedThreadPool( THREADS );
        
        for( int t = 0 ; t < THREADS ; t++ ){
            final int threadID = t;
            executor.submit(() -> {
                int startY = ( height / THREADS ) * threadID;
                int endY = ( threadID == THREADS - 1 ) ? height : ( height / THREADS ) * ( threadID + 1 );

                for( int x = 1 ; x < width - 1 ; x++ ){
                    for( int y = startY + 1 ; y < endY - 1 ; y++ ){
                        float r = 0, g = 0, b = 0;
                        for( int i = -1 ; i <= 1 ; i++ ){
                            for( int j = -1 ; j <= 1 ; j++ ){
                                int rgb = image.getRGB( x + i, y + j );
                                int red = ( rgb >> 16 ) & 0xFF;
                                int green = ( rgb >> 8 ) & 0xFF;
                                int blue = rgb & 0xFF;

                                r += red * KERNEL[i + 1][j + 1];
                                g += green * KERNEL[i + 1][j + 1];
                                b += blue * KERNEL[i + 1][j + 1];
                            }
                        }
                        int newRgb = ( ( int ) r << 16 ) | ( ( int ) g << 8 ) | ( int ) b;
                        result.setRGB( x, y, newRgb );
                    }
                }
            });
        }

        executor.shutdown();
        while( !executor.isTerminated() ){}

        return result;
    }

    public static void main( String[] args ) throws Exception{
        BufferedImage img = ImageIO.read( new File( "input.jpg" ) );
        BufferedImage blurred = applyGaussianBlur( img );
        ImageIO.write( blurred, "jpg", new File( "output.jpg" ) );
        System.out.println( "Multi-threaded Gaussian blur applied!" );
    }
}
