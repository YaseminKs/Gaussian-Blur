import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class GaussianBlur{
    private static final float[][] KERNEL = {
        { 1/16f, 2/16f, 1/16f },
        { 2/16f, 4/16f, 2/16f },
        { 1/16f, 2/16f, 1/16f }
    };

    public static BufferedImage applyGaussianBlur( BufferedImage image ){
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage result = new BufferedImage( width, height, BufferedImage.TYPE_INT_RGB );

        for( int x = 1; x < width - 1; x++) {
            for( int y = 1 ; y < height - 1 ; y++ ){
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
        return result;
    }

    public static void main( String[] args ) throws Exception{
        BufferedImage img = ImageIO.read( new File( "input.jpg" ) );
        BufferedImage blurred = applyGaussianBlur( img );
        ImageIO.write( blurred, "jpg", new File( "output.jpg" ) );
        System.out.println( "Gaussian blur applied!" );
    }
}
