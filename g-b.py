# using opencv

import cv2

def gaussian_blur( image_path, output_path, kernel_size=( 3,3 ) ):
    image = cv2.imread( image_path )
    blurred = cv2.GaussianBlur( image, kernel_size, 0 )
    cv2.imwrite( output_path, blurred )

gaussian_blur( "input.jpg", "output.jpg" )
print( "Gaussian blur applied!" )
