from matplotlib import pyplot as plt
from preprocessing import DeskewProcessor, ImageCropper, OrientationDetector
import cv2
import numpy as np
from typing import Optional

class ImageProcessingPipeline:
    """
    Pipeline to process an image through multiple processors.
    """
    def __init__(self, 
                 deskew_processor: Optional[DeskewProcessor] = None,
                 cropper: Optional[ImageCropper] = None,
                 orientation_detector: Optional[OrientationDetector] = None):
        """
        Initialize the ImageProcessingPipeline.

        Args:
            deskew_processor (Optional[DeskewProcessor]): Deskew processor.
            cropper (Optional[ImageCropper]): Image cropper.
            orientation_detector (Optional[OrientationDetector]): Orientation detector.
        """
        self.deskew_processor = deskew_processor or DeskewProcessor()
        self.cropper = cropper or ImageCropper()
        self.orientation_detector = orientation_detector or OrientationDetector()

    def process(self, image: np.ndarray) -> dict:
        """
        Process the image through the pipeline.

        Args:
            image (np.ndarray): Input image.

        Returns:
            dict: Dictionary containing the original, deskewed, cropped images and orientation status.
        """
        results = {
            'original_image': image,
            'deskewed_image': None,
            'cropped_image': None,
            'is_horizontal': None
        }
        results['is_horizontal'] = self.orientation_detector.process(image) 
        if not results['is_horizontal']:
            results['deskewed_image'] = self.deskew_processor.process(image)
            results['cropped_image'] = self.cropper.process(results['deskewed_image'])
        else:
            results['cropped_image'] = self.cropper.process(image)
        return results

def main():
    """
    Main function to demonstrate the image processing pipeline.
    """
    image_path = r'B:\Last Year\image\project\Image-Project\data\input\test1.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    

    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    pipeline = ImageProcessingPipeline()
    
    results = pipeline.process(image)
    
    print("Deskew Angle:", pipeline.deskew_processor.calculate_rotation_angle(image))
    print("Is Horizontal:", results['is_horizontal'])
    
    # if results['deskewed_image'] is not None:
    #     cv2.imshow('Deskewed Image', results['deskewed_image'])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    if results['cropped_image'] is not None:
        cv2.imshow('Cropped Image', results['cropped_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# import skimage.io as io
# from utils import gray_img, otsu, show_images
# from segmentation.staff_line_removal import StaffLineRemoval

# img = io.imread('./data/input/02.PNG')

# # Convert to grayscale
# gray = gray_img(img)

# # Threshold the image
# binary_img = otsu(gray)

# # Remove staff lines (horizontal is True for typical sheet music)
# spacing, staff_row_positions, no_staff_img = StaffLineRemoval.remove_staff_lines_from_image(binary_img, horizontal=True)

# show_images([img, gray, binary_img, no_staff_img],
#             ['Original', 'Grayscale', 'Binary', 'No Staff Lines'])