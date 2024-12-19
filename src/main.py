from matplotlib import pyplot as plt
from preprocessing import DeskewProcessor, BinarizationProcessor, NoiseRemovalProcessor
import cv2
import numpy as np
from typing import Optional
import os 
class ImageProcessingPipeline:
    """
    Pipeline to process an image through multiple processors.
    """
    def __init__(self, binarization_method: str = 'otsu', deskew_processor: Optional[DeskewProcessor] = None):
        """
        Initialize the ImageProcessingPipeline.

        Args:
            binarization_method (str): Binarization method. Default is 'otsu'.
            deskew_processor (Optional[DeskewProcessor]): Deskew processor.
        """
        self.binarization_method = binarization_method
        self.deskew_processor = deskew_processor or DeskewProcessor()
        self.binarizer = BinarizationProcessor(method=binarization_method)
        self.noise_remover = NoiseRemovalProcessor()

    def process(self, image: np.ndarray) -> dict:
        """
        Process the image through the pipeline.

        Args:
            image (np.ndarray): Input image.

        Returns:
            dict: Dictionary containing the original, binarized, deskewed, and denoised images.
        """
        results = {
            'original_image': image,
            'binarized_image': None,
            'deskewed_image': None,
            'denoised_image': None,
        }
        results['binarized_image'] = self.binarizer.process(image)
        results['deskewed_image'] = self.deskew_processor.process(results['binarized_image'])
        results['denoised_image'] = self.noise_remover.process(results['deskewed_image'])
        return results

def main():
    """
    Main function to demonstrate the image processing pipeline.
    """
    _path = os.path.join(os.getcwd(), 'data', 'input')
    for file in os.listdir(_path):
        if file.endswith(".jpg"):
            image_path = os.path.join(_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Image file not found at {image_path}")
            pipeline = ImageProcessingPipeline()
            results = pipeline.process(image)
            img_path = image_path.replace("input", "output")
            img_path = img_path.replace(img_path.split("/")[-1], image_path.split("/")[-1].replace(".jpg", "_clean.jpg"))
            cv2.imwrite(img_path, (results['denoised_image']))
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