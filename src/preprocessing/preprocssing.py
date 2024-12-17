import numpy as np
import cv2
import logging
from typing import Union, Tuple, List, Optional
from abc import ABC, abstractmethod
from skimage.transform import probabilistic_hough_line, hough_line, rotate, hough_line_peaks
from skimage.feature import corner_harris, canny

# Logging Configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base Image Processing Abstract Class
class BaseImageProcessor(ABC):
    """
    Abstract base class for image processing.
    """
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Abstract method to process an image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Processed image.
        """
        pass

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate input image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            bool: True if the image is valid.

        Raises:
            ValueError: If the image is None or has an invalid shape.
        """
        if image is None:
            raise ValueError("Image cannot be None")
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image shape: {image.shape}")
        return True

# Deskew Processor
class DeskewProcessor(BaseImageProcessor):
    """
    Processor to deskew an image.
    """
    def __init__(self, 
                 low_threshold: int = 50, 
                 high_threshold: int = 150, 
                 sigma: float = 1.065):
        """
        Initialize the DeskewProcessor.

        Args:
            low_threshold (int): Low threshold for Canny edge detection.
            high_threshold (int): High threshold for Canny edge detection.
            sigma (float): Sigma for Gaussian filter in Canny edge detection.
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.sigma = sigma

    def calculate_rotation_angle(self, image: np.ndarray) -> float:
        """
        Calculate the rotation angle to deskew the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            float: Rotation angle in degrees.
        """
        self.validate_image(image)
        
        edges = canny(image, 
                      low_threshold=self.low_threshold, 
                      high_threshold=self.high_threshold, 
                      sigma=self.sigma)
        
        harris = corner_harris(edges)
        
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
        h, theta, d = hough_line(harris, theta=tested_angles)
        
        out, angles, _ = hough_line_peaks(h, theta, d)
        
        rotation_number = np.average(np.degrees(angles))
        if rotation_number <= 20 and rotation_number >= -20:
            rotation_number = 0
        if rotation_number < 45 and rotation_number != 0:
            rotation_number += 90
        
        return rotation_number

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image to deskew it.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Deskewed image.
        """
        angle = self.calculate_rotation_angle(image)
        return rotate(image, angle, resize=True, mode='edge')



# Image Cropping Processor
class ImageCropper(BaseImageProcessor):
    """
    Processor to crop significant sections of an image.
    """
    def __init__(self, threshold_ratio: float = 0.002, sections: int = 50):
        """
        Initialize the ImageCropper.

        Args:
            threshold_ratio (float): Ratio of black pixels to consider a section significant.
            sections (int): Number of sections to divide the image into.
        """
        self.threshold_ratio = threshold_ratio
        self.sections = sections

    def _get_significant_sections(self, image: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Get significant sections of the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple[List[int], List[int]]: Lists of significant row and column indices.
        """
        rows_sections, cols_sections = [], []
        
        for x in range(self.sections):
            start_row = x * image.shape[0] // self.sections
            end_row = (x + 1) * image.shape[0] // self.sections
            section = image[start_row:end_row, :]
            black_pixel_count = np.sum(section == 0)
            section_area = section.shape[0] * section.shape[1]
            
            if black_pixel_count >= self.threshold_ratio * section_area:
                rows_sections.append(start_row)
        
        for x in range(self.sections):
            start_col = x * image.shape[1] // self.sections
            end_col = (x + 1) * image.shape[1] // self.sections
            section = image[:, start_col:end_col]
            
            black_pixel_count = np.sum(section == 0)
            section_area = section.shape[0] * section.shape[1]
            
            if black_pixel_count >= self.threshold_ratio * section_area:
                cols_sections.append(start_col)
        
        return rows_sections, cols_sections

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image to crop significant sections.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Cropped image.
        """
        self.validate_image(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, .4, 1, cv2.THRESH_BINARY)
        rows_sections, cols_sections = self._get_significant_sections(image)
        
        if not rows_sections or not cols_sections:
            logger.warning("No significant sections found. Returning original image.")
            return image
        
        new_img = image[
            rows_sections[0]:min(image.shape[0], rows_sections[-1] + image.shape[0] // self.sections),
            cols_sections[0]:min(image.shape[1], cols_sections[-1] + image.shape[1] // self.sections)
        ]
        
        return new_img

# Orientation Detector
class OrientationDetector(BaseImageProcessor):
    """
    Processor to detect the orientation of an image.
    """
    def __init__(self, horizontal_threshold: float = 0.9):
        """
        Initialize the OrientationDetector.

        Args:
            horizontal_threshold (float): Threshold ratio of black pixels to consider the image horizontal.
        """
        self.horizontal_threshold = horizontal_threshold

    def is_horizontal(self, image: np.ndarray) -> bool:
        """
        Check if the image is horizontal.

        Args:
            image (np.ndarray): Input image.

        Returns:
            bool: True if the image is horizontal, False otherwise.
        """
        self.validate_image(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows, cols = image.shape
        for i in range(rows):
            black_pixel_count = np.sum(image[i, :] == 0)
            
            if black_pixel_count >= self.horizontal_threshold * cols:
                return True
        
        return False

    def process(self, image: np.ndarray) -> bool:
        """
        Process the image to detect its orientation.

        Args:
            image (np.ndarray): Input image.

        Returns:
            bool: True if the image is horizontal, False otherwise.
        """
        return self.is_horizontal(image)

