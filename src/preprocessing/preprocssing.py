import numpy as np
import cv2
import logging
from typing import Union, Tuple, List, Optional
from abc import ABC, abstractmethod
from skimage.transform import probabilistic_hough_line, hough_line, rotate, hough_line_peaks
from skimage.feature import corner_harris, canny
from skimage import morphology, feature, transform
import skimage.io as io
from itertools import pairwise
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
# Binarization Processor
class BinarizationProcessor(BaseImageProcessor):
    """
    Processor to binarize an image.
    """
    def __init__(self, method: str = 'otsu'):
        """
        Initialize the BinarizationProcessor.

        Args:
            method (str): Binarization method. Options are 'otsu', 'mean', 'gaussian', 'niblack', 'sauvola'.
        """
        self.method = method.lower()
        self.method_map = {
            'otsu': self._otsu_threshold,
            'mean': self._adaptive_mean_threshold,
            'gaussian': self._adaptive_gaussian_threshold,
            'niblack': self._niblack_threshold,
            'sauvola': self._sauvola_threshold
        }

        if self.method not in self.method_map:
            raise ValueError(f"Invalid method '{self.method}'. Choose from 'otsu', 'mean', 'gaussian', 'niblack', 'sauvola'.")

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image to binarize it.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Binarized image.
        """
        self.validate_image(image)
        if image.dtype != np.uint8:
            image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
        
        img =  self.method_map[self.method](image)
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        return img

    def _otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        _, binarized_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized_image

    def _adaptive_mean_threshold(self, image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    def _adaptive_gaussian_threshold(self, image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def _niblack_threshold(self, image: np.ndarray) -> np.ndarray:
        from skimage.filters import threshold_niblack
        block_size = max(image.shape) // 20
        if block_size % 2 == 0:
            block_size += 1
        return (image > threshold_niblack(image, window_size=block_size, k=0.8)).astype(np.uint8) * 255

    def _sauvola_threshold(self, image: np.ndarray) -> np.ndarray:
        from skimage.filters import threshold_sauvola
        block_size = max(image.shape) // 20
        if block_size % 2 == 0:
            block_size += 1
        return (image > threshold_sauvola(image, window_size=block_size)).astype(np.uint8) * 255
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
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        Normalize the angle to be within the range [-90, 90] degrees.

        Args:
            angle (float): Input angle.

        Returns:
            float: Normalized angle.
        """
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        return angle
    def _crop_region(self, image:np.ndarray):
        """
        Crop the region of interest from the image.
        Args:
            image (np.ndarray): Input image.
        Returns:
            np.ndarray: Cropped image.
        """   
        x,y,w,h = cv2.boundingRect(cv2.findNonZero(image))
        return max(0, y - h//10), min(image.shape[0], y + h + h//10), max(0, x - w//20), min(image.shape[1], x + w + w//20)
    @staticmethod
    def _hough_lines(image: np.ndarray, rho: float=1.0) -> np.ndarray:
        """
        Detect lines in the image using Hough transform.
        Args:
            image (np.ndarray): Input image.
            rho (float): Distance resolution in pixels of the Hough grid.
        Returns:
            np.ndarray: Detected lines.
        """
        return  cv2.HoughLinesP(image, rho, np.pi / 180,np.min(image.shape) // 20 , np.array([]), np.max(image.shape) // 20, np.max(image.shape) // 200)
    @staticmethod
    def _merge_collinear_hull_segments(hull_points: np.ndarray, angle_threshold: float = 45) -> np.ndarray:
            """
            Merge collinear segments from hull points,
            Args:
                hull_points (np.ndarray): Convex hull points.
                angle_threshold (float): Maximum angle to consider a line as horizontal.

            Returns:
            np.ndarray of merged line segments.
            """
            hull_points = np.append(hull_points, [hull_points[0]], axis=0)
            lines = [[hull_points[0], hull_points[1]]]
            old_angle = DeskewProcessor._line_angle(np.array(lines[0]).flatten())
            for i in range(2, len(hull_points)):
                line = [hull_points[i - 1], hull_points[i]]
                new_angle = abs(DeskewProcessor._line_angle(np.array(line).flatten()))
                if abs(old_angle - new_angle) <= 45:
                    lines[-1][1] = hull_points[i]
                else:
                    lines.append(line)
                old_angle = abs(DeskewProcessor._line_angle(np.array(lines[-1]).flatten()))
            line_length = lambda x: DeskewProcessor._line_length(np.array(x).flatten())

            lines.sort(key=line_length, reverse=True)
            if len(lines) < 4:
                return np.array(lines)
            return DeskewProcessor._sort_boundary_lines(lines[:4])
    @staticmethod 
    def _hough_lines_peak(image: np.ndarray , rho: float = 1.0) -> np.ndarray:
        """
        Detect lines in the image using Hough transform.
        Args:
            image (np.ndarray): Input image.
            rho (float): Distance resolution in pixels of the Hough grid.
        Returns:
            np.ndarray: Detected lines.
        """
        return cv2.HoughLinesP(image, rho, np.pi / 180, np.min(image.shape) // 5, np.array([]), np.max(image.shape) // 4, np.max(image.shape) // 40)
    @staticmethod
    def _sort_boundary_lines(lines: np.ndarray) -> np.ndarray:
        """
        Sort boundary lines into left, top, right, bottom order based on their endpoints.
        
        Returns:
            np.ndarray: Sorted lines in order [left, top, right, bottom]
        """
        boundary_lines = np.array(lines)
        sums = boundary_lines.sum(axis=1).reshape((4, 2))
        indices = {
            'left': np.argmin(sums[:, 0]),
            'top': np.argmin(sums[:, 1]),
            'right': np.argmax(sums[:, 0]),
            'bottom': np.argmax(sums[:, 1])
        }

        return np.array([boundary_lines[indices[pos]] for pos in ['left', 'top', 'right', 'bottom']])
    @staticmethod
    def _draw_lines(image: np.ndarray, lines: np.ndarray, max_angle: int = 30, color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw lines on the image.
        Args:
            image (np.ndarray): Input image.
            lines (np.ndarray): Detected lines.
            max_angle (int): Maximum angle to consider a line as horizontal.
            color (Tuple[int, int, int]): Color for drawing lines.
            thickness (int): Thickness of the lines.
        Returns:
            np.ndarray: Image with lines drawn.
        """
        horizontal_lines = [x for x in lines if abs(DeskewProcessor._line_angle(x.flatten())) <= max_angle]
        img = np.zeros_like(image)
        for line in horizontal_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        return img
    @staticmethod
    def _hough_angle(lines: np.ndarray) -> float:
        """
        Calculate the rotation angle from the detected lines.
        Args:
            lines (np.ndarray): Detected lines.
        Returns:
            float: Rotation angle in degrees.
        """
        if lines is None:
            return 0.0
        lines_properties = np.array([(DeskewProcessor._line_length(x), DeskewProcessor._line_angle(x)) for x in lines[:, 0]], dtype=[('length', float), ('angle', float)])
        lines_properties = np.sort(lines_properties, order='length')[::-1]
        remove_outliers = lambda x: x[abs(x - np.mean(x)) <= 1.2 * np.std(x)]
        return np.median(remove_outliers(lines_properties[:10]['angle']))
    @staticmethod
    def _line_length(line: np.ndarray) -> float:
        return np.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
    @staticmethod
    def _line_angle(line: np.ndarray) -> float:
        return np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))
    def _rotate_image(self, img: np.ndarray, angle_in_degrees: float, interpolation_color: int = 0) -> np.ndarray:
        """
        Rotate the image by the given angle.
        Args:
            angle (float): Rotation angle in degrees.
            interpolation_color (int): Color for interpolation.

        Returns:
            np.ndarray: Rotated image.
        """
        (height, width) = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width //2, height // 2), angle_in_degrees, scale=1)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        rotation_matrix[0, 2] += (new_width / 2) - (width // 2)
        rotation_matrix[1, 2] += (new_height / 2) - (height // 2)
        return cv2.warpAffine(img, rotation_matrix, (new_width, new_height), flags=cv2.WARP_FILL_OUTLIERS,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=interpolation_color)
    @staticmethod 
    def _find_intersection_point(line: np.ndarray, next_line: np.ndarray) -> np.ndarray:
        """
        Find the intersection point of two lines.
        Args:
            line (np.ndarray): First line.
            next_line (np.ndarray): Second line.
        Returns:
            np.ndarray: Intersection point.
        """
        s1, e1 = line.reshape(2, 2).astype(float)
        s2, e2 = next_line.reshape(2, 2).astype(float)
        direction1, direction2 = e1 - s1, e2 - s2
        cross_product = np.cross(direction1, direction2)
        if np.isclose(cross_product, 0):
            raise ArithmeticError("Lines are parallel and do not intersect.")
        t1 = np.cross(s2 - s1, direction2) / cross_product
        return s1 + t1 * direction1
    @staticmethod
    def _find_boundary_points(bounding_lines: np.ndarray) -> np.ndarray:
        """
        Find the boundary points from the bounding lines.
        Args:
            bounding_lines (np.ndarray): Bounding lines.
        Returns:
            np.ndarray: Boundary points.
        """
        return np.array([
            DeskewProcessor._find_intersection_point(line, bounding_lines[(i + 1) % len(bounding_lines)])
            for i, line in enumerate(bounding_lines)
        ], dtype=np.float32)
    @staticmethod 
    def _transform_matrix(image: np.ndarray) -> np.ndarray:
        hough_lines = DeskewProcessor._hough_lines(image)
        img = DeskewProcessor._draw_lines(image, hough_lines)
        all_points = cv2.findNonZero(img)
        hull_points = cv2.convexHull(all_points)
        bounding_lines = DeskewProcessor._merge_collinear_hull_segments(hull_points)
        if bounding_lines.shape[0] < 4:
            return np.eye(3)
        try:
            bounding_points = DeskewProcessor._find_boundary_points(bounding_lines)
        except ArithmeticError:
            return np.eye(3)
        if DeskewProcessor._filter_outside_points(image, bounding_points):
            return np.eye(3)
        x, y, w, h = cv2.boundingRect(bounding_points)
        rectangle_points = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        return cv2.getPerspectiveTransform(bounding_points, rectangle_points)
            
    @staticmethod
    def _filter_outside_points(image: np.ndarray, points: np.ndarray) -> bool:
        """
        Filter out points that are outside the image.
        Args:
            image (np.ndarray): Input image.
            points (np.ndarray): Points to check.
        Returns:
            bool: True if any point is outside the image.
        """
        return any(
            point[0] < 0 or point[0] >= image.shape[1] or point[1] < 0 or point[1] >= image.shape[0]
            for point in points
        )
    @staticmethod
    def _calculate_deskew_angle(orient_angle: float, hough_angle: float) -> float:
        if abs(abs(hough_angle) - abs(orient_angle)) <= 5:
            return np.mean([orient_angle, hough_angle])
        deskew_angle = hough_angle 
        if (abs(abs(hough_angle) - abs(orient_angle)) >160):
            if abs(orient_angle) > abs(hough_angle) :
                deskew_angle = orient_angle
        return deskew_angle
    def process(self, img: np.ndarray) -> np.ndarray:
        """
        Process the image to deskew it.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Deskewed image.
        """
        self.validate_image(img)
        image = img.copy()
        image = cv2.bitwise_not(image)
        top,bottom,left,right = self._crop_region(image)
        image = image[top:bottom,left:right]
        img = img[top:bottom,left:right]
        hough_angle = self._hough_angle(self._hough_lines_peak(image))
        if abs(hough_angle) <= 2: 
            return img
        orient_angle = OrientationDetector._orientation_angle(image)
        angle = self._calculate_deskew_angle(orient_angle, hough_angle)
        print(f"Orientation Angle: {orient_angle} Hough Angle: {hough_angle} Deskew Angle: {angle}")
        if abs(angle) <= 2:
            return img
        image_rotated = self._rotate_image(image, angle)
        average_color = np.mean(img)
        orignal_rotated = self._rotate_image(img, angle, average_color)
        top, bottom, left, right = self._crop_region(image_rotated)
        image_rotated = image_rotated[top:bottom, left:right]
        orignal_rotated = orignal_rotated[top:bottom, left:right]
        (height, width) = image_rotated.shape[:2]
        return cv2.warpPerspective(orignal_rotated, self._transform_matrix(image_rotated), (width, height), flags=cv2.WARP_FILL_OUTLIERS, borderMode=cv2.BORDER_CONSTANT, borderValue=average_color)
         

#Noise Removal Processor 
class NoiseRemovalProcessor(BaseImageProcessor):
    """
    Processor to remove noise from an image using morphological operations.
    """
    def __init__(self, filter_size_ratio: float = 0.1, dilation_iterations: int = 15, connectivity: int = 8, min_component_ratio: float = 0.25):
        """
        Initialize the NoiseRemovalProcessor.

        Args:
            filter_size_ratio (float): Ratio to determine the filter size for dilation.
            dilation_iterations (int): Number of iterations for the dilation process.
            connectivity (int): Connectivity for connected components analysis.
            min_component_ratio (float): Minimum ratio of component size to the largest component size to keep.
        """
        self.filter_size_ratio = filter_size_ratio
        self.dilation_iterations = dilation_iterations
        self.connectivity = connectivity
        self.min_component_ratio = min_component_ratio

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image to remove noise.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with noise removed.
        """
        self.validate_image(image)
        return self.remove_noise(image)

    def remove_noise(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Remove noise from a binary image.

        Args:
            binary_image (np.ndarray): Input binary image.

        Returns:
            np.ndarray: Denoised binary image.
        """
        filter_size = int(np.mean(binary_image.shape) * self.filter_size_ratio)
        filter_size = filter_size + 1 if (filter_size % 2 == 0) else filter_size
        img = cv2.dilate(binary_image, np.ones((filter_size, filter_size), np.uint8), iterations=self.dilation_iterations)
        n, img = cv2.connectedComponents(img, connectivity=self.connectivity, ltype=cv2.CV_16U)
        unique, count = np.unique(img, return_counts=True)
        if len(count) <= 1:
            return binary_image
        max_component_index = np.argmax(count[1:]) + 1
        max_elements = []
        for i in range(1, n):
            component_ratio = count[i] / count[max_component_index]
            if component_ratio >= self.min_component_ratio:
                max_elements.append(unique[i])

        img[np.isin(img, max_elements)] = 1 << 15
        binary_image[img != 1 << 15] = 0
        return binary_image

# Orientation Detector
class OrientationDetector(BaseImageProcessor):
    """
    Processor to detect the orientation of an image.
    """
    def __init__(self):
        """
        Initialize the OrientationDetector.
        """
    def is_horizontal(self, image: np.ndarray) -> bool:
        """
        Check if the image is horizontal.

        Args:
            image (np.ndarray): Input image.

        Returns:
            bool: True if the image is horizontal, False otherwise.
        """
        self.validate_image(image)
        angle = self._orientation_angle(image)
        return abs(angle) <=2 
    def process(self, image: np.ndarray) -> bool:
        """
        Process the image to detect its orientation.

        Args:
            image (np.ndarray): Input image.

        Returns:
            bool: True if the image is horizontal, False otherwise.
        """
        return self.is_horizontal(image)
    @staticmethod
    def _orientation_angle(image: np.ndarray) -> float:
        """
        Calculate the orientation angle of the image.
        using the minimum area rectangle method.
        which is the angle of the minimum bounding rectangle.
        Args:
            image (np.ndarray): Input image.

        Returns:
            float: Orientation angle in degrees.
        """
        all_points = cv2.findNonZero(image)
        center, (width,height), angle = cv2.minAreaRect(all_points)
        if width < height:
            angle += 90
        return angle

