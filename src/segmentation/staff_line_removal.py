from collections import Counter
import numpy as np
from skimage.morphology import binary_opening, binary_closing, binary_erosion, square
from skimage.filters import gaussian, median
from skimage.filters import threshold_otsu
from .rle import RLE
from utils import otsu

class StaffLineRemoval:
    """
    Class for removing staff lines from musical scores.
    """
    row_percentage = 0.3

    @staticmethod
    def calculate_thickness_and_spacing(rle_lengths, most_common):
        """
        Calculate the thickness and spacing of staff lines.

        Parameters:
        rle_lengths (list): RLE lengths.
        most_common (int): Most common pair sum.

        Returns:
        tuple: Line thickness, line spacing.
        """
        bw_patterns = [RLE.most_common_bw_pattern(col, most_common) for col in rle_lengths]
        bw_patterns = [x for x in bw_patterns if x]  # Filter empty patterns

        flattened = []
        for col in bw_patterns:
            flattened += col

        pair, count = Counter(flattened).most_common()[0]

        line_thickness = min(pair)
        line_spacing = max(pair)

        return line_thickness, line_spacing

    @staticmethod
    def whiten_rle(rle_lengths, rle_values, max_height):
        """
        Whiten the RLE by setting small black segments to white.

        Parameters:
        rle_lengths (list): RLE lengths.
        rle_values (list): RLE values.
        max_height (int): Maximum height for black segments.

        Returns:
        tuple: Whitened RLE lengths, values.
        """
        rle_whitened = []
        for length, value in zip(rle_lengths, rle_values):
            if value == 0 and length < 1.1 * max_height:
                value = 1
            rle_whitened.append((length, value))

        new_rle_lengths, new_rle_values = [], []
        count = 0
        for length, value in rle_whitened:
            if value == 1:
                count = count + length
            else:
                if count > 0:
                    new_rle_lengths.append(count)
                    new_rle_values.append(1)

                count = 0
                new_rle_lengths.append(length)
                new_rle_values.append(0)
        if count > 0:
            new_rle_lengths.append(count)
            new_rle_values.append(1)

        return new_rle_lengths, new_rle_values

    @staticmethod
    def remove_staff_lines(rle_lengths, rle_values, thickness, shape):
        """
        Remove staff lines from an image using RLE.

        Parameters:
        rle_lengths (list): RLE lengths.
        rle_values (list): RLE values.
        thickness (int): Line thickness.
        shape (tuple): Shape of the output image.

        Returns:
        numpy array: Image with staff lines removed.
        """
        new_rle_lengths, new_rle_values = [], []
        for i in range(len(rle_lengths)):
            whitened_lengths, whitened_values = StaffLineRemoval.whiten_rle(rle_lengths[i], rle_values[i], thickness)
            new_rle_lengths.append(whitened_lengths)
            new_rle_values.append(whitened_values)

        return RLE.hv_decode(new_rle_lengths, new_rle_values, shape)

    @staticmethod
    def remove_staff_lines_with_projection(thickness, img_with_staff):
        """
        Remove staff lines from an image using horizontal projection.

        Parameters:
        thickness (int): Line thickness.
        img_with_staff (numpy array): Input image with staff lines.

        Returns:
        numpy array: Image with staff lines removed.
        """
        img = img_with_staff.copy()
        projected = []
        rows, cols = img.shape
        for i in range(rows):
            proj_sum = 0
            for j in range(cols):
                proj_sum += img[i][j] == 1
            projected.append([1] * proj_sum + [0] * (cols - proj_sum))
            if proj_sum <= StaffLineRemoval.row_percentage * cols:
                img[i, :] = 1
        closed_img = binary_opening(img, np.ones((3 * thickness, 1)))
        return closed_img

    @staticmethod
    def get_staff_line_positions(start, most_common, thickness, spacing):
        """
        Get the positions of staff lines.

        Parameters:
        start (int): Starting position.
        most_common (int): Most common pair sum.
        thickness (int): Line thickness.
        spacing (int): Line spacing.

        Returns:
        list: Positions of staff lines.
        """
        rows = []
        num_lines = 6
        if start - most_common >= 0:
            start -= most_common
            num_lines = 7
        for k in range(num_lines):
            row = []
            for i in range(thickness):
                row.append(start)
                start += 1
            start += spacing
            rows.append(row)
        if len(rows) == 6:
            rows = [0] + rows
        return rows

    @staticmethod
    def horizontal_projection(image):
        """
        Perform horizontal projection on an image.

        Parameters:
        image (numpy array): Input image.

        Returns:
        int: Row index with the highest projection sum.
        """
        projected = []
        rows, cols = image.shape
        for i in range(rows):
            proj_sum = 0
            for j in range(cols):
                proj_sum += image[i][j] == 1
            projected.append([1] * proj_sum + [0] * (cols - proj_sum))
            if proj_sum <= 0.1 * cols:
                return i
        return 0

    @staticmethod
    def get_first_black_pixel_position(image):
        """
        Get the position of the first black pixel in an image.

        Parameters:
        image (numpy array): Input image.

        Returns:
        int: Row index of the first black pixel.
        """
        found = 0
        row_position = -1
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] == 0:
                    row_position = i
                    found = 1
                    break
            if found == 1:
                break
        return row_position

    @staticmethod
    def remove_staff_lines_from_image(binary_image, horizontal):
        """
        Remove staff lines from a binary image.

        Parameters:
        binary_image (numpy array): Input binary image.
        horizontal (bool): Whether to use horizontal projection.

        Returns:
        tuple: Line spacing, staff line positions, image with staff lines removed.
        """
        rle_lengths, rle_values = RLE.hv_encode(binary_image)
        most_common = RLE.get_most_common(rle_lengths)
        thickness, spacing = StaffLineRemoval.calculate_thickness_and_spacing(rle_lengths, most_common)
        start = 0
        if horizontal:
            no_staff_img = StaffLineRemoval.remove_staff_lines_with_projection(thickness, binary_image)
            staff_lines = otsu(binary_image - no_staff_img)
            start = StaffLineRemoval.horizontal_projection(binary_image)
        else:
            no_staff_img = StaffLineRemoval.remove_staff_lines(rle_lengths, rle_values, thickness, binary_image.shape)
            no_staff_img = binary_closing(no_staff_img, np.ones((thickness + 2, thickness + 2)))
            no_staff_img = median(no_staff_img)
            no_staff_img = binary_opening(no_staff_img, np.ones((thickness + 2, thickness + 2)))
            staff_lines = otsu(binary_image - no_staff_img)
            staff_lines = binary_erosion(staff_lines, np.ones((thickness + 2, thickness + 2)))
            staff_lines = median(staff_lines, footprint=square(21))
            start = StaffLineRemoval.get_first_black_pixel_position(staff_lines)
        staff_row_positions = StaffLineRemoval.get_staff_line_positions(start, most_common, thickness, spacing)
        staff_row_positions = [np.average(x) for x in staff_row_positions]
        return spacing, staff_row_positions, no_staff_img