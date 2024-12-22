
from utils import *

def detect_vertical_line_and_crop(image):
    """
    Detects a vertical line in the image and crops the image to remove all pixels
    to the left of the detected line.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The cropped image with pixels to the left of the detected vertical line removed.
    """
    # Read the image
    image_gray = color.rgb2gray(image)

    # Apply Otsu's thresholding
    threshold_value = filters.threshold_otsu(image_gray)
    binary_image = image_gray < threshold_value

    # Define a vertical kernel for morphological operations
    vertical_kernel = morphology.rectangle(int(image.shape[0] * 0.8), 1)

    # Detect vertical lines using morphological operations
    detected_lines = morphology.opening(binary_image, vertical_kernel)

    # Find contours of the detected lines
    contours, _ = cv2.findContours((detected_lines * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_line_x=0
    # Identify the vertical line that is 80% or more of the height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= 0.8 * image.shape[0]:
            vertical_line_x = x
            break

    # Crop the image to remove pixels to the left of the detected vertical line
    cropped_image = image[:, vertical_line_x:]

    return cropped_image


def getRefLengths(img):
    """
    Computes the most frequent lengths of connected components in the binary image.

    Args:
        img (numpy.ndarray): Binary image input.

    Returns:
        tuple: The most frequent pair of lengths (l0, l1) of the connected components.
    """
    cols = img.shape[1]
    rows = img.shape[0]
    hist = np.zeros((rows, rows), dtype=np.uint32)
    
    for i in range(cols):
        a = img[:, i]
        #starts identifies transitions from non-zero to zero (black to white).
        #ends identifies transitions from zero to non-zero (white to black).
        starts = np.where((a[:-1] != 0) & (a[1:] == 0))[0] + 2
        ends = np.where((a[:-1] == 0) & (a[1:] != 0))[0] + 2
        
        if a[0] == 0:
            starts = np.append(1, starts)
        if a[-1] == 0:
            ends = np.append(ends, a.size + 1)
        
        l0 = ends - starts
        starts_shifted = np.pad(starts[1:], (0, 1), mode='constant', constant_values=(a.size + 1))
        l1 = starts_shifted - (starts + l0)
        
        for j in range(len(starts)):
            hist[l0[j], l1[j]] += 1
    
    hist[:, 0] = 0
    max_value = np.max(hist)
    max_indices = np.where(hist == max_value)
    
    return max_indices[0][0], max_indices[1][0]



def getCandidateStaffs(binaryImg, staffHeight):
    """
    Filters out non-staff candidates from a binary image based on height thresholds.

    Args:
        binaryImg (numpy.ndarray): Binary image.
        staffHeight (int): Expected height of the staff.

    Returns:
        tuple: 
            - numpy.ndarray: Filtered binary image.
            - list: List of candidate staff tuples (column index, start row, height).
    """
    filteredImg = np.copy(binaryImg)
    candidates = []  
    cols = filteredImg.shape[1]
    rows = filteredImg.shape[0]
    upperLimitHeight = staffHeight + 2
    lowerLimitHeight = abs(staffHeight - 2)
    flag = False
    
    for i in range(cols):
        for j in range(rows):
            if filteredImg[j, i] == 0 and not flag:
                beg = j
                flag = True
            elif filteredImg[j, i] == 1 and flag:
                flag = False
                height = j - beg
                if height > upperLimitHeight or height < lowerLimitHeight:
                    filteredImg[beg:j, i] = 1
                else:
                    candidates.append((i, beg, height))
    
    return filteredImg, candidates


def removeLonelyStaffs(v, filteredImg, staffHeight, spaceHeight, eliminated):
    """
    Removes disconnected (lonely) staff candidates based on connectivity analysis.

    Args:
        v (list): List of staff candidates.
        filteredImg (numpy.ndarray): Filtered binary image.
        staffHeight (int): Expected height of the staff.
        spaceHeight (int): Space between staff lines.
        eliminated (list): List to store eliminated candidates.

    Returns:
        tuple:
            - numpy.ndarray: Updated binary image with lonely staff lines removed.
            - list: Updated list of staff candidates.
            - list: Updated list of eliminated candidates.
    """
    img = np.copy(filteredImg)
    cols = filteredImg.shape[1]
    rows = filteredImg.shape[0]
    staffs = np.ones(img.shape)

    for i in v.copy():
        verConnected = False
        horConnected = False

        col, begin, length = i

        if col - 1 >= 0 and np.any(filteredImg[begin:begin + length, col - 1] == 0):
            horConnected = True
        if col + 1 < cols and np.any(filteredImg[begin:begin + length, col + 1] == 0):
            horConnected = True

        start = max(begin - (spaceHeight + staffHeight), 0)
        end = min(rows, begin + length + spaceHeight + staffHeight)
        start2 = min(rows, begin + length + spaceHeight)
        if np.any(filteredImg[start:start + staffHeight, col] == 0) or np.any(filteredImg[start2:end, col] == 0):
            verConnected = True

        if not verConnected:
            img[begin:begin + length, col] = 1
            v.remove(i)
            eliminated.append(i)

    for i in v:
        staffs[i[1]:i[1] + i[2], i[0]] = 0

    return staffs, v, eliminated



def getLines(img, staffHeight, spaceHeight):
    """
    Detects staff lines in a binary image using peak detection on row sums.

    Args:
        img (numpy.ndarray): Binary image input.
        staffHeight (int): Expected height of the staff.
        spaceHeight (int): Space between staff lines.

    Returns:
        list: List of detected staff line indices.
    """
    cp = img.copy()

    kernel = np.ones((staffHeight, 1))
    dilate = cv2.dilate(cp, kernel)

    rows_sum = np.sum(dilate, axis=1)    
    lines, _ = find_peaks(rows_sum, height = 0.2*img.shape[1], distance=spaceHeight)

    return lines