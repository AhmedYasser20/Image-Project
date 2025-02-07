from utils import *


def mergeContours(staffless, boundingRects, staffHeight, spaceHeight):
    """
    Merges bounding rectangles of objects in an image based on proximity and size constraints.

    Args:
        staffless (ndarray): Binary image with staffs removed.
        boundingRects (list of tuples): List of bounding rectangles represented as (x, y, w, h).
        staffHeight (int): Height of the staff lines in the image.
        spaceHeight (int): Height of the space between staff lines.

    Returns:
        list of tuples: Merged bounding rectangles represented as (x1, y1, x2, y2).
    """
    taken = np.zeros((len(boundingRects),), dtype=bool)
    boundingRects = sorted(boundingRects, key=lambda b: b[0])
 
    (x, y, w, h) = boundingRects[0]
    mergedRects = [(x, y, x + w, y + h)]
    k = 0

    for i in range(1, len(boundingRects)):
        (x, y, w, h) = boundingRects[i]
        old_x, old_y, old_x2, old_y2 = mergedRects[k]
        old_w = old_x2 - old_x
        old_h = old_y2 - old_y

        if w < 2 * staffHeight and h < 2 * staffHeight:
            continue

        if ((((x - old_x2) < 0.75 * spaceHeight and (x - old_x2) >= 0) and ((old_h <= spaceHeight + staffHeight) or (h <= spaceHeight + staffHeight)))
            or (x < old_x2 and h < 3 * spaceHeight)):

            mergedRects[k] = (min(old_x, x), min(old_y, y), max(old_x2, x + w), max(old_y2, y + h))
        else:
            mergedRects.append((x, y, x + w, y + h))
            k += 1

    return mergedRects

def getHalfs(lines, spaceHeight, height):
    """
    Divides the image height into segments based on detected lines.

    Args:
        lines (list): Detected horizontal line positions in the image.
        spaceHeight (int): Height of the space between staff lines.
        height (int): Height of the image.

    Returns:
        list: Positions of the halfway points dividing the image.
    """
    detected_lines = np.zeros((height,))
    detected_lines[lines] = 1

    starts = np.where((detected_lines[:-1] == 1) & (detected_lines[1:] == 0))[0] + 1
    ends = np.where((detected_lines[:-1] == 0) & (detected_lines[1:] == 1))[0]

    starts = starts[:-1]
    ends = ends[1:]

    halfs = [0]

    for i in range(len(starts)):
        diff = ends[i] - starts[i]
        if diff > 5 * spaceHeight:
            halfs.append((ends[i] + starts[i]) // 2)

    halfs.append(height - 1)
    return halfs

def getObjects(staffless, staffHeight, spaceHeight):
    """
    Identifies and merges objects in the staffless binary image.

    Args:
        staffless (ndarray): Binary image with staffs removed.
        staffHeight (int): Height of the staff lines in the image.
        spaceHeight (int): Height of the space between staff lines.

    Returns:
        list of tuples: Merged bounding rectangles represented as (x1, y1, x2, y2).
    """
    cnt, hir = cv2.findContours(staffless, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = []
    
    for i in range(len(cnt)):

        if hir[0,i,3] == -1:
            contours.append(cnt[i])
    
    boundingRects = [cv2.boundingRect(c) for c in contours]
 
    mergedRects = mergeContours(staffless, boundingRects, staffHeight, spaceHeight)
    
    boundingRects = []
    for m in mergedRects:
        boundingRects.append([m[0],m[1], m[2]-m[0], m[3]-m[1]])

    mergedRects = mergeContours(staffless, boundingRects, staffHeight, spaceHeight)
    
    return mergedRects

def segmentImage(staffless, lines, staffHeight, spaceHeight):
    """
    Segments an image into bounding rectangles based on horizontal lines and object detection.

    Args:
        staffless (ndarray): Binary image with staffs removed.
        lines (list): Detected horizontal line positions in the image.
        staffHeight (int): Height of the staff lines in the image.
        spaceHeight (int): Height of the space between staff lines.

    Returns:
        list of tuples: Bounding rectangles for segmented image regions.
    """
    halfs = getHalfs(lines, spaceHeight, staffless.shape[0])
    
    boundingRects = []
    for i in range(len(halfs) - 1):
        segment = staffless[halfs[i]:halfs[i+1] + 1]
        mergedRects = getObjects(segment, staffHeight, spaceHeight)

        for b in mergedRects:
            boundingRects.append((b[0], halfs[i] + b[1],b[2], halfs[i] + b[3]))
    
    return boundingRects