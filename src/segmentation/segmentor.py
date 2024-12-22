from utils import *


def mergeContours(staffless, boundingRects, staffHeight, spaceHeight):
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
    halfs = getHalfs(lines, spaceHeight, staffless.shape[0])
    
    boundingRects = []
    for i in range(len(halfs) - 1):
        segment = staffless[halfs[i]:halfs[i+1] + 1]
        mergedRects = getObjects(segment, staffHeight, spaceHeight)

        for b in mergedRects:
            boundingRects.append((b[0], halfs[i] + b[1],b[2], halfs[i] + b[3]))
    
    return boundingRects