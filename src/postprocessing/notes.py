from utils import *

def getNearestLine(y, lines):
    """
    Find the nearest musical staff line to a given y-coordinate.

    Args:
        y (int): The y-coordinate to find the nearest line for.
        lines (numpy array): Array of y-coordinates representing the musical staff lines.

    Returns:
        tuple: The closest line's y-coordinate, its position modulo 5, and the vertical distance to the input y-coordinate.
    """
    diff = lines - y
    min_pos = np.argmin(np.abs(diff))
    closest_line = lines[min_pos]
    distance = diff[min_pos]
    
    closest_line_pos = np.where(lines == closest_line)[0][0] % 5
    
    return closest_line, closest_line_pos, distance


def getBeamNoteHeads(img, boundingRect, staffHeight, spaceHeight):
    """
    Identify the note heads in a beamed note group and count the number of beams.

    Args:
        img (numpy array): Binary image of the musical sheet.
        boundingRect (tuple): Bounding rectangle (min_x, min_y, max_x, max_y) around the beam.
        staffHeight (int): Height of a single staff line.
        spaceHeight (int): Distance between adjacent staff lines.

    Returns:
        tuple: Array of detected note head positions and the number of beams.
    """
    (min_x,min_y,max_x,max_y) = boundingRect
    w = max_x - min_x
    h = max_y - min_y
    
    contourImage = img[min_y:max_y, min_x:max_x]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(staffHeight * 2, 1))
    contourImage = cv2.morphologyEx(contourImage, cv2.MORPH_OPEN, kernel)
    
    hist = np.zeros((w,4), dtype=np.uint32)
    
    for i in range(w):
        window = contourImage[:, i: min(i + 1, w)]
        xprojection = window

        starts = np.array((xprojection[:-1] == 0) & (xprojection[1:] != 0))
        starts_ix = np.where(starts)[0] + 1
        ends = np.array((xprojection[:-1] != 0) & (xprojection[1:] == 0))
        ends_ix = np.where(ends)[0]

        if xprojection[0] != 0:
            starts_ix = np.append(0, starts_ix)

        if xprojection[-1] != 0:
            ends_ix = np.append(ends_ix, xprojection.size-1)

        if starts_ix.size != 0:
            index = np.argmax(ends_ix - starts_ix)
            hist[i,1] = min_x + i
            hist[i,2] = min_y + starts_ix[index]
            hist[i,3] = min_y + ends_ix[index]
            length = hist[i,3] - hist[i,2]
            if 0.75*spaceHeight < length < spaceHeight*1.5:
                hist[i,0] = length
        
    peaks, _ = find_peaks(hist[:,0], distance=spaceHeight)
    
    hists = hist[peaks]
    mean_y = np.mean((hists[:,2] + hists[:,3])//2) - min_y
    beams = 0
    if mean_y > h/2:
        beams = getNumberOfBeams(contourImage[:np.min(hists[:,2])-min_y])
    else:
        beams = getNumberOfBeams(contourImage[np.max(hists[:,3]) - min_y:])
    
    return hists, beams



def getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight):
    """
    Determine the musical note based on the position of its head relative to staff lines.

    Args:
        top (int): Position of the top staff line.
        distanceTop (int): Distance of the note's top edge from the top line.
        bottom (int): Position of the bottom staff line.
        distanceBottom (int): Distance of the note's bottom edge from the bottom line.
        spaceHeight (int): Distance between adjacent staff lines.

    Returns:
        str: The musical note character.
    """
    if top == 3 and bottom == 4:
        if -distanceBottom >= 0.25 * spaceHeight:
            return 'e'
        if distanceTop >= 0.25 * spaceHeight:
            return 'g'
        return 'f'
    elif top == 2 and bottom == 3:
        if -distanceBottom >= 0.25 * spaceHeight:
            return 'g'
        if distanceTop >= 0.25 * spaceHeight:
            return 'b'
        return 'a'
    elif top == 1 and bottom == 2:
        if -distanceBottom >= 0.25 * spaceHeight:
            return 'b'
        if distanceTop >= 0.25 * spaceHeight:
            return 'd2'
        return 'c2'
    elif top == 0 and bottom == 1:
        if -distanceBottom >= 0.25 * spaceHeight:
            return 'd2'
        if distanceTop >= 0.25 * spaceHeight:
            return 'f2'
        return 'e2'
    
    if top == 3 and bottom == 3 and distanceTop > 0 and distanceBottom < 0:
        return 'g'
    elif top == 2 and bottom == 2 and distanceTop > 0 and distanceBottom < 0:
        return 'b'
    elif top == 1 and bottom == 1 and distanceTop > 0 and distanceBottom < 0:
        return 'd2'
    
    if top == 4 and bottom == 4:
        if distanceTop >= 0.25 * spaceHeight:
            return 'e'
        else:
            if -distanceTop <= 0.25 * spaceHeight:
                return 'd'
            else:
                return 'c'
        
    if top == 0 and bottom == 0:
        if -distanceBottom >= 0.25 * spaceHeight:
            return 'f2'
        if distanceTop <= 1.35 * spaceHeight:
            return 'g2'
        if distanceBottom <= 0.75 * spaceHeight:
            return 'a2'
        else:
            return 'b2'

    if top == 0 and bottom == 2:
        return 'd2'
    elif top == 1 and bottom == 3:
        return 'b'
    elif top == 2 and bottom == 4:
        return 'g'
    
    


def getNumberOfBeams(contour):
    """
    Count the number of beams (horizontal lines) in a musical note group.

    Args:
        contour (numpy array): Binary image of the contour containing the beams.

    Returns:
        int: Number of beams.
    """
    width = contour.shape[1]
    height = contour.shape[0]

    hist = np.zeros((height//2,), dtype=np.uint32)
    for i in range(width):
        a = contour[:,i]
        starts = np.array((a[:-1] == 0) & (a[1:] != 0))
        starts_ix = np.where(starts)[0] + 2
        ends = np.array((a[:-1] != 0) & (a[1:] == 0))
        ends_ix = np.where(ends)[0] + 2

        if a[0] != 0:
            starts_ix = np.append(1, starts_ix)

        if a[-1] != 0:
            ends_ix = np.append(ends_ix, a.size+1)
        
        runs = ends_ix - starts_ix
        hist[runs.size] += 1 
    
    return np.argmax(hist)



def getNoteCharacter(originalImage, boundingRect, noteClass, lines, staffHeight, spaceHeight):
    """
    Identify the musical character of a note based on its position and features.

    Args:
        originalImage (numpy array): Binary image of the musical sheet.
        boundingRect (tuple): Bounding rectangle (min_x, min_y, max_x, max_y) around the note.
        noteClass (str): Class of the note (e.g., 'a_1', 'a_4').
        lines (numpy array): Array of y-coordinates representing the staff lines.
        staffHeight (int): Height of a single staff line.
        spaceHeight (int): Distance between adjacent staff lines.

    Returns:
        str: The identified musical character.
    """
    img = originalImage.copy()
    
    (min_x,min_y,max_x,max_y) = boundingRect
    w = max_x - min_x
    h = max_y - min_y
    
    contourImage = img[min_y:max_y, min_x:max_x]

    character = ''
    
    if noteClass == 'a_1':
        noteTop = min_y
        noteBottom = max_y
        _, top, distanceTop = getNearestLine(noteTop,lines)
        _, bottom, distanceBottom = getNearestLine(noteBottom,lines)
        character = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
        character += '/1'

    elif noteClass == 'a_2':
        yprojection = np.sum(contourImage//255, axis=0)
        yprojection = np.where(yprojection>spaceHeight + 2*staffHeight)
        contourImage[:,yprojection] = 0
        
        a = np.sum(contourImage//255, axis=1)
    
        starts = np.array((a[:-1] == 0) & (a[1:] != 0))
        starts_ix = np.where(starts)[0] + 1
        ends = np.array((a[:-1] != 0) & (a[1:] == 0))
        ends_ix = np.where(ends)[0]

        if a[0] != 0:
            starts_ix = np.append(0, starts_ix)

        if a[-1] != 0:
            ends_ix = np.append(ends_ix, a.size-1)

        if starts_ix.size != 0:
            index = np.argmax(ends_ix - starts_ix)
            noteTop = min_y + starts_ix[index]
            noteBottom = min_y + ends_ix[index]
            
            _, top, distanceTop = getNearestLine(noteTop,lines)
            _, bottom, distanceBottom = getNearestLine(noteBottom,lines)
            character = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
            character += '/2'

    elif noteClass == 'a_4' or noteClass == 'a_8' or noteClass == 'a_16' or noteClass == 'a_32':
        yprojection = np.sum(contourImage//255, axis=0)
        yprojection = np.where(yprojection>spaceHeight+2*staffHeight)
        contourImage[:,yprojection] = 0
        
        hist = np.zeros((w,4), dtype=np.uint32)
    
        for i in range(w):
            window = contourImage[:, i: min(i + 1, w)]
            xprojection = window

            starts = np.array((window[:-1] == 0) & (window[1:] != 0))
            starts_ix = np.where(starts)[0] + 1
            ends = np.array((window[:-1] != 0) & (window[1:] == 0))
            ends_ix = np.where(ends)[0]

            if window[0] != 0:
                starts_ix = np.append(0, starts_ix)

            if window[-1] != 0:
                ends_ix = np.append(ends_ix, window.size-1)

            if starts_ix.size != 0:
                index = np.argmax(ends_ix - starts_ix)
                hist[i,1] = i
                hist[i,2] = starts_ix[index]
                hist[i,3] = ends_ix[index]
                length = hist[i,3] - hist[i,2]
                if 0.75*spaceHeight < length < spaceHeight*1.5:
                    hist[i,0] = length
        
        peaks, _ = find_peaks(hist[:,0], distance=spaceHeight)
        widths = peak_widths(hist[:,0], peaks)[0]
        
        peakIndex = np.argmax(widths)
        peak = peaks[peakIndex]
        h = hist[peak]
        noteTop = min_y + h[2]
        noteBottom = min_y + h[3]
        _, top, distanceTop = getNearestLine(noteTop,lines)
        _, bottom, distanceBottom = getNearestLine(noteBottom,lines)
        character = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
        if noteClass == 'a_4':
            character += '/4'
        elif noteClass == 'a_8':
            character += '/8'
        elif noteClass == 'a_16':
            character += '/16'
        else:
            character += '/32'
    elif noteClass == "beam":
        heads, noOfBeams = getBeamNoteHeads(img, boundingRect, staffHeight, spaceHeight)
        division = int(8*noOfBeams)
        for h in heads:
            _, top, distanceTop = getNearestLine(h[2],lines)
            _, bottom, distanceBottom = getNearestLine(h[3],lines)
            character += getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
            character += '/' + str(division) + ' '
        character = character[:-1]
    
    return character

