from utils import * 
from postprocessing.notes import *
import torch

def extract_hog_features(img,target_img_size):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from an image.

    Args:
        img (numpy.ndarray): Input image.
        target_img_size (tuple): Target size to resize the image (width, height).

    Returns:
        numpy.ndarray: Flattened HOG feature vector.
    """
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)
    
    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def SIFT(image):
    """
    Extracts SIFT (Scale-Invariant Feature Transform) descriptors from an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: SIFT descriptors of the input image.
    """
    sift = cv2.SIFT_create()
    if image.dtype != np.uint8:
      image = (image * 255).astype(np.uint8)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors


def isDot(img, spaceHeight):
    """
    Determines if the given image represents a dot based on its height.

    Args:
        img (numpy.ndarray): Input image.
        spaceHeight (int): The height of a space in the context of staff lines.

    Returns:
        bool: True if the image represents a dot, False otherwise.
    """
    if img.shape[0] > spaceHeight:
        return False
    return True


def chord2text(img,cnt_pos,staffHeight,spaceHeight,lines):
    """
    Converts a chord image into text representation.

    Args:
        img (numpy.ndarray): Input chord image.
        cnt_pos (list): Position of the chord in the image.
        staffHeight (int): Height of the staff.
        spaceHeight (int): Height of a space between staff lines.
        lines (list): Positions of staff lines.

    Returns:
        str: Text representation of the chord.
    """
    char_middle = ''
    char_top = ''
    char_down = ''
    height = cnt_pos[1] - cnt_pos[0]
    if height > 2.75 * spaceHeight: 

        middle = img[img.shape[0]//3:img.shape[0]*2//3,staffHeight:img.shape[1]-staffHeight]
       
        _, top, distanceTop = getNearestLine(cnt_pos[0],lines)
        _, bottom, distanceBottom = getNearestLine(cnt_pos[0] + height //3 ,lines)
        char_top = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
        _, top, distanceTop = getNearestLine(cnt_pos[0] + height*2//3,lines)
        _, bottom, distanceBottom = getNearestLine((cnt_pos[1]),lines)
        char_down = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
        if np.sum(middle) > len(middle)//2:

            _, top, distanceTop = getNearestLine(cnt_pos[0] + height //3 ,lines)
            _, bottom, distanceBottom = getNearestLine(cnt_pos[0] + height*2//3,lines)
            char_middle = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
    elif height < 1.5 * spaceHeight:

        _, top, distanceTop = getNearestLine(cnt_pos[0],lines)
        _, bottom, distanceBottom = getNearestLine(cnt_pos[1],lines)
        char_down = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
    else: 
        _, top, distanceTop = getNearestLine(cnt_pos[0],lines)
        _, bottom, distanceBottom = getNearestLine(cnt_pos[0]+ height//2,lines)
        char_top = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
        _, top, distanceTop = getNearestLine(cnt_pos[0]+height//2,lines)
        _, bottom, distanceBottom = getNearestLine(cnt_pos[1],lines)
        char_down = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)

    return str(char_down) + str(char_middle) + str(char_top)


def getchordText(cnt_pos,cnt_img,staffHeight,spaceHeight,lines):
    """
    Extracts text representation from an image of a chord.

    Args:
        cnt_pos (list): Position of the chord in the image.
        cnt_img (numpy.ndarray): Image containing the chord.
        staffHeight (int): Height of the staff.
        spaceHeight (int): Height of a space between staff lines.
        lines (list): Positions of staff lines.

    Returns:
        str: Text representation of the chord.
    """
    h_hist = np.sum(cnt_img,axis=0)
    bar_idx = np.where(h_hist== np.max(h_hist))[0][0]
    img = binary_opening(cnt_img.copy(),np.ones((staffHeight,cnt_img.shape[1]//2)))

    if ((bar_idx >  cnt_img.shape[1]//2 -staffHeight) and (bar_idx < cnt_img.shape[1]//2+staffHeight)):

        rh = img[:,:bar_idx] 
        rh = binary_opening(rh.copy(),np.ones((staffHeight,rh.shape[1]//2)))
        lh = img[:,bar_idx:] 
        lh = binary_opening(lh.copy(),np.ones((staffHeight,lh.shape[1]//2)))

        hist = np.sum(rh,axis=1)
        idxs = np.where(hist > staffHeight)
        min_y = idxs[0][0]
        max_y = idxs[0][len(idxs[0])-1]+1
        rcnt_pos = [min_y+cnt_pos[0],max_y+cnt_pos[0]]
        text1 = chord2text(img,rcnt_pos,staffHeight,spaceHeight,lines)

        hist = np.sum(lh,axis=1)
        idxs = np.where(hist > staffHeight)
        min_y = idxs[0][0]
        max_y = idxs[0][len(idxs[0])-1]+1
        lcnt_pos = [min_y+cnt_pos[0],max_y+cnt_pos[0]]     
        text2 = chord2text(img,lcnt_pos,staffHeight,spaceHeight,lines)
        return "".join(sorted(text1 + text2))
    else:

        hist = np.sum(img,axis=1)
        idxs = np.where(hist > staffHeight)
        min_y = idxs[0][0]
        max_y = idxs[0][len(idxs[0])-1]+1
        cnt_pos = [min_y+cnt_pos[0],max_y+cnt_pos[0]]
        return "".join(sorted(chord2text(img,cnt_pos,staffHeight,spaceHeight,lines)))


def isHalf(img, spaceHeight):
    """
    Determines if the given image represents a half note based on projections.

    Args:
        img (numpy.ndarray): Input image.
        spaceHeight (int): The height of a space in the context of staff lines.

    Returns:
        bool: True if the image represents a half note, False otherwise.
    """
    w = img.shape[1]
    h = img.shape[0]
    hist = np.zeros((w,4), dtype=np.uint32)
    min_x, max_x = 0, w
    min_y, max_y = 0, h 
    for i in range(w):
        window = img[:, i: min(i + 1, w)]

        xprojection = np.sum(window, axis=1)

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
            if 0.5*spaceHeight < length < spaceHeight*1.5:
                hist[i,0] = length
    projections = len(np.where(hist[:,0]>0)[0])
    if projections > img.shape[1]//3:
        return False
    else:
        return True


def downSize(image, width=1000):
    """
    Resizes an image to a specified width while maintaining the aspect ratio.

    Args:
        image (numpy.ndarray): Input image.
        width (int): Target width for resizing. Default is 1000.

    Returns:
        numpy.ndarray: Resized image.
    """
    (h, w) = image.shape[:2]
    print(h, w)
    shrinkingRatio = width / float(w)
    dsize  = (width, int(h * shrinkingRatio))
    resized = cv2.resize(image, dsize , interpolation=cv2.INTER_AREA)
    return resized  


def knn_predict(X_train, y_train, X_test, k=3):
    """
    Predicts the labels of test data using k-Nearest Neighbors (kNN) classification.

    Args:
        X_train (torch.Tensor): Training data.
        y_train (torch.Tensor): Training labels.
        X_test (torch.Tensor): Test data.
        k (int): Number of neighbors to consider. Default is 3.

    Returns:
        torch.Tensor: Predicted labels for the test data.
    """
    distances = torch.cdist(X_test, X_train)    
    knn_indices = distances.topk(k, largest=False).indices
    knn_labels = y_train[knn_indices]
    y_pred = torch.mode(knn_labels, dim=1).values
    return y_pred


def predict_image(image, X_train, y_train, k=17):
    """
    Predicts the class of an image using k-Nearest Neighbors (kNN) classification.

    Args:
        image (numpy.ndarray): Input image.
        X_train (torch.Tensor): Training data.
        y_train (torch.Tensor): Training labels.
        k (int): Number of neighbors to consider. Default is 17.

    Returns:
        int: Predicted label for the input image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if image is None:
        raise ValueError(f"Invalid image")
    
    descriptors = extract_hog_features(image,(32,32))
    if descriptors is None or len(descriptors) == 0:
        raise ValueError(f"No descriptors found for")
    
    X_test = torch.tensor([descriptors], dtype=torch.float32).to(device)
    y_pred = knn_predict(X_train, y_train, X_test, k)
    return y_pred.item()