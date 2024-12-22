import skimage as sk
import numpy as np
import hu_moments as hu


from utils import gray_img, get_thresholded

def feature_extraction(img):
    """
    Extracts a combined feature vector from the input image using HU moments and HOG features.

    Parameters:
    img (numpy.ndarray): The input image as a numpy array.

    Returns:
    numpy.ndarray: A concatenated feature vector combining HU moments and HOG features.
    """
    img_gray = gray_img(img)
    img_bin =img_gray>125
    hu_moments_feature = hu.hu_moments(img_bin)
    hog_features=sk.feature.hog(img_gray, 
               orientations=8, pixels_per_cell=(8, 8), 
               cells_per_block=(2, 2), block_norm='L1',feature_vector=1)
    return np.concatenate((hu_moments_feature, hog_features))

img =sk.io.imread('data\input\04.PNG')
sk.io.imshow(img)

features = feature_extraction(img)
print(features)