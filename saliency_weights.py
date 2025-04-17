import cv2
import numpy as np
from skimage.segmentation import slic

# .............................................................................
def calculate_saliency_map_2ds(image):
    # segents
    segments = slic(image, n_segments=100, compactness=10)
    
    # calculate average
    saliency = np.zeros_like(image[:,:,0], dtype=np.float32)
    for label in np.unique(segments):
        mask = (segments == label)
        region_color = np.mean(image[mask], axis=0)
        
        # saliency
        global_color = np.mean(image, axis=(0,1))
        color_diff = np.linalg.norm(region_color - global_color)
        saliency[mask] = color_diff
    
    # smoothing
    saliency = cv2.GaussianBlur(saliency, (5,5), 0)
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    return saliency


def visual_weights(image, lambda_= 0.1):
    saliency_map = calculate_saliency_map_2ds(image)

    t_i = lambda_ + (1 - lambda_) * saliency_map

    return t_i
