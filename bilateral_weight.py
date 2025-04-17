import numpy as np


def neighborhood(image, i, j, kernel_size = 3):

    half_k = kernel_size //2
    H,W,C = image.shape
    x_min, x_max = max(0, j - half_k), min(W, j + half_k + 1)
    y_min, y_max = max(0, i - half_k), min(H, i + half_k + 1)
    return image[y_min:y_max, x_min:x_max]    



# .........................................................................
def bilateral_weights_2ds(center_pixel, window, sigma_s, sigma_r, eps=1e-6):

    H, W, C = window.shape
    
    # generate spatial coordinates
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij') 
    y_center, x_center = H // 2, W // 2
    
    # calculate weight
    spatial_dist2 = (x - x_center)**2 + (y - y_center)**2
    spatial_weight = np.exp(-spatial_dist2 / (2 * sigma_s**2))
    
    # calculate weight
    color_diff = np.linalg.norm(window - center_pixel, axis=-1)
    color_weight = np.exp(-color_diff**2 / (2 * sigma_r**2))
    
    # merge and normalized
    weights = spatial_weight * color_weight
    return weights / (weights.sum() + eps) 

