import numpy as np
import cv2
from matplotlib import pyplot as plt
from initial_palette_create import initialize_palette
from optimization import optimization_2
import argparse


def initialize_mapping_nn(image, palette):

    H,W,_ =image.shape
    mapped_image = np.zeros((H,W,3),dtype=np.float32)
    flattened_image =image.reshape(-1,3)
    distances = np.linalg.norm(flattened_image[:,None] - palette, axis =2)
    nearest_color_indices = distances.argmin(axis=1)
    mapped_image = palette[nearest_color_indices].reshape(H,W,3)
    return np.clip(mapped_image,0,255).astype(int)


def calculate_psnr(original, quantized):
    mse = np.mean((original - quantized) ** 2)
    if mse == 0:
        return float('inf')  
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def adam_ESQE(image, colornum):
    initial_palette = initialize_palette(image, colornum,'adam')
    fin_image,fin_palette = optimization_2(image,initial_palette,sigma_s=2,sigma_r=1,max_iter=6)

    return fin_image


def adam(image, colornum):
    initial_palette = initialize_palette(image, colornum,'adam')
    fin_image = initialize_mapping_nn(image,initial_palette)

    return fin_image




def main(colors, image_path):
    colornum = colors
    image = cv2.imread(image_path)[:,:,::-1] 

    final_image = adam_ESQE(image, colornum)

    final_image =np.clip(final_image, 0, 255).astype(np.uint8)

    out_path = "output.jpg"
    img_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_path, img_rgb)
    plt.imshow(final_image)
    plt.axis("off")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image processing")
    
    parser.add_argument(
        "--colors", 
        type=int, 
        required=True, 
        help="determined color number"
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True, 
        help="image path"
    )
    
    args = parser.parse_args()
    
    main(args.colors, args.image_path)






