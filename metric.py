from lpips import LPIPS
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = LPIPS(net='alex').to(device)  

def calculate_lpips(img_org, img_processed):
  
    img_org_tensor = torch.tensor(img_org.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255 * 2 - 1  
    img_processed_tensor = torch.tensor(img_processed.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255 * 2 - 1

    img_org_tensor = img_org_tensor.to(device)
    img_processed_tensor = img_processed_tensor.to(device)

    with torch.no_grad():
        diff = loss_fn(img_org_tensor, img_processed_tensor)
    return diff.item()

def calculate_psnr(original, quantized):
    mse = np.mean((original - quantized) ** 2)
    if mse == 0:
        return float('inf')  
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr