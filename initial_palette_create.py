import torch
from sklearn.cluster import KMeans
import lpips



def initialize_palette(image, num_colors, method ='adam'):

    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    palette = kmeans.cluster_centers_   
    #palette 0 to 255
    if method == 'kmeans':    
       
       return palette.astype(int)
    
    elif method == 'adam':

        palette = ini_adam_opt_cuda(image,palette)

        palette =palette.astype(int)

        return palette


#  use adam palette optimized，with cuda.
def ini_adam_opt_cuda(image_orig, initial_palette):

    # normalized
    image_orig = image_orig / 255.0
    initial_palette = initial_palette / 255.0

    # transfer to tensor
    palette_tensor = torch.from_numpy(initial_palette).float().cuda()
    palette_tensor.requires_grad = True

    image_tensor = torch.from_numpy(image_orig).permute(2,0,1).float()
    image_tensor = image_tensor.cuda()

    loss_fn = lpips.LPIPS(net='alex').cuda()

    # optimizer adamW
    optimizer = torch.optim.AdamW([palette_tensor], lr=0.01, weight_decay=1e-4)
    
    num_epochs = 100

    #increase annealing mechanism, better performance.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        mapped_image = initialize_mapping_cuda(image_tensor,palette_tensor)
        loss = loss_fn(mapped_image, image_tensor)

        loss.backward()

        optimizer.step()

        # clamp
        with torch.no_grad():
            palette_tensor.clamp_(0.0,1.0)

        # update learning rate
        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    optimized_palette = palette_tensor.detach().cpu().numpy()
    optimized_palette = (optimized_palette * 255.0).astype(int)
    return optimized_palette


#  use cuda nn method，provide  mapping to ini_adam_opt_cuda
def initialize_mapping_cuda(image_tensor, palette_tensor):
    C, H, W = image_tensor.shape  
    num_colors = palette_tensor.size(0)  

    # calculate distance
    image_flat = image_tensor.view(C, -1).T  

    # expend
    palette_expanded = palette_tensor.unsqueeze(0)  
    image_expanded = image_flat.unsqueeze(1)        
    distances = torch.norm(image_expanded - palette_expanded, dim=2)  

    # nearest index
    nearest_color_indices = distances.argmin(dim=1)  

    quantized_image_flat = palette_tensor[nearest_color_indices]  

    # get back to origin shape
    quantized_image = quantized_image_flat.T.view(C, H, W)  

    
    return quantized_image



