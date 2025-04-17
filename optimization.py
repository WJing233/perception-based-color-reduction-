import numpy as np
from bilateral_weight import neighborhood, bilateral_weights_2ds
from saliency_weights import visual_weights
from metric import calculate_lpips
import torch
import lpips


# optimization function
def optimization_2(image, palette, sigma_s, sigma_r, max_iter = 6, tol = 1e-4):

    mapping = optimize_mapping(image,palette,sigma_s,sigma_r)
    loss = 1
    fin_palette = palette
    fin_image = np.zeros_like(image)

    for iteration in range(max_iter):

        # optimize mapping 9
        new_mapping = optimize_mapping_9(image,palette,mapping,sigma_s,sigma_r)
        new_palette = optimize_palette_cuda(image,palette,new_mapping)

        palette_diff = np.linalg.norm(new_palette-palette)
        print(palette_diff)

        qu_image = new_palette[new_mapping]

        loss_new = calculate_lpips(image, qu_image)

        # test minimal loss
        print('each iteration loss value',loss_new)
        if loss_new < loss:
            loss = loss_new
            fin_image = qu_image
            fin_palette = new_palette       

        if palette_diff < tol:

            break

        palette, mapping = new_palette,new_mapping

    print(loss) # test loss
    return fin_image, fin_palette



# ......................................(mappning optimize).........................................
# test1 optimize mapping
def optimize_mapping(image,palette, sigma_s = 1, sigma_r = 2):
    H,W,C = image.shape
    num_colors = palette.shape[0]
    mapping = np.zeros((H,W),dtype = np.int32)

    for i in range(H):
        for j in range(W):

            x_min,x_max = max(0, i - 1), min(H, i + 2)
            y_min,y_max =max(0,j - 1),min(W, j + 2)
            window = image[x_min:x_max, y_min:y_max]

            spatial_weight = np.exp(-((np.arange(x_min, x_max)[:, None] - i)**2 + 
                                      (np.arange(y_min, y_max)[None, :] - j)**2) / (2 * sigma_s**2))
            color_diff = np.linalg.norm(window - image[i, j], axis=-1)
            color_weight = np.exp(-color_diff**2 / (2 * sigma_r**2))
            weights = spatial_weight * color_weight

            costs = np.zeros(num_colors)
            for k in range(num_colors):
                diff = palette[k] - window
                costs[k] = np.sum(weights * np.linalg.norm(diff, axis=-1)**2)

            mapping[i,j] = np.argmin(costs)

    return mapping



#  test 9 optimize mapping(able to use，s3r1)
def optimize_mapping_9(image, palette, mapping, sigma_s, sigma_r, lambda_=0.1, kernel_size=3):
   
    H, W, C = image.shape
    t_i = visual_weights(image, lambda_)

    for i in range(H):
        for j in range(W):
            # extract nebourhood window
            window = neighborhood(image, i, j, kernel_size)
            current_mapping = mapping[i, j]
            current_color = palette[current_mapping]

            # calculate bilateral weights
            weights = bilateral_weights_2ds(image[i, j], window, sigma_s, sigma_r)

            # initialze local cost
            min_cost = float('inf')
            best_mapping = current_mapping

            for k, colors in enumerate(palette):
                # calculate loss
                local_error = np.sum(weights[..., np.newaxis] * (colors - window), axis=(0, 1))
                local_cost = t_i[i, j] * np.linalg.norm(local_error)**2

                # update
                if local_cost < min_cost:
                    min_cost = local_cost
                    best_mapping = k

            # update
            mapping[i, j] = best_mapping

    return mapping




# .........................................(palette optimization).....................................
# use cuda test 2（able，fast，performance）
def optimize_palette_cuda(image, palette, mapping):
    # normalized
    image = image / 255.0
    palette = palette / 255.0

    # transfer to tensor
    palette_tensor = torch.from_numpy(palette).float().cuda()
    palette_tensor.requires_grad = True

    image_tensor = torch.from_numpy(image).permute(2,0,1).float()
    image_tensor = image_tensor.cuda()

    mapping_tensor = torch.tensor(mapping, dtype=torch.long, device=palette_tensor.device)

    # loss transfer to cuda
    loss_fn = lpips.LPIPS(net='alex').cuda()

    # try better optimizer
    optimizer = torch.optim.AdamW([palette_tensor], lr=0.01, weight_decay=1e-4)

    num_epochs = 100
    #increase Annealing mechanism
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    for epoch in range(num_epochs):
        optimizer.zero_grad()

        mapped_image = palette_tensor[mapping_tensor]
        mapped_image = mapped_image.permute(2, 0, 1)
        loss = loss_fn(mapped_image, image_tensor)

        loss.backward()


        optimizer.step()
        scheduler.step()

    optimized_palette = palette_tensor.detach().cpu().numpy()
    optimized_palette = (optimized_palette * 255.0).astype(int)
    return optimized_palette













