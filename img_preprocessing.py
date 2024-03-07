import torch.utils.data
import glob
import torch.nn.functional as F
import numpy as np
import torch
import cv2

# Loads a folder of images into NumPy format
# Args: paths = string path to image folder of *.png
#
# Returns: NumPy array of images
def loader(paths):
    img_paths = glob.glob(paths)
    imgs = None

    for img_path in img_paths:
        # read in image into NumPy
        img = np.expand_dims(cv2.imread(img_path, mode='RGB'), 0)
        if imgs:
            imgs = np.concatenate((imgs, img))
        else:
            imgs = img
    return imgs

# Initializes the camera parameters
# Args: images = NumPy array of images
#
# Returns: intrinsics = size 1 tensor for intrinsics (focal length), 
# extrinsics = zero tensor of length(num images) by 6 for extrinsics
# where the first 3 cols represent translation vector and the last 
# 3 represent the rotation vector
def initializing_cam_params(imgs, device='cpu'):
    intrinsics = torch.ones(1, device=device, dtype=torch.float32, requires_grad=True)
    length = imgs.shape[0]
    extrinsics = torch.zeros((length, 6), device=device, dtype=torch.float32, requires_grad=True)
    return intrinsics, extrinsics

# Transforms from world space and returns ray positions and directions 
# in NDC space based on equations (25) and (26) in paper
# Args: o = positions of ray origins, d = directions of rays
# W = width of image in pixels, H = height, f = focal length of
# camera, and near = the near clipping plane
#
# Returns; rays in NDC space
def transform_ndc(o, d, W, H, f, near=1.0):
    # first shift o to ray's intersection with near
    t_near = -(near + o[:, 2]) / (d[:, 2])
    t_near = t_near[:, None]
    o_n = o + (t_near * d)

    # get o components
    o_x, o_y, o_z = o_n[:, 0], o_n[:, 1], o_n[:, 2]
    d_x, d_y, d_z = d[:, 0], d[:, 1], d[:, 2]

    o_ndc_1 = -(f * o_x) / ((W / 2.0) * o_z)
    o_ndc_2 = -(f * o_y) / ((H / 2.0) * o_z)
    o_ndc_3 = 1.0 + ((2.0 * near) / o_z)
    # stack along last dim
    o_ndc = torch.stack([o_ndc_1, o_ndc_2, o_ndc_3], -1)

    d_ndc_1 = -(f * ((d_x / d_z) - (o_x / o_z))) / (W / 2.0)
    d_ndc_2 = -(f * ((d_y / d_z) - (o_y / o_z))) / (H/ 2.0)
    d_ndc_3 = -(2.0 * near) / (o_z)
    # stack along last dim
    d_ndc = torch.stack([d_ndc_1, d_ndc_2, d_ndc_3], -1)

    return o_ndc, d_ndc


# transform_ndc transforms from world to ndc space, but we are 
# currently in camera space, so we must use the c2w matrix to 
# transform to the world space.

# Gets batches of rays from the given images
# Args: batch_size = num images to get H/W = height and width in pixels, 
# i = index specifying img we're working with, in/extrinsics = camera 
# params, test = bool representing whether this method is called during 
# testing or training.
#
#
def batch(batch_size, H, W, i, intrinsics, extrinsics, test):
    if test:
        image_indices = (torch.zeros(W * H) + i).type(torch.long)
        u, v = np.meshgrid(np.linspace(0, W - 1, W, dtype=int), np.linspace(0, H - 1, H, dtype=int))
        u = torch.from_numpy(u.reshape(-1)).to(intrinsics.device)
        v = torch.from_numpy(v.reshape(-1)).to(intrinsics.device)
    else:
        image_indices = (torch.zeros(batch_size) + i).type(torch.long)  # Sample random images
        u = torch.randint(W, (batch_size,), device=intrinsics.device)  # Sample random pixels (getting x and y)
        v = torch.randint(H, (batch_size,), device=intrinsics.device)

    # extract focal distance and translation/rotation vectors
    f = intrinsics[0] ** 2 * W
    t = extrinsics[i, :3]
    r = extrinsics[i, -3:]

    # begin building 4x4 c2w matrix which is composed of a rotation matrix and a translation vector
    phi_skewed = torch.tensor([[0.0, -r[2], r[1]],
                               [r[2], 0.0, -r[0]],
                               [-r[1], r[0], 0.0]])
    
    a = r.norm() + 1e-15

    # following 4.1 in paper
    rotation_matrix = torch.eye(3).to(device=t.device) + (torch.sin(a) / a) * (phi_skewed) + ((1-torch.cos(a)) / a ** 2) * (torch.matmul(phi_skewed, phi_skewed))

    # make t vertical
    t = torch.cat([t, torch.tensor([0.0]).to(device=t.device)])
    t = torch.unsqueeze(t, 1)

    c2w_matrix = torch.cat([rotation_matrix, torch.tensor([[0.0, 0.0, 0.0]]).to(device=t.device)], 0)
    c2w_matrix = torch.cat([c2w_matrix, t], 1)



    # following 4.2 in paper
    d =  torch.cat([((u.to(intrinsics.device) - .5 * W) / f).unsqueeze(-1),
                    (-(v.to(intrinsics.device) - .5 * H) / focal).unsqueeze(-1),
                    -torch.ones_like(u).unsqueeze(-1)], dim=-1)
    d_world = torch.matmul(c2w_matrix[:3, :3].view(1, 3, 3), d.unsqueeze(2)).squeeze(2)
    o_world = c2w_matrix[:3, 3].view(1, 3).expand_as(rays_d_world)
    rays_o_world, rays_d_world = transform_ndc(H, W, f, o=o_world, d=rays_d_world)
    return rays_o_world, F.normalize(rays_d_world, p=2, dim=1), (image_indices, v.cpu(), u.cpu())




