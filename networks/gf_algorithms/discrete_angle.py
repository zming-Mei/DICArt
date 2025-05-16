
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import math

def discretize_euler_angles(euler_angles, number_bins=720):

    euler_degrees = torch.rad2deg(euler_angles) % 360.0  
    bin_width = 360.0 / number_bins
    discretized_indices = (euler_degrees / bin_width).long()
    discretized_indices = torch.clamp(discretized_indices, 0, number_bins - 1)
    
    return discretized_indices

def matrix_to_spherical_angles(matrices):
    is_torch_tensor = isinstance(matrices, torch.Tensor)
    if is_torch_tensor:
        device = matrices.device
        matrices_np = matrices.cpu().detach().numpy()
    else:
        matrices_np = matrices

    rotation = Rotation.from_matrix(matrices_np)
    rotvec = rotation.as_rotvec()

    alpha = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    mask_alpha_small = alpha.squeeze(-1) < 1e-6 
    u = np.zeros_like(rotvec)
    non_zero_mask = ~mask_alpha_small
    u[non_zero_mask] = rotvec[non_zero_mask] / alpha[non_zero_mask]
    theta = np.arccos(np.clip(u[..., 2], -1.0, 1.0))
    phi = np.arctan2(u[..., 1], u[..., 0])
 
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    alpha_deg = np.degrees(alpha.squeeze(-1))

    theta_deg = np.where(
        np.isclose(theta_deg, 0, atol=1e-4),  
        0.5,  
        np.where(
            np.isclose(theta_deg, 180, atol=1e-4),  
            179.5,  
            theta_deg
        )
    )

    phi_deg = phi_deg % 360
    phi_deg = np.where(np.isclose(phi_deg, 0, atol=1e-6), 360, phi_deg)
    theta_deg[mask_alpha_small] = 0.0
    phi_deg[mask_alpha_small] = 0.0
    alpha_deg[mask_alpha_small] = 0.0
    result = np.stack([theta_deg, phi_deg, alpha_deg], axis=1)
    if is_torch_tensor:
        return torch.from_numpy(result).float().to(device)
    else:
        return result


def spherical_angles_to_matrix(angles):

    is_torch_tensor = isinstance(angles, torch.Tensor)
    if is_torch_tensor:
        device = angles.device
        angles_np = angles.cpu().detach().numpy()
    else:
        angles_np = angles
    if angles_np.ndim == 1:
        angles_np = angles_np[None, :]  
    elif angles_np.ndim != 2 or angles_np.shape[1] != 3:
        raise ValueError("The input shape must be (N,3) or (3,)")

    theta_deg, phi_deg, alpha_deg = angles_np[:, 0], angles_np[:, 1], angles_np[:, 2]
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    alpha_rad = np.radians(alpha_deg)
    u_x = np.sin(theta_rad) * np.cos(phi_rad)
    u_y = np.sin(theta_rad) * np.sin(phi_rad)
    u_z = np.cos(theta_rad)
    u = np.stack([u_x, u_y, u_z], axis=1)  

    alpha_zero_mask = np.isclose(alpha_deg, 0, atol=1e-6)
    rotvec = np.zeros_like(u)
    non_zero_mask = ~alpha_zero_mask
    rotvec[non_zero_mask] = u[non_zero_mask] * alpha_rad[non_zero_mask, np.newaxis]
    rotation = Rotation.from_rotvec(rotvec)
    matrices = rotation.as_matrix() 
    identity_matrix = np.eye(3)
    for i in np.where(alpha_zero_mask)[0]:
        matrices[i] = identity_matrix

    if matrices.shape[0] == 1 and (angles.ndim == 1 or (is_torch_tensor and angles.dim() == 1)):
        matrices = matrices[0]

    if is_torch_tensor:
        return torch.from_numpy(matrices).float().to(device)
    else:
        return matrices

def discretize_angle(angle, bins, min_val, max_val):

    discrete_values = np.linspace(min_val, max_val, bins, endpoint=False)
    idx = np.argmin(np.abs(discrete_values - angle))
    return discrete_values[idx]


def get_bin_index(angles, bins_theta, bins_phi, bins_alpha):
    is_torch_tensor = False
    try:
        import torch
        if isinstance(angles, torch.Tensor):
            is_torch_tensor = True
            device = angles.device
            angles_np = angles.cpu().detach().numpy()
        else:
            angles_np = np.asarray(angles)
    except ImportError:
        angles_np = np.asarray(angles)

    N = angles_np.shape[0]
    discretized_angles = np.zeros_like(angles_np)

    for i in range(N):
        theta = angles_np[i, 0]
        phi = angles_np[i, 1]
        alpha = angles_np[i, 2]
        theta_discrete = discretize_angle(theta, bins_theta, 0, 180)
        phi_discrete = discretize_angle(phi, bins_phi, 0, 360)
        alpha_discrete = discretize_angle(alpha, bins_alpha, 0, 180)

        theta_discrete = np.where(
        np.isclose(theta_discrete, 0, atol=1e-4), 
        0.5,  
        np.where(
            np.isclose(theta_discrete, 180, atol=1e-4), 
            179.5,
            theta_discrete)
        )

        discretized_angles[i] = [theta_discrete, phi_discrete, alpha_discrete]
    theta_min, theta_max = 0.0, 180.0
    bin_width_theta = (theta_max - theta_min) / bins_theta
    theta_indices = np.floor((discretized_angles[:, 0] - theta_min) / bin_width_theta).astype(int)
    theta_indices = np.clip(theta_indices, 0, bins_theta - 1)
    phi_min, phi_max = 0.0, 360.0
    bin_width_phi = (phi_max - phi_min) / bins_phi
    phi_indices = np.floor((discretized_angles[:, 1] - phi_min) / bin_width_phi).astype(int)
    phi_indices = np.clip(phi_indices, 0, bins_phi - 1)

    alpha_min, alpha_max = 0.0, 180.0
    bin_width_alpha = (alpha_max - alpha_min) / bins_alpha
    alpha_indices = np.floor((discretized_angles[:, 2] - alpha_min) / bin_width_alpha).astype(int)
    alpha_indices = np.clip(alpha_indices, 0, bins_alpha - 1)
    indices = np.column_stack((theta_indices, phi_indices, alpha_indices))
    if is_torch_tensor:
        indices = torch.from_numpy(indices).to(device)
    
    return indices

def bins_to_angles(indices, bins_theta, bins_phi, bins_alpha):

    is_torch_tensor = False
    try:
        import torch
        if isinstance(indices, torch.Tensor):
            is_torch_tensor = True
            device = indices.device
            indices_np = indices.cpu().detach().numpy()
        else:
            indices_np = np.asarray(indices)
    except ImportError:
        indices_np = np.asarray(indices)

    N = indices_np.shape[0]
    angles = np.zeros_like(indices_np, dtype=np.float32)

    theta_min, theta_max = 0.0, 180.0
    phi_min, phi_max = 0.0, 360.0
    alpha_min, alpha_max = 0.0, 180.0

    bin_width_theta = (theta_max - theta_min) / bins_theta
    bin_width_phi = (phi_max - phi_min) / bins_phi
    bin_width_alpha = (alpha_max - alpha_min) / bins_alpha

    for i in range(N):
        theta_idx = indices_np[i, 0]
        phi_idx = indices_np[i, 1]
        alpha_idx = indices_np[i, 2]


        theta_continuous = theta_idx * bin_width_theta  + theta_min
        phi_continuous = phi_idx * bin_width_phi  + phi_min
        alpha_continuous = alpha_idx * bin_width_alpha  + alpha_min
        theta_continuous = np.where(
            np.isclose(theta_continuous, 0, atol=1e-4),  
            0.5,  
            np.where(
                np.isclose(theta_continuous, 180, atol=1e-4), 
                179.5,
                theta_continuous)
        )
        angles[i] = [theta_continuous, phi_continuous, alpha_continuous]

    if is_torch_tensor:
        angles = torch.from_numpy(angles).float().to(device)
    
    return angles


def discretize_spherical_angles(angles, bins_theta, bins_phi, bins_alpha):

    is_torch_tensor = False
    try:
        import torch
        if isinstance(angles, torch.Tensor):
            is_torch_tensor = True
            device = angles.device
            angles_np = angles.cpu().detach().numpy()
        else:
            angles_np = np.asarray(angles)
    except ImportError:
        angles_np = np.asarray(angles)

    N = angles_np.shape[0]
    discretized_angles = np.zeros_like(angles_np)

    for i in range(N):
        theta = angles_np[i, 0]
        phi = angles_np[i, 1]
        alpha = angles_np[i, 2]

        theta_discrete = discretize_angle(theta, bins_theta, 0, 180)
        phi_discrete = discretize_angle(phi, bins_phi, 0, 360)
        alpha_discrete = discretize_angle(alpha, bins_alpha, 0, 180)

        theta_discrete = np.where(
        np.isclose(theta_discrete, 0, atol=1e-4),
        0.5,
        np.where(
            np.isclose(theta_discrete, 180, atol=1e-4),
            179.5,
            theta_discrete)
        )

        discretized_angles[i] = [theta_discrete, phi_discrete, alpha_discrete]
    

    if is_torch_tensor:
        return torch.from_numpy(discretized_angles).float().to(device)
    return discretized_angles


def discretize_euler_angles(euler_angles, num_bins):
    batch_size = euler_angles.shape[0]
    discretized_indices = torch.zeros_like(euler_angles, dtype=torch.long)

    angle_ranges = torch.tensor([
        [-math.pi, math.pi],
        [-math.pi/2, math.pi/2],
        [-math.pi, math.pi]
    ], device=euler_angles.device)
    
    for i in range(3):
        angle_min, angle_max = angle_ranges[i]
        angle_range = angle_max - angle_min
        
        if i == 1:
            near_pos_90 = torch.abs(euler_angles[:, i] - math.pi/2) < 1e-6
            near_neg_90 = torch.abs(euler_angles[:, i] + math.pi/2) < 1e-6
            
            special_cases = near_pos_90 | near_neg_90
            
            if special_cases.any():
                discretized_indices[near_pos_90, i] = num_bins - 1
                discretized_indices[near_neg_90, i] = 0
                
                normal_cases = ~special_cases
                normalized_angles = (euler_angles[normal_cases, i] - angle_min) / angle_range
                bin_indices = torch.floor(normalized_angles * num_bins).long()
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
                discretized_indices[normal_cases, i] = bin_indices
            else:
                normalized_angles = (euler_angles[:, i] - angle_min) / angle_range
                bin_indices = torch.floor(normalized_angles * num_bins).long()
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
                discretized_indices[:, i] = bin_indices
        else:
            normalized_angles = (euler_angles[:, i] - angle_min) / angle_range
            bin_indices = torch.floor(normalized_angles * num_bins).long()
            bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
            discretized_indices[:, i] = bin_indices
            
    return discretized_indices


def euler_angles_from_bins(bin_indices, num_bins):
    batch_size = bin_indices.shape[0]
    euler_angles = torch.zeros_like(bin_indices, dtype=torch.float32)
    
    angle_ranges = torch.tensor([
        [-math.pi, math.pi],
        [-math.pi/2, math.pi/2],
        [-math.pi, math.pi]
    ], device=bin_indices.device)
    
    for i in range(3):
        angle_min, angle_max = angle_ranges[i]
        angle_range = angle_max - angle_min
        
        if i == 1:
            is_max_bin = bin_indices[:, i] == num_bins - 1
            is_min_bin = bin_indices[:, i] == 0
            
            if is_max_bin.any():
                euler_angles[is_max_bin, i] = math.pi/2
            
            if is_min_bin.any():
                euler_angles[is_min_bin, i] = -math.pi/2
            
            normal_cases = ~(is_max_bin | is_min_bin)
            
            if normal_cases.any():
                bin_centers = (bin_indices[normal_cases, i].float() + 0.5) / num_bins
                euler_angles[normal_cases, i] = bin_centers * angle_range + angle_min
        else:
            bin_centers = (bin_indices[:, i].float() + 0.5) / num_bins
            euler_angles[:, i] = bin_centers * angle_range + angle_min
    
    return euler_angles


def get_topN_rank(probs, real_indices, N=50):
    bs, num_angles, num_bins = probs.shape
    
    _, top_indices = torch.topk(probs, k=N, dim=2)
    
    real_ranks = torch.zeros((bs, num_angles), dtype=torch.long, device=probs.device)
    
    for b in range(bs):
        for a in range(num_angles):
            sorted_probs, sorted_indices = torch.sort(probs[b, a], descending=True)
            
            real_idx = real_indices[b, a]
            for rank, idx in enumerate(sorted_indices):
                if idx == real_idx:
                    real_ranks[b, a] = rank + 1
                    break
    
    return top_indices, real_ranks