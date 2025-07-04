import numpy as np
import torch

def forward_kinematics(joint_positions):
    if isinstance(joint_positions, torch.Tensor):
        joint_positions = joint_positions.cpu().numpy()
    if joint_positions.ndim == 1:
        joint_positions = joint_positions.reshape(1, -1)
    x = np.cos(joint_positions[:, 0]) + np.sin(joint_positions[:, 1])
    y = np.sin(joint_positions[:, 0]) + np.cos(joint_positions[:, 2])
    z = joint_positions[:, 3] + joint_positions[:, 4]
    return np.stack([x, y, z], axis=1)

def nMSE(y_true, y_pred):
    error = np.mean((y_true - y_pred)**2, axis=0)
    variance = np.var(y_true, axis=0)
    variance[variance == 0] = 1
    return error / variance

def euclidean_error(y_true, y_pred):
    ee_pos_true = forward_kinematics(y_true[:, :7])
    ee_pos_pred = forward_kinematics(y_pred[:, :7])
    return np.linalg.norm(ee_pos_true - ee_pos_pred, axis=1)

def get_transformation_matrix(theta, d, a, alpha):
    """
    Calculates the transformation matrix for a single joint using Denavit-Hartenberg (DH) parameters.
    This matrix represents the position and orientation of one joint's frame relative to the previous one.
    """
    # pre-calculate sine and cosine for efficiency
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    
    # construct the 4x4 transformation matrix
    A = np.array([
        [c_theta, -s_theta * c_alpha,  s_theta * s_alpha, a * c_theta],
        [s_theta,  c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
        [0,       s_alpha,            c_alpha,           d],
        [0,       0,                  0,                 1]
    ])
    return A

def calculate_op_space_error(y_true, y_pred):
    """
    Calculates the mean Euclidean error in the operational space for a full trajectory.
    y_true and y_pred are arrays of shape (n_samples, 14).
    """
    # extract just the position columns (first 7 columns) from the target and prediction arrays
    true_joint_positions = y_true[:, :7]
    pred_joint_positions = y_pred[:, :7]
    
    # apply the forward_kinematics function to every row (i.e., every time step)
    true_xyz_coords = np.apply_along_axis(forward_kinematics, 1, true_joint_positions)
    pred_xyz_coords = np.apply_along_axis(forward_kinematics, 1, pred_joint_positions)
    
    # calculate the Euclidean distance between the true and predicted xyz coordinates for each time step
    errors = np.linalg.norm(true_xyz_coords - pred_xyz_coords, axis=1)
    
    # return the average error over the whole trajectory
    return np.mean(errors)