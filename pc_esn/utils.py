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