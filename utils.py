import numpy as np
import torch

def nMSE(y_true, y_pred):
    error = np.mean((y_true - y_pred)**2, axis=0)
    variance = np.var(y_true, axis=0)
    variance[variance == 0] = 1
    return error / variance

def get_transformation_matrix(theta, d, a, alpha):
    """
    Calculates the transformation matrix for a single joint using Denavit-Hartenberg (DH) parameters.
    """
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    
    A = np.array([
        [c_theta, -s_theta * c_alpha,  s_theta * s_alpha, a * c_theta],
        [s_theta,  c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
        [0,       s_alpha,            c_alpha,           d],
        [0,       0,                  0,                 1]
    ])
    return A

def calculate_op_space_error(y_true, y_pred):
    """
    This function was already correct. It calculates the mean Euclidean error 
    for the full trajectory.
    """
    pred_joint_positions = y_pred[:, :7]
    
    # This correctly applies the FK function to each time step
    true_xyz_coords = np.apply_along_axis(forward_kinematics, 1, y_true[:, :7])
    pred_xyz_coords = np.apply_along_axis(forward_kinematics, 1, pred_joint_positions)
    
    # Calculate the Euclidean distance for each time step
    errors = np.linalg.norm(true_xyz_coords - pred_xyz_coords, axis=1)
    
    # Return the average error over the whole trajectory
    return np.mean(errors)

def forward_kinematics(joint_positions):
    """
    Calculates the end-effector XYZ position for the KUKA LWR4+.
    DH parameters are updated for the LWR4+ model based on standard specifications.
    """
    if isinstance(joint_positions, torch.Tensor):
        joint_positions = joint_positions.cpu().numpy()

    # Standard DH parameters for KUKA LWR4+
    # Parameters are [alpha_{i-1}, a_{i-1}, d_i, theta_i]
    # 'a' is link length, 'd' is link offset, 'alpha' is link twist.
    dh_params = np.array([
        # alpha_im1, a_im1,   d_i,   theta_i
        [0,           0,     0.310,  joint_positions[0]],
        [-np.pi/2,    0,     0,      joint_positions[1]],
        [np.pi/2,     0,     0.400,  joint_positions[2]],
        [np.pi/2,     0,     0,      joint_positions[3]],
        [-np.pi/2,    0,     0.390,  joint_positions[4]],
        [-np.pi/2,    0,     0,      joint_positions[5]],
        [np.pi/2,     0,     0.078,  joint_positions[6]]
    ])

    T = np.eye(4) # Initialize the transformation matrix
    for i in range(7): # Iterate through each of the 7 joints
        alpha_im1 = dh_params[i, 0]
        a_im1     = dh_params[i, 1]
        d_i       = dh_params[i, 2]
        theta_i   = dh_params[i, 3]

        # Calculate the transformation matrix for the current joint
        A_i = get_transformation_matrix(theta_i, d_i, a_im1, alpha_im1)
        
        # Update the overall transformation matrix
        T = T @ A_i
        
    # The XYZ position of the end-effector is in the last column
    return T[:3, 3]

def euclidean_error(y_true, y_pred):
    """
    Calculates the Euclidean error in operational space for each time step.
    It now correctly iterates over the input arrays.
    """
    # Use np.apply_along_axis to call forward_kinematics for each row (time step)
    ee_pos_true = np.apply_along_axis(forward_kinematics, 1, y_true[:, :7])
    ee_pos_pred = np.apply_along_axis(forward_kinematics, 1, y_pred[:, :7])
    
    # Calculate the norm (distance) between the true and predicted XYZ for each time step
    return np.linalg.norm(ee_pos_true - ee_pos_pred, axis=1)