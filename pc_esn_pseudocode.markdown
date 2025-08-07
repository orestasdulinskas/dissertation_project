# Physics-informed Echo State Network (PC-ESN) Pseudocode

## 1. Initialize Environment
- Set random seed for reproducibility
- If GPU is available:
  - Configure GPU settings for consistent behavior

## 2. Load and Prepare Data
- Load manipulator dynamics data (e.g., KUKA trajectories) from file
- Initialize empty collection for trajectories
- For each data entry:
  - If entry is a valid trajectory:
    - Store trajectory in collection
    - Count total samples

## 3. Define Evaluation Functions
### Calculate Normalized MSE
- Compute mean squared error between true and predicted values
- Normalize by variance of true values
- Return normalized error

### Compute Forward Kinematics
- Use Denavit-Hartenberg parameters to model robot arm
- For each joint:
  - Calculate transformation matrix using joint angle and DH parameters
  - Update overall transformation
- Return end-effector position (XYZ coordinates)

### Calculate Operational Space Error
- Compute true end-effector positions using forward kinematics
- Compute predicted end-effector positions using forward kinematics
- Calculate Euclidean distance between true and predicted positions
- Return average error across trajectory

## 4. Define Robot Dynamic Parameters
- Define dynamic parameters for each robot link (e.g., mass, center of mass, inertia)

## 5. Physics-informed Echo State Network (PC-ESN)
### Initialize Model
- Set model parameters (input/output sizes, reservoir size, spectral radius, sparsity, leak rate, learning rate, decay steps)
- Initialize input weights (W_in) as random matrix
- Initialize reservoir weights (W_res):
  - Incorporate dynamic parameters (mass, inertia) into weights
  - Apply sparsity and scale to desired spectral radius
- Initialize output weights (W_out) as zeros
- Initialize covariance matrices for output learning
- Initialize state vectors (input and reservoir states)

### Update Input Layer
- Update input state using input data and learning rule (e.g., GHL)

### Update Reservoir
- Compute reservoir activation using input and reservoir states
- Apply leaky integration with tanh activation
- Update reservoir state

### Update Output Weights
- Compute extended state (reservoir state + bias)
- For each output dimension:
  - Calculate prediction error (target - current prediction)
  - Compute Kalman gain using covariance and state
  - Update output weights using error and Kalman gain
  - Update covariance matrix efficiently
  - Update noise parameters for stability
- Return prediction errors

### Train Model
- Set training mode ON
- Reset internal states
- For each input-target pair:
  - Update input layer with current input
  - Update reservoir state
  - Update output weights with target data
- Set training mode OFF

## 6. Main Workflow
- Initialize random seed
- Load trajectory data
- Initialize PC-ESN model with specified parameters
- Train model on input-target trajectory pairs
- Evaluate model using normalized MSE and operational space error
- Save results to file (e.g., CSV)