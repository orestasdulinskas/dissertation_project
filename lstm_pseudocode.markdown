# Pseudo Code for LSTM Implementation for KUKA Manipulator Dynamics

This pseudo code outlines the implementation of an LSTM model for predicting the dynamics of a KUKA robotic manipulator, as derived from the provided Jupyter notebook.

---

## 1. Setup and Initialization
- Import required libraries: `torch`, `numpy`, `random`, `scipy.io`, `sklearn.preprocessing`, `optuna`
- Set random seed for reproducibility
- Initialize device (CUDA if available, else CPU)
- Set hyperparameters for optimization (to be tuned via Optuna):
  - Number of LSTM layers
  - Hidden size
  - Learning rate
  - Dropout rate
  - Sequence length
  - Number of training epochs

---

## 2. Data Ingestion
- Load Baxter manipulator dataset (`KukaDirectDynamics.mat`) using `scipy.io.loadmat`
- Extract trajectory data (keys starting with `kukatraj`)
- Store trajectories in a dictionary and compute total samples
- Split trajectories into training (first 8 trajectories) and validation (remaining 2)

---

## 3. Data Preprocessing
- Concatenate training trajectories into a single array
- Split data into inputs `X` (positions, velocities, torques) and outputs `y` (next positions, velocities)
- Apply `StandardScaler` to scale `X` and `y` for training
- Define function `create_sequences(X, y, seq_length)`:
  - For each index `i` from 0 to `len(X) - seq_length`:
    - Append `X[i:i+seq_length]` to `X_seq`
    - Append `y[i+seq_length-1]` to `y_seq`
  - Return `X_seq`, `y_seq` as numpy arrays

---

## 4. Forward Kinematics and Evaluation Metrics
- Define function `get_transformation_matrix(theta, d, a, alpha)`:
  - Compute cosine and sine of `theta` and `alpha`
  - Construct 4x4 Denavit-Hartenberg transformation matrix
  - Return matrix
- Define function `forward_kinematics(joint_positions)`:
  - Convert input tensor to numpy if needed
  - Define DH parameters for Baxter (7 joints)
  - Initialize identity transformation matrix `T`
  - For each joint:
    - Extract DH parameters (`alpha`, `a`, `d`, `theta`)
    - Compute transformation matrix using `get_transformation_matrix`
    - Update `T` by matrix multiplication
  - Return end-effector XYZ position (`T[:3, 3]`)
- Define function `calculate_op_space_error(y_true, y_pred)`:
  - Extract joint positions from `y_true` and `y_pred`
  - Apply `forward_kinematics` to compute true and predicted XYZ coordinates
  - Compute Euclidean distances between true and predicted coordinates
  - Return mean Euclidean error
- Define function `euclidean_error(y_true, y_pred)`:
  - Apply `forward_kinematics` to compute XYZ coordinates for each time step
  - Return Euclidean distances for each time step
- Define function `nMSE(y_true, y_pred)`:
  - Compute mean squared error per feature
  - Normalize by variance of `y_true` (handle zero variance)
  - Return normalized MSE

---

## 5. LSTM Model Definition
- Define class `LSTMModel` (inherits `nn.Module`):
  - Initialize:
    - LSTM layer with `input_size`, `hidden_size`, `num_layers`, `dropout_rate`, `batch_first=True`
    - Linear layer mapping `hidden_size` to `output_size`
  - Forward pass:
    - Pass input through LSTM
    - Take output from last time step
    - Pass through linear layer
    - Return predictions

---

## 6. Prediction Functions
- Define function `predict_step_by_step_lstm(model, X_test, y_test, seq_length)`:
  - Set model to evaluation mode
  - Create sequences from `X_test` using `create_sequences`
  - Convert sequences to tensor and move to device
  - Initialize zero array for predictions
  - With no gradient computation:
    - Predict outputs for sequences
  - Pad predictions to match original trajectory length
  - Return predictions
- Define function `predict_full_trajectory_lstm(model, X_data, seq_length)`:
  - Set model to evaluation mode
  - Initialize zero tensor for predictions
  - Start with initial sequence from `X_data`
  - With no gradient computation:
    - For each time step `t` from 0 to `n_samples - seq_length`:
      - Predict next state
      - Store prediction
      - Update sequence with predicted state and true torques
  - Return predictions as numpy array

---

## 7. Training Function
- Define function `train_lstm_model(model, data_loader, learning_rate, epochs)`:
  - Initialize MSE loss criterion
  - Initialize AdamW optimizer with `learning_rate`
  - Initialize `ReduceLROnPlateau` scheduler
  - Initialize `GradScaler` for mixed precision training
  - Set model to training mode
  - For each epoch:
    - Initialize epoch loss
    - For each batch in `data_loader`:
      - Move batch to device
      - Zero gradients
      - With autocast:
        - Forward pass to get predictions
        - Compute MSE loss
      - Scale loss, backpropagate, update optimizer and scaler
      - Accumulate batch loss
    - Compute average epoch loss
    - Update scheduler with epoch loss
  - Return trained model

---

## 8. Hyperparameter Optimization
- Define function `objective(trial)` for Optuna:
  - Suggest hyperparameters:
    - `n_layers`: 1 to 3
    - `hidden_size`: 64 to 512 (step 32)
    - `learning_rate`: 1e-5 to 1e-3 (log scale)
    - `dropout`: 0.1 to 0.7 (step 0.1)
    - `sequence_length`: 10 to 80 (step 10)
    - `epochs`: 50 to 200 (step 25)
  - Create training sequences from pre-scaled training data
  - Convert sequences to tensors and create `DataLoader`
  - Initialize `LSTMModel` with suggested parameters
  - Train model using `train_lstm_model`
  - For each validation trajectory:
    - Scale validation inputs using training scaler
    - Predict step-by-step
    - Inverse transform predictions to original scale
    - Align predictions and true outputs
    - Compute nMSE for joint positions
  - Return average nMSE across validation trajectories
- Create Optuna study (minimize objective)
- Optimize with 40 trials

---

## 9. Results Analysis
- Load results into DataFrame
- Compute mean and standard deviation of metrics:
  - Training time
  - Step-by-step position nMSE
  - Step-by-step velocity nMSE
  - Step-by-step Euclidean error
  - Full trajectory position nMSE
  - Full trajectory velocity nMSE
  - Full trajectory Euclidean error
- For prediction horizons (100, 200, ..., 1000):
  - Compute mean and std of position nMSE and Euclidean error
- Print formatted results tables