import torch

class PC_ESN:
    def __init__(self, n_inputs, n_outputs, n_reservoir, spectral_radius,
                 sparsity, leak_rate, ghl_eta, ghl_decay_steps, dh_params,
                 dynamic_params, device='cuda'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.ghl_eta_initial = ghl_eta
        self.ghl_decay_steps = ghl_decay_steps
        self.dh_params = dh_params
        self.dynamic_params = dynamic_params  # Store the dynamic parameters
        self.device = device
        self.training = False  # Set to True during training

        self.extended_state_size = self.n_reservoir + 1  # +1 for bias

        # === Initialize Weights and Parameters ===
        self.W_in = torch.tril(torch.rand(n_inputs, n_inputs, device=device))
        self.W_self = (torch.rand(n_reservoir, n_inputs, device=device) - 0.5)
        self.W_res = self._create_reservoir_matrix().to(device)

        self.W_out = [torch.zeros(self.extended_state_size, device=device) for _ in range(n_outputs)] # w_0 = 0 vector
        self.V = [torch.eye(self.extended_state_size, device=device) for _ in range(n_outputs)]       # V_0 = Identity matrix

        self.alpha = [torch.tensor(0.0, device=device) for _ in range(n_outputs)] # alpha_0 = 0
        self.beta = [torch.tensor(1.0, device=device) for _ in range(n_outputs)]  # beta_0 = 1.0 (common choice)
        self.sample_counts = [0 for _ in range(n_outputs)] # Tracks number of samples seen for each output

        # === Initialize States ===
        self.h_state = torch.zeros(n_inputs, device=device)
        self.r_state = torch.zeros(n_reservoir, device=device)

    def _create_reservoir_matrix(self):
      """Creates the reservoir's internal weight matrix (W_res) with influence from dynamic parameters."""
      W = torch.rand(self.n_reservoir, self.n_reservoir, device=self.device) - 0.5
      W[torch.rand(*W.shape, device=self.device) < self.sparsity] = 0

      # Incorporate dynamic parameters into the reservoir matrix
      for i in range(len(self.dynamic_params)):
          mass = self.dynamic_params[i]['mass']
          com = self.dynamic_params[i]['com']
          inertia = self.dynamic_params[i]['inertia']

          # Example of incorporating dynamic parameters
          W[i, i] += mass * com[0] * inertia[0]  # Adjust as needed

      eigenvalues = torch.linalg.eigvals(W).abs()
      W *= self.spectral_radius / torch.max(eigenvalues)
      return W

    def _update_ghl(self, u, t):
        """Updates the self-organized layer weights using GHL."""
        self.h_state = self.W_in @ u
        dw_in = self.ghl_eta_initial * (torch.outer(u, self.h_state) -
                                torch.tril(torch.outer(self.h_state, self.h_state)) @ self.W_in)
        self.W_in += dw_in

    def _update_reservoir(self):
        """Updates the DYNAMIC part of the reservoir state, now influenced by physics."""
        pre_activation = self.W_self @ self.h_state + self.W_res @ self.r_state

        self.r_state = (1 - self.leak_rate) * self.r_state + torch.tanh(pre_activation)

    def _get_extended_state(self):
        """Constructs the state vector for the readout layer (dynamic state + bias)."""
        bias_tensor = torch.tensor([1.0], device=self.device)
        return torch.hstack((self.r_state, bias_tensor))

    def _get_h_state_for_prediction(self, u):
        """Calculates h_state without applying the GHL learning rule."""
        self.h_state = self.W_in @ u

    def _update_output_weights(self, target_vector):
      """
      Final corrected implementation: Combines the efficient matrix update with a
      numerically stable Bayesian learning rule. This avoids expensive inversions
      and ensures the noise parameter Beta remains positive.
      """
      c_t = self._get_extended_state()
      c_t_col = c_t.reshape(-1, 1)
      c_t_row = c_t.reshape(1, -1)
      tau_t = target_vector

      prediction_errors = torch.zeros(self.n_outputs, device=self.device)

      for i in range(self.n_outputs):
          # Retrieve previous state
          w_t_minus_1 = self.W_out[i]
          V_t_minus_1 = self.V[i]
          alpha_t_minus_1 = self.alpha[i]
          beta_t_minus_1 = self.beta[i]

          # --- Efficient & Stable Update ---

          # Calculate prediction error BEFORE the update
          pred_before_update = w_t_minus_1 @ c_t
          error = tau_t[i] - pred_before_update
          prediction_errors[i] = error

          # 1. Efficiently calculate the Kalman gain (k_t) and the denominator
          # This avoids the expensive and unstable torch.inverse() call
          V_c = V_t_minus_1 @ c_t_col
          denominator = 1.0 + (c_t_row @ V_c)

          # Ensure denominator is not zero to avoid division by zero
          if torch.abs(denominator) < 1e-8:
              denominator = 1e-8

          k_t = V_c / denominator

          # 2. Update the weights using the Kalman gain
          self.W_out[i] = w_t_minus_1 + k_t.flatten() * error

          # 3. Efficiently update the covariance matrix V
          # This is the Sherman-Morrison identity, which is much faster.
          self.V[i] = V_t_minus_1 - (k_t @ V_c.T)

          # 4. Update alpha (incrementally)
          self.alpha[i] = alpha_t_minus_1 + 0.5

          # 5. Update beta with a STABLE formula
          # This update is always positive and prevents Beta from becoming negative.
          self.beta[i] = beta_t_minus_1 + 0.5 * (error * error) / denominator

      return prediction_errors


    def train(self, X_data, y_data):
      """Trains the network online, one sample at a time."""
      self.training = True

      # Reset internal states before training
      self.h_state = torch.zeros(self.n_inputs, device=self.device)
      self.r_state = torch.zeros(self.n_reservoir, device=self.device)

      # Re-initialize Bayesian parameters for clean training
      self.w_out = [torch.zeros(self.extended_state_size, device=self.device) for _ in range(self.n_outputs)]
      self.v = [torch.eye(self.extended_state_size, device=self.device) for _ in range(self.n_outputs)]
      self.alpha = [torch.tensor(0.0, device=self.device) for _ in range(self.n_outputs)]
      self.beta = [torch.tensor(1.0, device=self.device) for _ in range(self.n_outputs)]
      self.sample_counts = [0 for _ in range(self.n_outputs)]

      for t in range(X_data.shape[0]):
          self.ghl_eta = self.ghl_eta_initial / (1 + t / self.ghl_decay_steps)

          # Store old W_in to calculate update norm
          w_in_before_update = self.W_in.clone()

          self._update_ghl(X_data[t], t)  # Updates h_state and W_in
          self._update_reservoir()  # Updates r_state using h_state
          self._update_output_weights(y_data[t])  # Updates W_out, V, alpha, beta

      self.training = False

    def predict_step_by_step(self, X_data):
      """Predicts outputs step-by-step, using ground truth for each next step."""
      self.training = False
      self.h_state = torch.zeros(self.n_inputs, device=self.device)
      self.r_state = torch.zeros(self.n_reservoir, device=self.device)
      X_data = X_data.to(self.device)
      predictions = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
      variances = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)

      for t in range(X_data.shape[0]):
          self._get_h_state_for_prediction(X_data[t])
          self._update_reservoir()
          c_t = self._get_extended_state()

          for i in range(self.n_outputs):
              predictions[t, i] = self.W_out[i] @ c_t
              variance = (self.beta[i] / self.alpha[i]) * (1 + c_t @ self.V[i] @ c_t)
              variances[t, i] = variance

      return predictions.cpu().numpy(), variances.cpu().numpy()

    def predict_full_trajectory(self, X_data):
        """Predicts a full trajectory recursively, using its own predictions as input."""
        self.training = False
        self.h_state = torch.zeros(self.n_inputs, device=self.device)
        self.r_state = torch.zeros(self.n_reservoir, device=self.device)
        X_data = X_data.to(self.device)
        predictions = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
        variances = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
        self.h_state = torch.zeros(self.n_inputs, device=self.device)
        self.r_state = torch.zeros(self.n_reservoir, device=self.device)
        current_input = X_data[0].clone()

        for t in range(X_data.shape[0]):
            self._get_h_state_for_prediction(current_input)
            self._update_reservoir()
            c_t = self._get_extended_state()

            prediction = torch.zeros(self.n_outputs, device=self.device)

            # Compute variance for each output dimension
            for i in range(self.n_outputs):
                prediction[i] = self.W_out[i] @ c_t
                var_i = (self.beta[i] / self.alpha[i]) * (1 + c_t.T @ self.V[i] @ c_t)
                variances[t, i] = var_i

            predictions[t, :] = prediction

            if t < X_data.shape[0] - 1:
                next_pos_vel = prediction
                torque_start_index = self.n_outputs
                next_torque = X_data[t + 1, torque_start_index:]
                current_input = torch.hstack((next_pos_vel, next_torque))

        return predictions.cpu().numpy(), variances.cpu().numpy()