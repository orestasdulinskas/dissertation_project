import torch

class PC_ESN:
    """
    A PyTorch implementation of the Principal-Components Echo State Network (PC-ESN++)
    with physics injection into the reservoir update equation.
    """
    def __init__(self, n_inputs, n_outputs, n_reservoir=30,
                spectral_radius=1.1, sparsity=0.9, leak_rate=0.8,
                ghl_eta=1e-4, ghl_decay_steps=1000,
                 device='cuda'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.ghl_eta_initial = ghl_eta
        self.ghl_decay_steps = ghl_decay_steps
        self.device = device
        self.training = False # Set to True during training for noise injection
        
        self.extended_state_size = self.n_reservoir + 1 # +1 for bias
        
        # === Initialize Weights and Parameters ===
        self.W_in = torch.tril(torch.rand(n_inputs, n_inputs, device=device))
        self.W_self = (torch.rand(n_reservoir, n_inputs, device=device) - 0.5)
        self.W_res = self._create_reservoir_matrix().to(device)
        
        # REVERTED: W_out and V are back to their original size
        self.W_out = torch.zeros(n_outputs, self.extended_state_size, device=device)
        self.V = [torch.eye(self.extended_state_size, device=device) for _ in range(n_outputs)]
        self.alpha = [torch.tensor(1.0, device=device) for _ in range(n_outputs)]
        self.beta = [torch.tensor(1.0, device=device) for _ in range(n_outputs)]
        
        # === Initialize States ===
        self.h_state = torch.zeros(n_inputs, device=device)
        self.r_state = torch.zeros(n_reservoir, device=device)

    def _create_reservoir_matrix(self):
        """Creates the reservoir's internal weight matrix (W_res)."""
        W = torch.rand(self.n_reservoir, self.n_reservoir, device=self.device) - 0.5
        W[torch.rand(*W.shape, device=self.device) < self.sparsity] = 0
        eigenvalues = torch.linalg.eigvals(W).abs()
        W *= self.spectral_radius / torch.max(eigenvalues)
        return W
    
    def _update_ghl(self, u):
        """Updates the self-organized layer weights using GHL."""
        self.h_state = self.W_in @ u
        dw_in = self.ghl_eta * (torch.outer(u, self.h_state) - torch.tril(torch.outer(self.h_state, self.h_state)) @ self.W_in)
        self.W_in += dw_in

    def _update_reservoir(self):
        """Updates the DYNAMIC part of the reservoir state, now influenced by physics."""
        pre_activation = self.W_self @ self.h_state + self.W_res @ self.r_state
        
        self.r_state = (1 - self.leak_rate) * self.r_state + torch.tanh(pre_activation)

    def _get_extended_state(self):
        """Constructs the state vector for the readout layer (dynamic state + bias)."""
        bias_tensor = torch.tensor([1.0], device=self.device)
        # REVERTED: Only concatenates the dynamic state and the bias term
        return torch.hstack((self.r_state, bias_tensor))

    def _get_h_state_for_prediction(self, u):
        """Calculates h_state without applying the GHL learning rule."""
        self.h_state = self.W_in @ u

    def _update_output_weights(self, target_vector):
        """Updates output weights using Iterative Bayesian Linear Regression."""
        c_t = self._get_extended_state()
        c_t_col = c_t.reshape(-1, 1)
        c_t_row = c_t.reshape(1, -1)
        for i in range(self.n_outputs):
            w_tm1, V_tm1 = self.W_out[i], self.V[i]
            alpha_tm1, beta_tm1 = self.alpha[i], self.beta[i]
            tau_t = target_vector[i]
        
            k_numerator = V_tm1 @ c_t_col
            k_denominator = 1 + c_t_row @ V_tm1 @ c_t_col
            k = k_numerator / k_denominator
            V_t = V_tm1 - k @ c_t_row @ V_tm1
            self.V[i] = V_t
        
            error = tau_t - (w_tm1 @ c_t)
            w_t = w_tm1 + error * k.flatten()
            self.W_out[i] = w_t
        
            self.alpha[i] = alpha_tm1 + 0.5
        
            term1 = w_tm1.reshape(1, -1) @ V_tm1 @ w_tm1.reshape(-1, 1)
            term2 = tau_t**2
            term3 = w_t.reshape(1, -1) @ V_t @ w_t.reshape(-1, 1)
            self.beta[i] = 0.5 * (term1 + term2 - term3)

    def train(self, X_data, y_data):
        """Trains the network online, one sample at a time."""
        self.training = True
        for t in range(X_data.shape[0]):
            self.ghl_eta = self.ghl_eta_initial / (1 + t / self.ghl_decay_steps)
            self._update_ghl(X_data[t])
            self._update_reservoir()
            self._update_output_weights(y_data[t])
            self.training = False

    def predict_step_by_step(self, X_data):
        """Predicts outputs step-by-step, using ground truth for each next step."""
        self.training = False
        X_data = X_data.to(self.device)
        predictions = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
        self.h_state = torch.zeros(self.n_inputs, device=self.device)
        self.r_state = torch.zeros(self.n_reservoir, device=self.device)

        for t in range(X_data.shape[0]):
            self._get_h_state_for_prediction(X_data[t])
            
            self._update_reservoir()
            c_t = self._get_extended_state()
            predictions[t, :] = self.W_out @ c_t
        
        return predictions.cpu().numpy()

    def predict_full_trajectory(self, X_data):
        """Predicts a full trajectory recursively, using its own predictions as input."""
        self.training = False
        X_data = X_data.to(self.device)
        predictions = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
        self.h_state = torch.zeros(self.n_inputs, device=self.device)
        self.r_state = torch.zeros(self.n_reservoir, device=self.device)
        current_input = X_data[0].clone()

        for t in range(X_data.shape[0]):
            self._get_h_state_for_prediction(current_input)
            self._update_reservoir()
        
            c_t = self._get_extended_state()
            prediction = self.W_out @ c_t
            predictions[t, :] = prediction
        
            if t < X_data.shape[0] - 1:
                next_pos_vel = prediction
                torque_start_index = self.n_outputs
                next_torque = X_data[t+1, torque_start_index:]
                current_input = torch.hstack((next_pos_vel, next_torque))
        
        return predictions.cpu().numpy()