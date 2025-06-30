import torch

class PC_ESN:
    """
    A PyTorch implementation of the Principal-Components Echo State Network (PC-ESN++).
    The logic is based on the provided pc-esn++implementation.pdf.
    """
    def __init__(self, n_inputs, n_outputs, n_reservoir=300,
                 spectral_radius=1.1, sparsity=0.9, leak_rate=0.2,
                 ghl_eta=1e-4, device='cpu'):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.ghl_eta = ghl_eta
        self.device = device

        # Initialize Weights
        # 1. Self-organized layer (GHL)
        self.W_in = torch.tril(torch.rand(n_inputs, n_inputs, device=device))
        
        # 2. Dynamic Reservoir (ESN)
        self.W_self = (torch.rand(n_reservoir, n_inputs, device=device) - 0.5)
        self.W_res = self._create_reservoir_matrix()
        
        # 3. Output layer (Bayesian Linear Regression)
        self.W_out = torch.zeros(n_outputs, n_reservoir + 1, device=device)
        self.P = [torch.eye(n_reservoir + 1, device=device) * 1e-6 for _ in range(n_outputs)]
        
        # Initialize States
        self.h_state = torch.zeros(n_inputs, device=device)
        self.r_state = torch.zeros(n_reservoir, device=device)

    def _create_reservoir_matrix(self):
        W = torch.rand(self.n_reservoir, self.n_reservoir, device=self.device) - 0.5
        W[torch.rand(*W.shape, device=self.device) < self.sparsity] = 0
        eigenvalues = torch.linalg.eigvals(W).abs()
        W *= self.spectral_radius / torch.max(eigenvalues)
        return W

    def _update_ghl(self, u):
        self.h_state = self.W_in @ u
        dw_in = self.ghl_eta * (torch.outer(u, self.h_state) - torch.tril(torch.outer(self.h_state, self.h_state)) @ self.W_in)
        self.W_in += dw_in

    def _update_reservoir(self):
        pre_activation = self.W_self @ self.h_state + self.W_res @ self.r_state
        self.r_state = (1 - self.leak_rate) * self.r_state + self.leak_rate * torch.tanh(pre_activation)

    def _update_output_weights(self, target_vector):
        c_t = torch.hstack((self.r_state, torch.tensor([1.0], device=self.device)))
        for i in range(self.n_outputs):
            c_t_col = c_t.reshape(-1, 1)
            P_i = self.P[i]
            k = (P_i @ c_t_col) / (1 + c_t_col.T @ P_i @ c_t_col)
            self.W_out[i] += (k * (target_vector[i] - self.W_out[i] @ c_t)).flatten()
            self.P[i] -= k @ c_t_col.T @ P_i

    def train(self, X_data, y_data):
        for t in range(X_data.shape[0]):
            self._update_ghl(X_data[t])
            self._update_reservoir()
            self._update_output_weights(y_data[t])

    def predict_step_by_step(self, X_data):
        predictions = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
        # Reset state before prediction
        self.h_state = torch.zeros(self.n_inputs, device=self.device)
        self.r_state = torch.zeros(self.n_reservoir, device=self.device)
        for t in range(X_data.shape[0]):
            self._update_ghl(X_data[t])
            self._update_reservoir()
            c_t = torch.hstack((self.r_state, torch.tensor([1.0], device=self.device)))
            predictions[t, :] = self.W_out @ c_t
        return predictions.cpu().numpy()

    def predict_full_trajectory(self, X_data):
        predictions = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
        # Reset state and initialize with first true input
        self.h_state = torch.zeros(self.n_inputs, device=self.device)
        self.r_state = torch.zeros(self.n_reservoir, device=self.device)
        
        current_input = X_data[0]
        
        for t in range(X_data.shape[0]):
            self._update_ghl(current_input)
            self._update_reservoir()
            c_t = torch.hstack((self.r_state, torch.tensor([1.0], device=self.device)))
            prediction = self.W_out @ c_t
            predictions[t, :] = prediction

            # Recursive step for next input
            if t < X_data.shape[0] - 1:
                # Use own predicted state with next true torque
                next_pos_vel = prediction
                next_torque = X_data[t+1, 14:]
                current_input = torch.hstack((next_pos_vel, next_torque))

        return predictions.cpu().numpy()