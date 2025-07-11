import torch

class PC_ESN:
    """
    A PyTorch implementation of the Principal-Components Echo State Network (PC-ESN++)
    as described in "A Reservoir Computing Approach for Learning Forward Dynamics
    of Industrial Manipulators" (Polydoros & Nalpantidis, 2016).
    """
    def __init__(self, n_inputs, n_outputs, n_reservoir=300,
                 spectral_radius=1.1, sparsity=0.9, leak_rate=0.2,
                 ghl_eta=1e-4, ghl_decay_steps=1000, device='cpu'):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate # corresponds to ζ in the paper
        self.ghl_eta_initial = ghl_eta     # corresponds to η_t in the paper
        self.ghl_decay_steps = ghl_decay_steps
        self.device = device

        # === Initialize Weights and Parameters ===

        # 1. Self-organized layer (GHL)
        # This is W_in in the paper.
        self.W_in = torch.tril(torch.rand(n_inputs, n_inputs, device=device))
        
        # 2. Dynamic Reservoir (ESN)
        # W_self connects the self-organized layer to the reservoir.
        self.W_self = (torch.rand(n_reservoir, n_inputs, device=device) - 0.5)
        # W_res is the internal reservoir weight matrix.
        self.W_res = self._create_reservoir_matrix()
        
        # 3. Output layer (Iterative Bayesian Linear Regression)
        # w_0 is a zero vector. Bias is included in the state.
        self.W_out = torch.zeros(n_outputs, n_reservoir + 1, device=device)
        # V_0 = I. This is the covariance matrix V_t.
        self.V = [torch.eye(n_reservoir + 1, device=device) for _ in range(n_outputs)]
        # Initialize alpha and beta for the NIG prior (t-1)
        # The paper is not explicit on initial values, so a weak prior is used.
        self.alpha = [torch.tensor(1.0, device=device) for _ in range(n_outputs)]
        self.beta = [torch.tensor(1.0, device=device) for _ in range(n_outputs)]
        
        # === Initialize States ===
        self.h_state = torch.zeros(n_inputs, device=device)
        self.r_state = torch.zeros(n_reservoir, device=device)

    def _create_reservoir_matrix(self):
        """Creates the reservoir's internal weight matrix (W_res)."""
        W = torch.rand(self.n_reservoir, self.n_reservoir, device=self.device) - 0.5
        # Enforce sparsity
        W[torch.rand(*W.shape, device=self.device) < self.sparsity] = 0
        # Scale by spectral radius
        eigenvalues = torch.linalg.eigvals(W).abs()
        W *= self.spectral_radius / torch.max(eigenvalues)
        return W

    def _update_ghl(self, u):
        """Updates the self-organized layer weights using GHL."""
        # Eq. (1): h_{t+1} = W_t^in * u_t
        # Note: The paper uses h_{t+1}, but it's computed from u_t.
        self.h_state = self.W_in @ u
        # Eq. (2): ΔW = η * (u*h^T - LT[h*h^T]*W)
        dw_in = self.ghl_eta * (torch.outer(u, self.h_state) - torch.tril(torch.outer(self.h_state, self.h_state)) @ self.W_in)
        self.W_in += dw_in

    def _update_reservoir(self):
        """Updates the reservoir state using a leaky integrator."""
        pre_activation = self.W_self @ self.h_state + self.W_res @ self.r_state
        # Using a standard leaky-integrator form r_t+1 = (1-ζ)r_t + g(...) which is a
        # stable interpretation of the paper's text and equation.
        self.r_state = (1 - self.leak_rate) * self.r_state + torch.tanh(pre_activation)

    def _update_output_weights(self, target_vector):
        """
        Updates output weights using Iterative Bayesian Linear Regression.
        """
        # Concatenate bias term to the reservoir state. The paper refers to this
        # combined vector as c_{t+1}.
        c_t = torch.hstack((self.r_state, torch.tensor([1.0], device=self.device)))
        c_t_col = c_t.reshape(-1, 1)
        c_t_row = c_t.reshape(1, -1)

        for i in range(self.n_outputs):
            # Retrieve parameters from the previous time step (t-1)
            w_tm1 = self.W_out[i]
            V_tm1 = self.V[i]
            alpha_tm1 = self.alpha[i]
            beta_tm1 = self.beta[i]
            tau_t = target_vector[i] # Target value is τ_t in the paper

            # --- Update V (Covariance Matrix) ---
            # Based on V_t = (V_{t-1}^-1 + c_t*c_t^T)^-1, implemented efficiently
            # using the Sherman-Morrison formula.
            k_numerator = V_tm1 @ c_t_col
            k_denominator = 1 + c_t_row @ V_tm1 @ c_t_col
            k = k_numerator / k_denominator
            V_t = V_tm1 - k @ c_t_row @ V_tm1
            self.V[i] = V_t

            # --- Update w (Weights) ---
            # w_t = V_t * (V_{t-1}^-1 * w_{t-1} + c_t * τ_t).
            # This is equivalent to the efficient update below.
            error = tau_t - (w_tm1 @ c_t)
            w_t = w_tm1 + error * k.flatten()
            self.W_out[i] = w_t

            # --- Update alpha ---
            # The update is recursive: α_t = α_{t-1} + n/2.
            # For online learning, n=1 at each time step.
            self.alpha[i] = alpha_tm1 + 0.5

            # --- Update beta ---
            # β_t = 1/2 * (w_{t-1}V_{t-1}w_{t-1}^T + τ^2 - w_tV_tw_t^T).
            # Note: This update is not recursive as written in the paper, which is highly
            # unusual for an iterative Bayesian scheme. A recursive update like
            # β_t = β_{t-1} + ... is standard. I implement it as written, but add the
            # previous beta value, which is the standard interpretation for iterative updates.
            term1 = w_tm1.reshape(1, -1) @ V_tm1 @ w_tm1.reshape(-1, 1)
            term2 = tau_t**2
            term3 = w_t.reshape(1, -1) @ V_t @ w_t.reshape(-1, 1)
            self.beta[i] = beta_tm1 + 0.5 * (term1 + term2 - term3)

    def train(self, X_data, y_data):
        """Trains the network online, one sample at a time."""
        for t in range(X_data.shape[0]):
            # Update GHL learning rate based on time step 't'
            self.ghl_eta = self.ghl_eta_initial / (1 + t / self.ghl_decay_steps)
            self._update_ghl(X_data[t])
            self._update_reservoir()
            self._update_output_weights(y_data[t])

    def predict_step_by_step(self, X_data):
        """
        Predicts outputs step-by-step, using ground truth for each next step.
        """
        predictions = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
        # Reset state before prediction for consistency
        self.h_state = torch.zeros(self.n_inputs, device=self.device)
        self.r_state = torch.zeros(self.n_reservoir, device=self.device)
        
        for t in range(X_data.shape[0]):
            # Use ground truth as input, don't update weights
            self.h_state = self.W_in @ X_data[t]
            pre_activation = self.W_self @ self.h_state + self.W_res @ self.r_state
            self.r_state = (1 - self.leak_rate) * self.r_state + torch.tanh(pre_activation)
            
            c_t = torch.hstack((self.r_state, torch.tensor([1.0], device=self.device)))
            # The prediction is the mean of the posterior predictive distribution,
            # which is W_out @ c_t.
            predictions[t, :] = self.W_out @ c_t
            
        return predictions.cpu().numpy()

    def predict_full_trajectory(self, X_data):
        """
        Predicts a full trajectory recursively, using its own predictions as input.
        """
        predictions = torch.zeros(X_data.shape[0], self.n_outputs, device=self.device)
        # Reset state and initialize with first true input
        self.h_state = torch.zeros(self.n_inputs, device=self.device)
        self.r_state = torch.zeros(self.n_reservoir, device=self.device)
        
        # Assume inputs are [pos, vel, torque].
        # The number of position/velocity states is self.n_outputs.
        current_input = X_data[0].clone()
        
        for t in range(X_data.shape[0]):
            # Update GHL and reservoir based on current input (either true or predicted)
            self.h_state = self.W_in @ current_input
            pre_activation = self.W_self @ self.h_state + self.W_res @ self.r_state
            self.r_state = (1 - self.leak_rate) * self.r_state + torch.tanh(pre_activation)
            
            c_t = torch.hstack((self.r_state, torch.tensor([1.0], device=self.device)))
            prediction = self.W_out @ c_t
            predictions[t, :] = prediction

            # Recursive step: prepare the input for the next time step
            if t < X_data.shape[0] - 1:
                # Use own predicted state (pos, vel) with the next true action (torque)
                next_pos_vel = prediction
                # Assuming the torques are the last part of the input vector
                torque_start_index = self.n_outputs
                next_torque = X_data[t+1, torque_start_index:]
                current_input = torch.hstack((next_pos_vel, next_torque))

        return predictions.cpu().numpy()