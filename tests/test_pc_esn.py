import unittest
import torch
import sys
sys.path.append('models')

import pc_esn_model

class TestPCESN(unittest.TestCase):
    """
    Test suite for the updated PC_ESN implementation with physics-informed
    reservoir and per-output Bayesian learning.
    """
    def setUp(self):
        """Set up a default model instance and data for testing."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n--- Setting up test on device: {self.device} ---")
        
        # Parameters for a 7-DOF robot arm scenario
        self.n_inputs = 21   # 7 pos + 7 vel + 7 torques
        self.n_outputs = 14  # 7 next_pos + 7 next_vel
        self.n_reservoir = 50 # Smaller reservoir for faster tests
        self.ghl_eta_initial = 1e-3
        self.ghl_decay_steps = 100
        
        # Dummy dynamic parameters for testing
        self.dynamic_params = [
            {'mass': 1.0, 'com': [0.1, 0, 0], 'inertia': [0.01, 0.01, 0.01]} 
            for _ in range(7) # 7 robot links
        ]

        self.model = pc_esn_model.PC_ESN(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_reservoir=self.n_reservoir,
            spectral_radius=1.1,
            sparsity=0.9,
            leak_rate=0.3,
            ghl_eta=self.ghl_eta_initial,
            ghl_decay_steps=self.ghl_decay_steps,
            dh_params=None,  # Not used in this version, can be None
            dynamic_params=self.dynamic_params,
            device=self.device
        )

        # Dummy data for testing
        self.batch_size = 10
        self.X_data = torch.randn(self.batch_size, self.n_inputs, device=self.device)
        self.y_data = torch.randn(self.batch_size, self.n_outputs, device=self.device)

    def test_01_initialization(self):
        """Test if the model initializes with the correct dimensions and properties."""
        print("Running Test: Initialization")
        extended_state_size = self.n_reservoir + 1

        # Check weight matrix dimensions
        self.assertEqual(self.model.W_in.shape, (self.n_inputs, self.n_inputs))
        self.assertEqual(self.model.W_self.shape, (self.n_reservoir, self.n_inputs))
        self.assertEqual(self.model.W_res.shape, (self.n_reservoir, self.n_reservoir))
        
        # Check output layer parameter dimensions (now lists)
        self.assertEqual(len(self.model.W_out), self.n_outputs)
        self.assertEqual(self.model.W_out[0].shape, (extended_state_size,))
        
        self.assertEqual(len(self.model.V), self.n_outputs)
        self.assertEqual(self.model.V[0].shape, (extended_state_size, extended_state_size))
        
        self.assertEqual(len(self.model.alpha), self.n_outputs)
        self.assertEqual(len(self.model.beta), self.n_outputs)

        # Check device placement
        self.assertEqual(str(self.model.W_in.device), self.device)
        self.assertEqual(str(self.model.W_out[0].device), self.device)
        self.assertEqual(str(self.model.V[0].device), self.device)

        # Check specific properties
        self.assertTrue(torch.all(self.model.W_in == torch.tril(self.model.W_in)), "W_in should be lower triangular")
        
        # Check sparsity of W_res
        num_zero_elements = torch.sum(self.model.W_res == 0)
        total_elements = self.model.W_res.numel()
        sparsity_ratio = num_zero_elements / total_elements
        self.assertAlmostEqual(sparsity_ratio.item(), self.model.sparsity, delta=0.05, msg="W_res sparsity is incorrect")

    def test_02_ghl_update(self):
        """Test the GHL update rule for the input weights."""
        print("Running Test: GHL Update")
        w_in_before = self.model.W_in.clone()
        h_state_before = self.model.h_state.clone()
        
        # GHL update now requires a timestep `t`
        self.model._update_ghl(self.X_data[0], t=0)
        
        w_in_after = self.model.W_in
        h_state_after = self.model.h_state
        
        self.assertFalse(torch.equal(w_in_before, w_in_after), "W_in should be updated by GHL.")
        self.assertFalse(torch.equal(h_state_before, h_state_after), "h_state should be updated.")
        self.assertEqual(h_state_after.shape, (self.n_inputs,))

    def test_03_reservoir_update(self):
        """Test the reservoir state update."""
        print("Running Test: Reservoir Update")
        r_state_before = self.model.r_state.clone()
        
        # GHL must run first to update h_state
        self.model._update_ghl(self.X_data[0], t=0)
        self.model._update_reservoir()
        
        r_state_after = self.model.r_state
        
        self.assertFalse(torch.equal(r_state_before, r_state_after), "r_state should be updated.")
        self.assertEqual(r_state_after.shape, (self.n_reservoir,))

    def test_04_bayesian_output_update(self):
        """Test the per-output Bayesian update for the output layer."""
        print("Running Test: Bayesian Output Update")
        
        # Clone initial state of all Bayesian parameters (they are lists)
        w_out_before = [w.clone() for w in self.model.W_out]
        v_before = [v.clone() for v in self.model.V]
        alpha_before = [a.clone() for a in self.model.alpha]
        beta_before = [b.clone() for b in self.model.beta]

        # Run one update step
        self.model._update_ghl(self.X_data[0], t=0)
        self.model._update_reservoir()
        self.model._update_output_weights(self.y_data[0])

        # Check that all parameters have changed for each output dimension
        for i in range(self.n_outputs):
            self.assertFalse(torch.equal(w_out_before[i], self.model.W_out[i]), f"W_out[{i}] should be updated.")
            self.assertFalse(torch.equal(v_before[i], self.model.V[i]), f"V[{i}] should be updated.")
            self.assertGreater(self.model.alpha[i], alpha_before[i], f"alpha[{i}] should increase.")
            self.assertGreater(self.model.beta[i], 0, f"beta[{i}] should remain positive.")
            self.assertNotEqual(self.model.beta[i].item(), beta_before[i].item(), f"beta[{i}] should be updated.")

    def test_05_train_method(self):
        """Test the full online training loop and parameter re-initialization."""
        print("Running Test: Full Train Method")
        
        # Set some parameters to non-default values to test re-initialization
        self.model.W_out = [torch.randn_like(w) for w in self.model.W_out]
        
        w_in_before = self.model.W_in.clone()
        w_out_before_train_call = [w.clone() for w in self.model.W_out]

        # Train the model
        self.model.train(self.X_data, self.y_data)
        
        # W_in is updated continuously, so it should have changed.
        self.assertFalse(torch.equal(w_in_before, self.model.W_in), "W_in should be updated after training.")
        
        # W_out is re-initialized to zeros inside train() and then updated.
        # It should be different from the random values we set before the call.
        self.assertFalse(torch.equal(w_out_before_train_call[0], self.model.W_out[0]), "W_out should be updated after training.")
        
        # Check that the GHL learning rate has decayed correctly.
        expected_eta = self.ghl_eta_initial / (1 + (self.batch_size - 1) / self.ghl_decay_steps)
        self.assertAlmostEqual(self.model.ghl_eta, expected_eta, places=6, msg="ghl_eta should decay over time.")

    def test_06_predict_step_by_step(self):
        """Test step-by-step prediction mode, including variance output."""
        print("Running Test: Step-by-Step Prediction")
        
        # Train the model first to get non-zero weights
        self.model.train(self.X_data, self.y_data)
        
        w_out_before = [w.clone() for w in self.model.W_out]
        
        # Prediction now returns two values
        predictions, variances = self.model.predict_step_by_step(self.X_data)
        
        # Check output shapes
        self.assertEqual(predictions.shape, (self.batch_size, self.n_outputs))
        self.assertEqual(variances.shape, (self.batch_size, self.n_outputs))
        
        # Check that variances are non-negative
        self.assertTrue((variances >= 0).all(), "Variances must be non-negative.")
        
        # Check that prediction does not alter the model weights
        for i in range(self.n_outputs):
            self.assertTrue(torch.equal(w_out_before[i], self.model.W_out[i]), "Prediction should not change model weights.")

    def test_07_predict_full_trajectory(self):
        """Test the full trajectory (recursive) prediction mode."""
        print("Running Test: Full Trajectory Prediction")
        
        # Train the model first
        self.model.train(self.X_data, self.y_data)
        
        w_out_before = [w.clone() for w in self.model.W_out]
        
        # Prediction now returns two values
        predictions, variances = self.model.predict_full_trajectory(self.X_data)
        
        # Check output shape
        self.assertEqual(predictions.shape, (self.batch_size, self.n_outputs))
        self.assertEqual(variances.shape, (self.batch_size, self.n_outputs))
        
        # Check that variances are non-negative
        self.assertTrue((variances >= 0).all(), "Variances must be non-negative.")

        # Check that prediction does not alter the model weights
        for i in range(self.n_outputs):
            self.assertTrue(torch.equal(w_out_before[i], self.model.W_out[i]), "Prediction should not change model weights.")

    def test_08_dynamic_params_influence(self):
        """Test that dynamic_params affect the reservoir matrix W_res."""
        print("Running Test: Dynamic Parameters Influence")

        # Model with dynamic parameters (from setUp)
        w_res_with_params = self.model.W_res.clone()

        # Model without dynamic parameters
        model_no_params = pc_esn_model.PC_ESN(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_reservoir=self.n_reservoir,
            spectral_radius=1.1,
            sparsity=0.9,
            leak_rate=0.3,
            ghl_eta=self.ghl_eta_initial,
            ghl_decay_steps=self.ghl_decay_steps,
            dh_params=None,
            dynamic_params=[], # Empty list
            device=self.device
        )
        w_res_no_params = model_no_params.W_res.clone()
        
        # Because the creation of W_res is stochastic, we can't just check for inequality.
        # However, the diagonal should be different due to the line:
        # W[i, i] += mass * com[0] * inertia[0]
        # We will check if the diagonals are different.
        diag_with_params = torch.diag(w_res_with_params)
        diag_no_params = torch.diag(w_res_no_params)
        
        # To make the test deterministic, let's re-seed before creating each model
        torch.manual_seed(42)
        model1 = pc_esn_model.PC_ESN(n_inputs=self.n_inputs, n_outputs=self.n_outputs, n_reservoir=self.n_reservoir, spectral_radius=1.1, sparsity=0.9, leak_rate=0.3, ghl_eta=self.ghl_eta_initial, ghl_decay_steps=self.ghl_decay_steps, dh_params=None, dynamic_params=self.dynamic_params, device=self.device)
        
        torch.manual_seed(42)
        model2 = pc_esn_model.PC_ESN(n_inputs=self.n_inputs, n_outputs=self.n_outputs, n_reservoir=self.n_reservoir, spectral_radius=1.1, sparsity=0.9, leak_rate=0.3, ghl_eta=self.ghl_eta_initial, ghl_decay_steps=self.ghl_decay_steps, dh_params=None, dynamic_params=[], device=self.device)

        self.assertFalse(torch.equal(model1.W_res, model2.W_res), "W_res should be different when dynamic_params are provided.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)