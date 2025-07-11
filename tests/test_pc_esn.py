import unittest
import torch

import pc_esn_model

class TestPCESN(unittest.TestCase):
    """
    Test suite for the improved PC_ESN++ implementation.
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
        
        self.model = pc_esn_model.PC_ESN(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_reservoir=self.n_reservoir,
            spectral_radius=1.1,
            sparsity=0.9,
            leak_rate=0.3,
            ghl_eta=self.ghl_eta_initial,
            ghl_decay_steps=self.ghl_decay_steps,
            device=self.device
        )

        self.model.ghl_eta = self.ghl_eta_initial
        
        # Dummy data for testing
        self.batch_size = 10
        self.X_data = torch.randn(self.batch_size, self.n_inputs, device=self.device)
        self.y_data = torch.randn(self.batch_size, self.n_outputs, device=self.device)

    def test_01_initialization(self):
        """Test if the model initializes with the correct dimensions and properties."""
        print("Running Test: Initialization")
        # Check weight matrix dimensions
        self.assertEqual(self.model.W_in.shape, (self.n_inputs, self.n_inputs))
        self.assertEqual(self.model.W_self.shape, (self.n_reservoir, self.n_inputs))
        self.assertEqual(self.model.W_res.shape, (self.n_reservoir, self.n_reservoir))
        self.assertEqual(self.model.W_out.shape, (self.n_outputs, self.n_reservoir + 1))

        # Check Bayesian parameter dimensions
        self.assertEqual(len(self.model.V), self.n_outputs)
        self.assertEqual(self.model.V[0].shape, (self.n_reservoir + 1, self.n_reservoir + 1))
        self.assertEqual(len(self.model.alpha), self.n_outputs)
        self.assertEqual(len(self.model.beta), self.n_outputs)

        # Check device placement
        self.assertEqual(str(self.model.W_in.device), self.device)
        self.assertEqual(str(self.model.W_out.device), self.device)
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
        
        self.model._update_ghl(self.X_data[0])
        
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
        self.model._update_ghl(self.X_data[0])
        self.model._update_reservoir()
        
        r_state_after = self.model.r_state
        
        self.assertFalse(torch.equal(r_state_before, r_state_after), "r_state should be updated.")
        self.assertEqual(r_state_after.shape, (self.n_reservoir,))

    def test_04_bayesian_output_update(self):
        """Test the iterative Bayesian update for the output weights and noise parameters."""
        print("Running Test: Bayesian Output Update")
        # Get initial state of all Bayesian parameters
        w_out_before = self.model.W_out.clone()
        v_before = [v.clone() for v in self.model.V]
        alpha_before = [a.clone() for a in self.model.alpha]
        beta_before = [b.clone() for b in self.model.beta]

        # Run one update step
        self.model._update_ghl(self.X_data[0])
        self.model._update_reservoir()
        self.model._update_output_weights(self.y_data[0])

        # Check that all parameters have changed
        self.assertFalse(torch.equal(w_out_before, self.model.W_out), "W_out should be updated.")
        for i in range(self.n_outputs):
            self.assertFalse(torch.equal(v_before[i], self.model.V[i]), f"V[{i}] should be updated.")
            self.assertGreater(self.model.alpha[i], alpha_before[i], f"alpha[{i}] should increase.")
            self.assertNotEqual(self.model.beta[i].item(), beta_before[i].item(), f"beta[{i}] should be updated.")
    
    def test_05_train_method(self):
        """Test the full training loop."""
        print("Running Test: Full Train Method")
        w_in_before = self.model.W_in.clone()
        w_out_before = self.model.W_out.clone()
        
        self.model.train(self.X_data, self.y_data)
        
        # Check that weights have been updated after training on a batch
        self.assertFalse(torch.equal(w_in_before, self.model.W_in), "W_in should be updated after training.")
        self.assertFalse(torch.equal(w_out_before, self.model.W_out), "W_out should be updated after training.")
        
        # Check that the GHL learning rate has decayed
        expected_eta = self.ghl_eta_initial / (1 + (self.batch_size - 1) / self.ghl_decay_steps)
        self.assertAlmostEqual(self.model.ghl_eta, expected_eta, msg="ghl_eta should decay over time.")

    def test_06_predict_step_by_step(self):
        """Test the step-by-step prediction mode."""
        print("Running Test: Step-by-Step Prediction")
        # Train the model first to get non-zero weights
        self.model.train(self.X_data, self.y_data)
        
        w_out_before = self.model.W_out.clone()
        
        predictions = self.model.predict_step_by_step(self.X_data)
        
        # Check output shape
        self.assertEqual(predictions.shape, (self.batch_size, self.n_outputs))
        
        # Check that prediction does not alter the model weights
        self.assertTrue(torch.equal(w_out_before, self.model.W_out), "Prediction should not change model weights.")

    def test_07_predict_full_trajectory(self):
        """Test the full trajectory (recursive) prediction mode."""
        print("Running Test: Full Trajectory Prediction")
        # Train the model first
        self.model.train(self.X_data, self.y_data)
        
        w_out_before = self.model.W_out.clone()
        
        predictions = self.model.predict_full_trajectory(self.X_data)
        
        # Check output shape
        self.assertEqual(predictions.shape, (self.batch_size, self.n_outputs))
        
        # Check that prediction does not alter the model weights
        self.assertTrue(torch.equal(w_out_before, self.model.W_out), "Prediction should not change model weights.")

if __name__ == '__main__':
    # To run the tests, save the PC_ESN class in a file named 'pc_esn_model.py'
    # and this test script in another file, then run this script.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)