import unittest
import torch
import numpy as np
import sys
sys.path.append('./pc_esn')

import model

class TestPCESN(unittest.TestCase):
    """
    Test suite for the user's PC_ESN implementation.
    """
    def setUp(self):
        """Set up a default model instance and data for testing."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n--- Setting up test on device: {self.device} ---")
        
        # parameters for a 7-DOF robot arm scenario
        self.n_inputs = 21  # 7 pos + 7 vel + 7 torques
        self.n_outputs = 14 # 7 next_pos + 7 next_vel
        self.n_reservoir = 50 # Smaller reservoir for faster tests
        
        self.model = model.PC_ESN(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_reservoir=self.n_reservoir,
            spectral_radius=1.1,
            sparsity=0.9,
            leak_rate=0.3,
            device=self.device
        )
        
        # dummy data for testing
        self.batch_size = 10
        self.X_data = torch.randn(self.batch_size, self.n_inputs, device=self.device)
        self.y_data = torch.randn(self.batch_size, self.n_outputs, device=self.device)

    def test_01_initialization(self):
        """Test if the model initializes with the correct dimensions and on the correct device."""
        print("Running Test: Initialization")
        self.assertEqual(self.model.W_in.shape, (self.n_inputs, self.n_inputs))
        self.assertEqual(self.model.W_self.shape, (self.n_reservoir, self.n_inputs))
        self.assertEqual(self.model.W_res.shape, (self.n_reservoir, self.n_reservoir))
        self.assertEqual(self.model.W_out.shape, (self.n_outputs, self.n_reservoir + 1))
        
        self.assertEqual(len(self.model.P), self.n_outputs)
        self.assertEqual(self.model.P[0].shape, (self.n_reservoir + 1, self.n_reservoir + 1))
        
        self.assertEqual(str(self.model.W_in.device), self.device)
        self.assertEqual(str(self.model.r_state.device), self.device)
        print("Initialization test passed.")

    def test_02_create_reservoir_matrix(self):
        """Test the reservoir matrix creation for sparsity and spectral radius."""
        print("Running Test: Reservoir Matrix Creation")
        W_res = self.model.W_res
        self.assertEqual(W_res.shape, (self.n_reservoir, self.n_reservoir))

        # test spectral radius
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        radius = torch.max(eigenvalues)
        self.assertAlmostEqual(radius.item(), self.model.spectral_radius, places=4)

        # test sparsity
        num_zeros = (W_res == 0).sum().item()
        total_elements = W_res.numel()
        actual_sparsity = num_zeros / total_elements
        self.assertAlmostEqual(actual_sparsity, self.model.sparsity, delta=0.05)
        print("Reservoir matrix creation test passed.")

    def test_03_internal_updates(self):
        """Test the internal state and weight update functions."""
        print("Running Test: Internal Update Functions")
        u = self.X_data[0]
        target = self.y_data[0]

        # store initial states and weights
        W_in_before = self.model.W_in.clone()
        r_state_before = self.model.r_state.clone()
        W_out_before = self.model.W_out.clone()
        P_before = self.model.P[0].clone()

        # --- Test GHL update ---
        self.model._update_ghl(u)
        self.assertFalse(torch.equal(self.model.W_in, W_in_before), "GHL weights W_in did not update.")
        
        # --- Test Reservoir update ---
        self.model._update_reservoir()
        self.assertFalse(torch.equal(self.model.r_state, r_state_before), "Reservoir state r_state did not update.")
        
        # --- Test Output Weight update ---
        self.model._update_output_weights(target)
        self.assertFalse(torch.equal(self.model.W_out, W_out_before), "Output weights W_out did not update.")
        self.assertFalse(torch.equal(self.model.P[0], P_before), "Covariance matrix P did not update.")
        print("Internal update functions test passed.")

    def test_04_train_method(self):
        """Test the main training loop."""
        print("Running Test: Full Train Method")
        W_out_before = self.model.W_out.clone()
        
        self.model.train(self.X_data, self.y_data)
        
        self.assertFalse(torch.equal(self.model.W_out, W_out_before), "W_out should change after training.")
        print("Train method test passed.")

    def test_05_predict_step_by_step(self):
        """Test step-by-step prediction mode."""
        print("Running Test: Step-by-Step Prediction")
        W_out_before = self.model.W_out.clone()
        r_state_before = self.model.r_state.clone()

        predictions = self.model.predict_step_by_step(self.X_data)
        
        self.assertEqual(predictions.shape, (self.batch_size, self.n_outputs))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(torch.equal(self.model.W_out, W_out_before), "Weights should not change during prediction.")
        self.assertFalse(torch.equal(self.model.r_state, r_state_before), "Reservoir state should change during prediction.")
        print("Step-by-step prediction test passed.")
        
    def test_06_predict_full_trajectory(self):
        """Test full trajectory (recursive) prediction mode."""
        print("Running Test: Full Trajectory Prediction")
        W_out_before = self.model.W_out.clone()
        
        # pre-train the model slightly to get non-zero output weights
        self.model.train(self.X_data, self.y_data)
        
        predictions = self.model.predict_full_trajectory(self.X_data)
        
        self.assertEqual(predictions.shape, (self.batch_size, self.n_outputs))
        self.assertIsInstance(predictions, np.ndarray)
        # check that the model is actually making non-zero predictions
        self.assertFalse(np.all(predictions == 0), "Predictions should not be all zeros after training.")
        print("Full trajectory prediction test passed.")
        
    def test_07_learning_process(self):
        """A simple integration test to see if loss decreases over a few training epochs."""
        print("Running Test: Learning Process")
        
        # simple target that the model should learn to approximate
        target = torch.ones(self.batch_size, self.n_outputs, device=self.device) * 0.5
        
        # measure initial loss
        predictions_before = self.model.predict_step_by_step(self.X_data)
        loss_before = np.mean((predictions_before - target.cpu().numpy())**2)
        
        # train for a few epochs
        for _ in range(5):
            self.model.train(self.X_data, target)
            
        # measure final loss
        predictions_after = self.model.predict_step_by_step(self.X_data)
        loss_after = np.mean((predictions_after - target.cpu().numpy())**2)
        
        print(f"Loss before training: {loss_before:.6f}")
        print(f"Loss after training:  {loss_after:.6f}")
        self.assertLess(loss_after, loss_before, "Loss should decrease after training.")
        print("Learning process test passed.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
