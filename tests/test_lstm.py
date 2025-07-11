import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import lstm_model

# Define the target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_sequences(X_data, y_data, seq_length):
    """
    Creates sequences for LSTM training.
    Input shape: (num_samples, num_features)
    Output shape: (num_sequences, seq_length, num_features)
    """
    X_seq, y_seq = [], []
    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i : (i + seq_length)])
        y_seq.append(y_data[i + seq_length -1]) # Target is the state at the end of the sequence
    return np.array(X_seq), np.array(y_seq)

class LSTMModel(nn.Module):
    """
    A standard LSTM network for time-series prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0 # Dropout is not applied if num_layers is 1
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # We only use the output from the last time step
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step_out)
        return predictions

def train_lstm_model(model, data_loader, learning_rate, epochs=50):
    """Function to train the LSTM model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train() # Set the model to training mode
    for epoch in range(epochs):
        for seq, labels in data_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
    return model

def predict_step_by_step_lstm(model, X_test_np, y_test_np, seq_length):
    """
    Predicts outputs one step ahead, using ground truth for each input.
    """
    model.eval() # Set the model to evaluation mode
    n_samples = X_test_np.shape[0]
    n_outputs = y_test_np.shape[1]

    # Create input sequences from the ground truth test data
    X_seq_np, _ = create_sequences(X_test_np, y_test_np, seq_length)
    if X_seq_np.shape[0] == 0: # Handle case where test data is too short
        return np.zeros((n_samples, n_outputs))
        
    X_seq_t = torch.from_numpy(X_seq_np).float().to(device)

    with torch.no_grad():
        predictions_seq = model(X_seq_t).cpu().numpy()

    # Pad the start of the predictions to match the original trajectory length
    padded_predictions = np.zeros((n_samples, n_outputs))
    # The first prediction corresponds to the state at index `seq_length`
    padded_predictions[seq_length:] = predictions_seq
    
    return padded_predictions

def predict_full_trajectory_lstm(model, X_data_t, seq_length):
    """
    Predicts a full trajectory recursively (open-loop).
    Uses its own predictions as input for subsequent steps.
    """
    model.eval() # Set the model to evaluation mode
    n_samples = X_data_t.shape[0]
    n_features = X_data_t.shape[1]
    n_outputs = 14  # As per the function's hardcoded value; 7 positions, 7 velocities
    predictions = torch.zeros(n_samples, n_outputs, device=device)

    # Initial sequence from ground truth
    current_sequence = X_data_t[0:seq_length, :].reshape(1, seq_length, -1)

    with torch.no_grad():
        for t in range(n_samples - seq_length):
            # Predict the next state
            pred_next_state = model(current_sequence)
            predictions[t + seq_length, :] = pred_next_state

            # Prepare the next sequence
            # Get the next true action (torques)
            next_torques = X_data_t[t + seq_length, 14:]

            # Create the next feature vector using the predicted state and true action
            next_feature_vector = torch.hstack((pred_next_state.flatten(), next_torques))

            # Update the sequence: drop the oldest, append the new
            next_sequence_features = torch.vstack((current_sequence.squeeze(0)[1:], next_feature_vector))
            current_sequence = next_sequence_features.reshape(1, seq_length, -1)

    return predictions.cpu().numpy()


class TestLSTMSuite(unittest.TestCase):
    """
    Test suite for the LSTM model, sequence creation, training, and prediction.
    """

    def setUp(self):
        """Set up common parameters and data for tests."""
        print(f"\n--- Setting up test on device: {device} ---")
        self.seq_length = 5
        self.n_features = 10
        self.n_outputs = 4
        self.n_samples = 20
        self.batch_size = 4
        
        # Model parameters
        self.input_size = self.n_features
        self.hidden_size = 32
        self.num_layers = 2
        self.output_size = self.n_outputs
        self.dropout_rate = 0.1

        # Create dummy data
        self.X_data_np = np.random.randn(self.n_samples, self.n_features)
        self.y_data_np = np.random.randn(self.n_samples, self.n_outputs)
        
        # Instantiate the model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate
        ).to(device)

    # --- 1. Test Sequence Creation ---
    def test_01_create_sequences_shapes(self):
        """Test if create_sequences produces correct output shapes."""
        print("Running Test: create_sequences output shapes")
        X_seq, y_seq = create_sequences(self.X_data_np, self.y_data_np, self.seq_length)
        
        expected_num_sequences = self.n_samples - self.seq_length
        self.assertEqual(X_seq.shape, (expected_num_sequences, self.seq_length, self.n_features))
        self.assertEqual(y_seq.shape, (expected_num_sequences, self.n_outputs))

    def test_02_create_sequences_values(self):
        """Test if create_sequences produces correct values."""
        print("Running Test: create_sequences output values")
        # Use a predictable dataset to check the logic
        X_simple = np.array([[i] for i in range(10)]) # Features are 0, 1, 2,...
        y_simple = np.array([[i * 10] for i in range(10)]) # Targets are 0, 10, 20,...
        seq_len = 3

        X_seq, y_seq = create_sequences(X_simple, y_simple, seq_len)
        
        # Check first sequence and target
        np.testing.assert_array_equal(X_seq[0], np.array([[0], [1], [2]]))
        np.testing.assert_array_equal(y_seq[0], np.array([20])) # Target is y at the end of the first sequence window (index 2)
        
        # Check last sequence and target
        np.testing.assert_array_equal(X_seq[-1], np.array([[6], [7], [8]]))
        np.testing.assert_array_equal(y_seq[-1], np.array([80])) # Target is y at index 8

    def test_03_create_sequences_edge_case(self):
        """Test create_sequences when input is too short."""
        print("Running Test: create_sequences edge case (short input)")
        X_short = np.random.randn(4, self.n_features)
        y_short = np.random.randn(4, self.n_outputs)
        X_seq, y_seq = create_sequences(X_short, y_short, self.seq_length) # seq_length is 5
        
        self.assertEqual(X_seq.shape[0], 0, "Should produce no sequences if data length < seq_length")
        self.assertEqual(y_seq.shape[0], 0, "Should produce no targets if data length < seq_length")

    # --- 2. Test Model Architecture ---
    def test_04_model_initialization(self):
        """Test if the LSTMModel initializes correctly."""
        print("Running Test: Model Initialization")
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model.lstm, nn.LSTM)
        self.assertIsInstance(self.model.linear, nn.Linear)
        self.assertEqual(self.model.lstm.input_size, self.input_size)
        self.assertEqual(self.model.linear.out_features, self.output_size)
        
    def test_05_model_forward_pass_shape(self):
        """Test the forward pass for correct output shape."""
        print("Running Test: Model Forward Pass Shape")
        # Create a dummy batch of sequences
        input_tensor = torch.randn(self.batch_size, self.seq_length, self.input_size).to(device)
        self.model.eval() # Set to eval mode for consistency
        with torch.no_grad():
            output = self.model(input_tensor)
        
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        self.assertEqual(str(output.device), str(device))

    # --- 3. Test Training & Prediction Functions ---
    def test_06_train_model_runs(self):
        """Test if the training function runs without errors."""
        print("Running Test: Training Function Execution")
        X_seq, y_seq = create_sequences(self.X_data_np, self.y_data_np, self.seq_length)
        dataset = TensorDataset(torch.from_numpy(X_seq).float(), torch.from_numpy(y_seq).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        try:
            train_lstm_model(self.model, loader, learning_rate=0.001, epochs=2)
        except Exception as e:
            self.fail(f"train_lstm_model raised an exception: {e}")

    def test_07_train_model_updates_weights(self):
        """Test if the training function actually updates model weights."""
        print("Running Test: Training Function Updates Weights")
        X_seq, y_seq = create_sequences(self.X_data_np, self.y_data_np, self.seq_length)
        dataset = TensorDataset(torch.from_numpy(X_seq).float(), torch.from_numpy(y_seq).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Clone initial weights of the linear layer
        initial_weights = self.model.linear.weight.clone().detach()
        
        train_lstm_model(self.model, loader, learning_rate=0.01, epochs=2)
        
        # Get weights after training
        updated_weights = self.model.linear.weight.clone().detach()
        
        self.assertFalse(torch.equal(initial_weights, updated_weights), "Model weights should change after training.")

    def test_08_predict_step_by_step_shape_and_padding(self):
        """Test step-by-step prediction for correct shape and initial padding."""
        print("Running Test: Step-by-Step Prediction Shape & Padding")
        # Train model slightly to get non-zero predictions
        self.test_07_train_model_updates_weights()

        predictions = predict_step_by_step_lstm(self.model, self.X_data_np, self.y_data_np, self.seq_length)
        
        # Check overall shape
        self.assertEqual(predictions.shape, (self.n_samples, self.n_outputs))
        
        # Check for zero-padding at the beginning
        expected_padding = np.zeros((self.seq_length, self.n_outputs))
        np.testing.assert_array_equal(predictions[:self.seq_length], expected_padding, "Initial predictions should be padded with zeros.")
        
        # Check that subsequent predictions are not all zero
        self.assertTrue(np.any(predictions[self.seq_length:] != 0), "Predictions after padding should not be all zero.")

    def test_09_predict_full_trajectory_shape(self):
        """Test recursive full trajectory prediction for correct shape."""
        print("Running Test: Full Trajectory Prediction Shape")
        # This function has hardcoded dependencies (e.g., n_outputs=14, input structure)
        # We need to adjust our test setup to match it.
        n_features_recursive = 21 # 7 pos, 7 vel, 7 torques
        n_outputs_recursive = 14
        X_data_recursive_t = torch.randn(self.n_samples, n_features_recursive, device=device)

        # Re-initialize model to match the function's expectations
        recursive_model = LSTMModel(
            input_size=n_features_recursive,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=n_outputs_recursive,
            dropout_rate=self.dropout_rate
        ).to(device)
        
        predictions = predict_full_trajectory_lstm(recursive_model, X_data_recursive_t, self.seq_length)
        
        self.assertEqual(predictions.shape, (self.n_samples, n_outputs_recursive))
    
    def test_10_prediction_does_not_alter_model(self):
        """Test that prediction functions do not alter model weights."""
        print("Running Test: Prediction Does Not Alter Model")
        initial_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Run step-by-step prediction
        _ = predict_step_by_step_lstm(self.model, self.X_data_np, self.y_data_np, self.seq_length)
        
        state_dict_after_sbs = self.model.state_dict()
        for key in initial_state_dict:
            self.assertTrue(torch.equal(initial_state_dict[key], state_dict_after_sbs[key]),
                            f"Step-by-step prediction altered model parameter: {key}")

        # Run full trajectory prediction (using the setup from the previous test)
        n_features_recursive = 21
        n_outputs_recursive = 14
        X_data_recursive_t = torch.randn(self.n_samples, n_features_recursive, device=device)
        recursive_model = LSTMModel(n_features_recursive, self.hidden_size, self.num_layers, n_outputs_recursive, self.dropout_rate).to(device)
        
        initial_state_dict_recursive = {k: v.clone() for k, v in recursive_model.state_dict().items()}
        _ = predict_full_trajectory_lstm(recursive_model, X_data_recursive_t, self.seq_length)
        state_dict_after_full = recursive_model.state_dict()
        for key in initial_state_dict_recursive:
             self.assertTrue(torch.equal(initial_state_dict_recursive[key], state_dict_after_full[key]),
                            f"Full trajectory prediction altered model parameter: {key}")


if __name__ == '__main__':
    # This allows the test suite to be run from the command line
    unittest.main(argv=['first-arg-is-ignored'], exit=False)