import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Define the target device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            dropout=dropout_rate
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # We only use the output from the last time step
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step_out)
        return predictions
    
# --- 4. Training and Prediction Functions ---
def train_lstm_model(model, data_loader, learning_rate, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
    scaler = GradScaler()
    model.train()
    for epoch in range(epochs):
        for seq, labels in data_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                y_pred = model(seq)
                loss = criterion(y_pred, labels)
                scheduler.step(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model

def predict_step_by_step_lstm(model, X_test_np, y_test_np, seq_length):
    """
    Predicts outputs one step ahead, using ground truth for each input.
    """
    model.eval()
    n_samples = X_test_np.shape[0]
    n_outputs = y_test_np.shape[1]
    
    # Create input sequences from the ground truth test data
    X_seq_np, _ = create_sequences(X_test_np, y_test_np, seq_length)
    X_seq_t = torch.from_numpy(X_seq_np).float().to(device)
    
    predictions_seq = np.zeros((X_seq_np.shape[0], n_outputs))

    with torch.no_grad():
        predictions_seq = model(X_seq_t).cpu().numpy()
            
    # Pad the start of the predictions to match the original trajectory length
    padded_predictions = np.zeros((n_samples, n_outputs))
    padded_predictions[seq_length:] = predictions_seq # Shift by seq_length
    
    return padded_predictions

def predict_full_trajectory_lstm(model, X_data, seq_length):
    """
    Predicts a full trajectory recursively (open-loop).
    Uses its own predictions as input for subsequent steps.
    """
    model.eval()
    n_samples = X_data.shape[0]
    n_outputs = 14 # 7 positions, 7 velocities
    predictions = torch.zeros(n_samples, n_outputs).to(device)

    # Initial sequence from ground truth
    current_sequence = X_data[0:seq_length, :].reshape(1, seq_length, -1)

    with torch.no_grad():
        for t in range(n_samples - seq_length + 1):
            # Predict the next state
            pred_next_state = model(current_sequence)
            predictions[t + seq_length -1, :] = pred_next_state

            # Prepare the next sequence
            if t < n_samples - seq_length:
                # Get the next true action (torques)
                next_torques = X_data[t + seq_length, 14:]

                # Create the next feature vector using the predicted state and true action
                next_feature_vector = torch.hstack((pred_next_state.flatten(), next_torques))

                # Update the sequence: drop the oldest, append the new
                next_sequence_features = torch.vstack((current_sequence.squeeze(0)[1:], next_feature_vector))
                current_sequence = next_sequence_features.reshape(1, seq_length, -1)

    return predictions.cpu().numpy()