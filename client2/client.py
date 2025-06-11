import flwr as fl
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model_lstm import LSTMClassifier
import os

# Data preprocessing functions
def load_tep_data(data_path, client_id=2, num_clients=2):
    # Load the TEP dataset
    df = pd.read_csv(data_path)
    
    # Split data based on client ID for federated learning simulation
    # Each client gets a different portion of the data
    df_fault_free = df[df['faultNumber'] == 0]
    df_faulty = df[df['faultNumber'] != 0]
    
    # Partition data for this client
    fault_free_size = len(df_fault_free) // num_clients
    faulty_size = len(df_faulty) // num_clients
    
    start_idx_fault_free = (client_id - 1) * fault_free_size
    end_idx_fault_free = client_id * fault_free_size if client_id < num_clients else len(df_fault_free)
    
    start_idx_faulty = (client_id - 1) * faulty_size
    end_idx_faulty = client_id * faulty_size if client_id < num_clients else len(df_faulty)
    
    client_fault_free = df_fault_free.iloc[start_idx_fault_free:end_idx_fault_free]
    client_faulty = df_faulty.iloc[start_idx_faulty:end_idx_faulty]
    
    # Combine data for this client
    client_data = pd.concat([client_fault_free, client_faulty])
    
    return client_data

def preprocess_data(df):
    # Extract features and labels - Fix the column selection to get exactly 52 features
    # Check the column count first
    print(f"DataFrame has {len(df.columns)} columns")
    
    # Identify metadata columns and the label column
    metadata_cols = ['simulationRun', 'measurementNumber', 'faultNumber']
    
    # Select all columns except metadata
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Ensure we have exactly 52 feature columns
    print(f"Selected {len(feature_cols)} feature columns")
    if len(feature_cols) != 52:
        print("WARNING: Expected 52 feature columns but found", len(feature_cols))
        # If needed, adjust to get 52 columns
        if len(feature_cols) > 52:
            feature_cols = feature_cols[:52]  # Take first 52
        else:
            # If less than 52, there might be a deeper issue with the data format
            print("Not enough feature columns in the data!")
    
    # Extract features using the column names
    X = df[feature_cols]
    y = (df['faultNumber'] > 0).astype(int)  # Binary classification: 0 for fault-free, 1 for faulty
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply sliding window
    window_size = 10
    stride = 5
    X_windowed, y_windowed = sliding_window(X_scaled, y, window_size, stride)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_windowed, y_windowed, test_size=0.2, random_state=42
    )
    
    return X_train, y_train, X_val, y_val, scaler

def sliding_window(data, labels, window_size, stride):
    X = []
    y = []
    
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i + window_size])
        # Use the label of the last point in the window
        y.append(labels.iloc[i + window_size - 1])
    
    return np.array(X), np.array(y)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LSTMClassifier(input_dim=52, hidden_dim=64, output_dim=2).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Load and preprocess data - Now looking in the same folder
        data_path = os.path.join("TEP_test.csv")
        # Check if file exists before loading
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found at {data_path}")
            # Fall back to another location or create synthetic data
            if os.path.exists("TEP_test.csv"):
                print("Found alternative file: TEP_test.csv")
                data_path = "TEP_test.csv"
            else:
                raise FileNotFoundError(f"Cannot find TEP data file")

        print(f"Loading data from: {data_path}")
        client_data = load_tep_data(data_path, client_id=2, num_clients=2)
        self.X_train, self.y_train, self.X_val, self.y_val, self.scaler = preprocess_data(client_data)
        
        print(f"Client 1 loaded {len(self.X_train)} training samples and {len(self.X_val)} validation samples")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        state_dict_new = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict_new, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        # Convert to PyTorch tensors
        X = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_train, dtype=torch.long).to(self.device)
        
        # Training loop
        batch_size = 32
        epochs = 5
        n_batches = len(X) // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(X))
            running_loss = 0.0
            
            for i in range(n_batches):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/n_batches:.4f}")
        
        return self.get_parameters(config), len(self.X_train), {"loss": running_loss/n_batches}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        X = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_val, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y).item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / len(y)
        
        return loss, len(self.X_val), {"accuracy": accuracy}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, required=True)
    args = parser.parse_args()
    fl.client.start_client(server_address=args.server_address, client=FlowerClient())
