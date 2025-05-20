import flwr as fl
import torch
import torch.nn.functional as F
import numpy as np
from model_lstm import LSTMClassifier

def load_data(num_samples=100):
    X = np.random.randn(num_samples, 10, 1).astype(np.float32)
    y = (X.mean(axis=(1,2)) > 0).astype(np.int64)
    return X, y

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = LSTMClassifier(input_dim=1, hidden_dim=32, output_dim=2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.X_train, self.y_train = load_data()
        self.X_val, self.y_val     = load_data(num_samples=30)

    # *** Updated signature ***
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (k, _), arr in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(arr)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        X = torch.tensor(self.X_train)
        y = torch.tensor(self.y_train)
        self.optimizer.zero_grad()
        preds = self.model(X)
        loss = self.criterion(preds, y)
        loss.backward()
        self.optimizer.step()
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        X = torch.tensor(self.X_val)
        y = torch.tensor(self.y_val)
        with torch.no_grad():
            preds = self.model(X)
            loss = self.criterion(preds, y).item()
            accuracy = (preds.argmax(dim=1) == y).float().mean().item()
        return loss, len(self.X_val), {"accuracy": accuracy}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, required=True)
    args = parser.parse_args()
    fl.client.start_client(server_address=args.server_address, client=FlowerClient())
