import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics, Parameters, FitIns, EvaluateIns, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy


# Define the weighted average strategy for federated learning
class WeightedAverageStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        """Aggregate model weights using weighted average."""
        if not results:
            return None

        # Extract weights and num_examples
        weights_results = [
            (fit_res.parameters, fit_res.num_examples) 
            for _, fit_res in results
        ]
        
        # Use the parent class (FedAvg) method to aggregate
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Evaluate model on server (optional)."""
        # No server-side evaluation in this implementation
        return None

if __name__ == "__main__":
    # Define strategy
    strategy = WeightedAverageStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Minimum number of clients to be sampled for training
        min_evaluate_clients=2,  # Minimum number of clients to be sampled for evaluation
        min_available_clients=2,  # Minimum number of clients that need to be connected to the server
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

