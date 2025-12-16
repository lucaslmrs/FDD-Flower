"""Custom FedAvg strategy for Personalized Federated Learning with fine-tuning."""

import json
import os
from datetime import datetime

import torch
import wandb
from flwr.common import FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .task import Net, set_weights


class CustomFedAvg(FedAvg):
    """A strategy that extends FedAvg with:
    - Checkpointing of global models
    - Metrics logging to W&B and JSON
    - Personalized FL: Skip aggregation during fine-tuning rounds
    """

    def __init__(self, *args, run_dir: str = "artifacts", 
                 num_federated_rounds: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.run_dir = run_dir
        self.num_federated_rounds = num_federated_rounds
        self.results_to_save = {
            "federated_rounds": {},
            "fine_tuning_results": {}
        }
        self.last_aggregated_params = None

        # Initialize W&B
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(
            project="pick-and-place-fault-detection-pfl",
            name=f"personalized-fl-{name}",
            config={
                "model": "Neural Network",
                "num_classes": 2,
                "num_features": 11,
                "task": "Pick-and-Place Fault Detection",
                "personalized": True,
            }
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate model updates - skip aggregation during fine-tuning."""
        
        # Check if we're in fine-tuning phase
        is_fine_tuning = server_round > self.num_federated_rounds
        
        if is_fine_tuning:
            fine_tuning_round = server_round - self.num_federated_rounds
            print(f"\n[Fine-tuning Round {fine_tuning_round}] "
                  f"Skipping aggregation - clients training locally")
            
            # Collect per-client fine-tuning metrics
            metrics_aggregated = {}
            client_metrics = {}
            
            if results:
                train_losses = []
                val_losses = []
                val_accuracies = []
                
                for client_proxy, fit_res in results:
                    train_loss = fit_res.metrics.get("train_loss", 0.0)
                    val_loss = fit_res.metrics.get("val_loss", 0.0)
                    val_accuracy = fit_res.metrics.get("val_accuracy", 0.0)
                    partition_id = fit_res.metrics.get("partition_id", -1)
                    
                    # Store per-client metrics
                    client_key = f"client_{partition_id}"
                    client_metrics[client_key] = {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    }
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                
                # Calculate aggregated metrics
                metrics_aggregated["avg_fine_tune_loss"] = sum(train_losses) / len(train_losses)
                metrics_aggregated["avg_val_loss"] = sum(val_losses) / len(val_losses)
                metrics_aggregated["avg_val_accuracy"] = sum(val_accuracies) / len(val_accuracies)
                
                # Store in results
                round_key = f"round_{fine_tuning_round}"
                self.results_to_save["fine_tuning_results"][round_key] = {
                    "clients": client_metrics,
                    "aggregated": metrics_aggregated,
                }
                
                # Save to JSON files
                self._save_results()
                
                # Log to W&B
                wandb_metrics = {
                    f"fine_tuning/{k}": v for k, v in metrics_aggregated.items()
                }
                wandb_metrics["fine_tuning/round"] = fine_tuning_round
                
                # Also log individual client metrics
                for client_key, c_metrics in client_metrics.items():
                    for metric_name, metric_value in c_metrics.items():
                        wandb_metrics[f"fine_tuning/{client_key}/{metric_name}"] = metric_value
                
                wandb.log(wandb_metrics, step=server_round)
                
            return self.last_aggregated_params, metrics_aggregated
        
        # Regular federated aggregation
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Store last aggregated parameters for fine-tuning phase
        self.last_aggregated_params = parameters_aggregated
        
        # Save global model checkpoint
        if parameters_aggregated is not None:
            ndarrays = parameters_to_ndarrays(parameters_aggregated)
            model = Net()
            set_weights(model, ndarrays)
            
            # Save to run directory
            os.makedirs(self.run_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.run_dir, f"global_model_round_{server_round}")
            torch.save(model.state_dict(), checkpoint_path)

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model - skip during fine-tuning (personalized models)."""
        
        is_fine_tuning = server_round > self.num_federated_rounds
        
        # Skip central evaluation during fine-tuning (each client has personalized model)
        if is_fine_tuning:
            fine_tuning_round = server_round - self.num_federated_rounds
            print(f"[Fine-tuning Round {fine_tuning_round}] "
                  f"Skipping central evaluation - models are personalized")
            return None
        
        # Regular evaluation
        loss, metrics = super().evaluate(server_round, parameters)

        # Store and log metrics
        my_results = {"loss": loss, **metrics, "round": server_round}
        self.results_to_save["federated_rounds"][server_round] = my_results

        # Save to JSON files
        self._save_results()

        # Log metrics to W&B
        wandb_metrics = {f"federated/{k}": v for k, v in my_results.items()}
        wandb.log(wandb_metrics, step=server_round)

        return loss, metrics
    
    def _save_results(self):
        """Save results to JSON files."""
        # Save to run directory
        results_path = os.path.join(self.run_dir, "results.json")
        with open(results_path, "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)
        
        # Also save to main folder for backwards compatibility
        with open("results.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)