"""Flower Client App for Pick-and-Place Fault Detection using Federated Learning."""

import json
from random import random

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ConfigRecord, Context

from flower_app.task import (
    Net, get_weights, load_data, set_weights, test, train,
    freeze_first_half, fine_tune, save_personalized_model
)


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context, partition_id: int):
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        """Train a model using as starting point the parameters sent by the ServerApp.

        Then, communicate the weights of the locally-updated model back to the
        ServerApp.
        """
        # Check if this is a fine-tuning round
        is_fine_tuning = config.get("fine_tuning", False)
        
        # Apply parameters to local model
        set_weights(self.net, parameters)
        
        if is_fine_tuning:
            # Fine-tuning phase: freeze first half, train second half locally
            print(f"\n=== Fine-tuning phase for client {self.partition_id} ===")
            freeze_first_half(self.net)
            
            fine_tuning_lr = config.get("fine_tuning_lr", 0.001)
            fine_tuning_epochs = config.get("fine_tuning_epochs", 5)
            
            train_loss = fine_tune(
                self.net,
                self.trainloader,
                fine_tuning_epochs,
                fine_tuning_lr,
                self.device,
            )
            
            # Save personalized model
            run_dir = config.get("run_dir", "artifacts/run_000")
            save_personalized_model(self.net, self.partition_id, run_dir)
            
            # Evaluate personalized model
            val_loss, val_accuracy = test(self.net, self.valloader, self.device)
            print(f"Client {self.partition_id} personalized - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
        else:
            # Regular federated training
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                config["lr"],
                self.device,
            )
            val_loss, val_accuracy = 0.0, 0.0

        # Append to persistent state the `train_loss` just obtained
        fit_metrics = self.client_state.config_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            # If first entry, create the list
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            # If it's not the first entry, append to the existing list
            fit_metrics["train_loss_hist"].append(train_loss)

        # Complex metric for demonstration
        complex_metric = {"a": 123, "b": random(), "mylist": [1, 2, 3, 4]}
        complex_metric_str = json.dumps(complex_metric)

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "partition_id": self.partition_id,
                "my_metric": complex_metric_str,
                "fine_tuning": is_fine_tuning,
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the global model weights using the local validation set."""
        # Apply weights from global model
        set_weights(self.net, parameters)
        # Run the test evaluation function
        loss, accuracy = test(self.net, self.valloader, self.device)
        # Report results. Note the last argument is of type `Metrics` so you could communicate
        # other values that are relevant to your use case.
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """A function that returns a Client."""

    # Read the run config (defined in the `pyproject.toml`)
    local_epochs = context.run_config["local-epochs"]
    num_classes = context.run_config["num-classes"]
    num_features = context.run_config["num-features"]

    # Instantiate the model with config parameters
    net = Net(num_features=num_features, num_classes=num_classes)
    
    # Read node config and fetch data for the ClientApp that is being constructed
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Return Client instance with partition_id for personalization
    return FlowerClient(net, trainloader, valloader, local_epochs, context, partition_id).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)