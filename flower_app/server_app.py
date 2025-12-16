"""Flower Server App for Pick-and-Place Fault Detection using Federated Learning."""

import json
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from flower_app.my_strategy import CustomFedAvg
from flower_app.task import Net, get_weights, load_test_data, set_weights, test, get_next_run_dir


def get_evaluate_fn(testloader, device, num_features, num_classes):
    """Return a callback that evaluates the global model."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""
        # Instantiate model
        net = Net(num_features=num_features, num_classes=num_classes)
        # Apply global_model parameters
        set_weights(net, parameters_ndarrays)
        net.to(device)
        # Run test
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # Loop trough all metrics received compute accuracies x examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Return weighted average accuracy
    return {"accuracy": sum(accuracies) / total_examples}


def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from a fit round.

    This function showed a mechanism to communicate almost arbitrary metrics by
    converting them into a JSON on the ClientApp side. Here that JSON string is
    deserialized and reinterpreted as a dictionary we can use.
    """
    b_values = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]
        # Deserialize JSON and return dict
        my_metric = json.loads(my_metric_str)
        b_values.append(my_metric["b"])
    # Return maximum value from deserialized metrics
    return {"max_b": max(b_values)}


def on_fit_config(server_round: int) -> Metrics:
    """Adjusts learning rate based on current round."""
    lr = 0.01
    # Appply a simple learning rate decay
    if server_round > 2:
        lr = 0.005
    return {"lr": lr, "fine_tuning": False}


def create_on_fit_config(num_federated_rounds: int, fine_tuning_rounds: int, 
                          fine_tuning_lr: float, run_dir: str):
    """Create the on_fit_config function with fine-tuning awareness."""
    
    def on_fit_config_fn(server_round: int) -> Metrics:
        """Adjusts learning rate and signals fine-tuning phase."""
        
        # Check if we're in fine-tuning phase
        is_fine_tuning = server_round > num_federated_rounds
        
        if is_fine_tuning:
            # Fine-tuning configuration
            fine_tuning_round = server_round - num_federated_rounds
            print(f"\n{'='*60}")
            print(f"FINE-TUNING ROUND {fine_tuning_round}/{fine_tuning_rounds}")
            print(f"{'='*60}")
            return {
                "lr": fine_tuning_lr,
                "fine_tuning": True,
                "fine_tuning_lr": fine_tuning_lr,
                "fine_tuning_epochs": 5,
                "run_dir": run_dir,
            }
        else:
            # Regular federated training with LR decay
            lr = 0.01 if server_round <= 2 else 0.005
            return {
                "lr": lr,
                "fine_tuning": False,
            }
    
    return on_fit_config_fn


def server_fn(context: Context):
    """A function that creates the components for a ServerApp."""
    # Read from Run config
    num_federated_rounds = context.run_config["num-server-rounds"]
    fine_tuning_rounds = context.run_config["fine-tuning-rounds"]
    fine_tuning_lr = context.run_config["fine-tuning-lr"]
    fraction_fit = context.run_config["fraction-fit"]
    num_classes = context.run_config["num-classes"]
    num_features = context.run_config["num-features"]
    
    # Total rounds = federated + fine-tuning
    total_rounds = num_federated_rounds + fine_tuning_rounds
    
    # Create run directory for this execution
    run_dir = get_next_run_dir("artifacts")
    print(f"\n{'='*60}")
    print(f"PERSONALIZED FEDERATED LEARNING")
    print(f"Federated rounds: {num_federated_rounds}")
    print(f"Fine-tuning rounds: {fine_tuning_rounds}")
    print(f"Total rounds: {total_rounds}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*60}\n")

    # Initialize model parameters
    ndarrays = get_weights(Net(num_features=num_features, num_classes=num_classes))
    parameters = ndarrays_to_parameters(ndarrays)

    # Load global test set (pick-and-place fault detection data)
    testloader = load_test_data()

    # Create on_fit_config with fine-tuning awareness
    on_fit_config_fn = create_on_fit_config(
        num_federated_rounds, fine_tuning_rounds, fine_tuning_lr, run_dir
    )

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,  # All nodes are sampled for evaluation
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu", num_features=num_features, num_classes=num_classes),
        # Pass extra info for the strategy
        run_dir=run_dir,
        num_federated_rounds=num_federated_rounds,
    )
    config = ServerConfig(num_rounds=total_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)