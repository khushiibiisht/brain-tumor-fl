# server.py
# The central FL server — coordinates all hospital clients
#
# What it does every round:
#   1. Sends current global model weights → all clients
#   2. Waits for all clients to finish local training
#   3. Aggregates (averages) all weight updates — FedAvg
#   4. Updates global model with averaged weights
#   5. Logs accuracy and loss for your report
#   Repeat for NUM_ROUNDS rounds

import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import json
import os


# ─────────────────────────────────────────
# CONFIGURATION
# Change these to run different experiments
# ─────────────────────────────────────────
NUM_ROUNDS   = 10   # how many FL rounds to run
MIN_CLIENTS  = 3    # minimum clients needed to start
NUM_CLIENTS  = 3    # total number of hospital clients


# ─────────────────────────────────────────
# METRICS AGGREGATION
# Flower calls these functions to combine
# metrics reported by all clients each round
# ─────────────────────────────────────────
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Averages metrics from all clients weighted by dataset size.
    A client with 1868 images contributes more than one with 100.

    metrics — list of (num_samples, metrics_dict) from each client
    """
    # Extract accuracies weighted by number of samples
    accuracies = [
        num_samples * m.get('test_acc', 0)
        for num_samples, m in metrics
    ]
    losses = [
        num_samples * m.get('test_loss', 0)
        for num_samples, m in metrics
    ]
    total_samples = sum(
        num_samples for num_samples, _ in metrics
    )

    avg_acc  = sum(accuracies) / total_samples
    avg_loss = sum(losses) / total_samples

    print(f'\n{"="*50}')
    print(f'GLOBAL MODEL METRICS THIS ROUND:')
    print(f'  Accuracy : {avg_acc*100:.2f}%')
    print(f'  Loss     : {avg_loss:.4f}')
    print(f'{"="*50}\n')

    return {
        'test_acc' : avg_acc,
        'test_loss': avg_loss
    }


# ─────────────────────────────────────────
# RESULTS LOGGER
# Saves accuracy/loss to a JSON file
# so you can plot graphs for your report
# ─────────────────────────────────────────
class ResultsLogger:
    def __init__(self, log_path='logs/results.json'):
        self.log_path = log_path
        self.results  = {
            'rounds'    : [],
            'accuracy'  : [],
            'loss'      : [],
            'strategy'  : 'FedAvg',
            'model'     : 'ResNet-18',
            'num_clients': NUM_CLIENTS
        }
        os.makedirs('logs', exist_ok=True)

    def log(self, round_num, acc, loss):
        self.results['rounds'].append(round_num)
        self.results['accuracy'].append(round(acc * 100, 2))
        self.results['loss'].append(round(loss, 4))

        # Save after every round so data isn't lost
        # if training is interrupted
        with open(self.log_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f'[Logger] Round {round_num} results saved.')


# ─────────────────────────────────────────
# CUSTOM STRATEGY
# Extends FedAvg to add logging
# ─────────────────────────────────────────
class FedAvgWithLogging(fl.server.strategy.FedAvg):
    """
    FedAvg strategy with result logging added.
    Everything else is standard FedAvg.
    """

    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self.round  = 0

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures
    ):
        """Called after evaluation each round — we log here."""

        # Call parent FedAvg aggregation first
        aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Log the results
        if aggregated is not None:
            loss, metrics = aggregated
            acc  = metrics.get('test_acc',  0)
            loss_val = metrics.get('test_loss', 0)
            self.logger.log(server_round, acc, loss_val)

        return aggregated


# ─────────────────────────────────────────
# MAIN — starts the FL server
# ─────────────────────────────────────────
if __name__ == '__main__':

    print('='*50)
    print('  BRAIN TUMOR FL SERVER')
    print(f'  Strategy  : FedAvg')
    print(f'  Model     : ResNet-18')
    print(f'  Rounds    : {NUM_ROUNDS}')
    print(f'  Clients   : {NUM_CLIENTS}')
    print('='*50)
    print(f'\nWaiting for {MIN_CLIENTS} clients to connect...')
    print('Start clients in separate terminals:')
    print('  python client.py 0')
    print('  python client.py 1')
    print('  python client.py 2\n')

    # Create logger
    logger = ResultsLogger('logs/results.json')

    # Create FedAvg strategy with logging
    strategy = FedAvgWithLogging(
        logger=logger,

        # Fraction of clients used for training each round
        # 1.0 = use ALL clients every round
        fraction_fit=1.0,

        # Fraction of clients used for evaluation
        fraction_evaluate=1.0,

        # Minimum clients needed before training starts
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,

        # How to aggregate evaluation metrics from all clients
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start the Flower server
    # It will listen on port 8080 for client connections
    fl.server.start_server(
        server_address='0.0.0.0:8080',
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy=strategy,
    )

    print('\nTraining complete!')
    print('Results saved to logs/results.json')
    print('Run plot_results.py to visualise accuracy curves.')