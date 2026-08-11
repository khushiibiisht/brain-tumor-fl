# server.py
import flwr as fl
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics
import json
import os

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
NUM_ROUNDS  = 10
MIN_CLIENTS = 3
NUM_CLIENTS = 3


# ─────────────────────────────────────────
# METRICS AGGREGATION
# ─────────────────────────────────────────
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [
        num_samples * m.get('test_acc', 0)
        for num_samples, m in metrics
    ]
    losses = [
        num_samples * m.get('test_loss', 0)
        for num_samples, m in metrics
    ]
    total_samples = sum(num_samples for num_samples, _ in metrics)

    avg_acc  = sum(accuracies) / total_samples
    avg_loss = sum(losses) / total_samples

    print(f'\n{"="*50}')
    print(f'GLOBAL MODEL METRICS THIS ROUND:')
    print(f'  Accuracy : {avg_acc*100:.2f}%')
    print(f'  Loss     : {avg_loss:.4f}')
    print(f'{"="*50}\n')

    return {'test_acc': avg_acc, 'test_loss': avg_loss}


# ─────────────────────────────────────────
# RESULTS LOGGER
# ─────────────────────────────────────────
class ResultsLogger:
    def __init__(self, log_path='logs/results.json'):
        self.log_path = log_path
        self.results  = {
            'rounds'     : [],
            'accuracy'   : [],
            'loss'       : [],
            'strategy'   : 'FedAvg',
            'model'      : 'ResNet-18',
            'num_clients': NUM_CLIENTS
        }
        os.makedirs('logs', exist_ok=True)

    def log(self, round_num, acc, loss):
        self.results['rounds'].append(round_num)
        self.results['accuracy'].append(round(acc * 100, 2))
        self.results['loss'].append(round(loss, 4))

        with open(self.log_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f'[Logger] Round {round_num} results saved.')


# ─────────────────────────────────────────
# CUSTOM STRATEGY — FedAvg + logging + weight saving
# ─────────────────────────────────────────
class FedAvgWithLogging(fl.server.strategy.FedAvg):

    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def aggregate_fit(self, server_round, results, failures):
        """
        Called after all clients finish local training.
        Aggregates weights using FedAvg and saves them to disk.
        """
        # Run standard FedAvg aggregation
        aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated is not None:
            parameters, metrics = aggregated

            # Convert Flower parameters → numpy arrays
            import flwr.common
            ndarrays = flwr.common.parameters_to_ndarrays(parameters)

            # Save numpy arrays to disk
            # allow_pickle=True needed for arrays of different shapes
            os.makedirs('logs', exist_ok=True)
            np.save(
                'logs/global_weights.npy',
                np.array(ndarrays, dtype=object),
                allow_pickle=True
            )
            print(f'[Server] Round {server_round} weights saved '
                  f'→ logs/global_weights.npy')

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Called after all clients finish evaluation.
        Aggregates metrics and logs them.
        """
        # Run standard FedAvg evaluation aggregation
        aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Log accuracy and loss for this round
        if aggregated is not None:
            loss, metrics = aggregated
            acc      = metrics.get('test_acc',  0)
            loss_val = metrics.get('test_loss', 0)
            self.logger.log(server_round, acc, loss_val)

        return aggregated


# ─────────────────────────────────────────
# MAIN
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

    logger = ResultsLogger('logs/results.json')

    strategy = FedAvgWithLogging(
        logger=logger,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address='0.0.0.0:8080',
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print('\nTraining complete!')
    print('Results saved to logs/results.json')
    print('Weights saved to logs/global_weights.npy')
    print('Run: python app.py to start the Flask UI')