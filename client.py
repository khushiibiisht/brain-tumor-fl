# client.py
# Defines what each hospital client does in every FL round
#
# In real federated learning, this code runs ON the hospital's
# own server — the central server never sees their MRI data.
# We simulate 3 hospitals by running 3 instances of this file.
#
# Every FL round, each client:
#   1. Receives global model weights from server
#   2. Trains on its own local MRI data
#   3. Sends updated weights back to server
#   4. Discards local changes (server will aggregate)

import os
import sys
import torch
import flwr as fl
import numpy as np
from model import get_model
from train_utils import train, test, get_dataloader


# ─────────────────────────────────────────
# THE FL CLIENT CLASS
# Inherits from Flower's NumPyClient
# NumPy = weights are exchanged as numpy arrays
# ─────────────────────────────────────────
class BrainTumorClient(fl.client.NumPyClient):

    def __init__(self, client_id, device):
        """
        client_id — which hospital this is (0, 1, or 2)
        device    — 'cuda' or 'cpu'
        """
        self.client_id = client_id
        self.device = device

        # Each client loads ONLY its own portion of data
        # client_0 never sees client_1's images — ever
        train_dir = f'data/split/client_{client_id}'
        test_dir  = 'data/Testing'

        print(f'\n[Client {client_id}] Loading data from {train_dir}')
        self.train_loader = get_dataloader(
            train_dir, batch_size=16, shuffle=True
        )
        self.test_loader = get_dataloader(
            test_dir, batch_size=16, shuffle=False
        )

        # Each client has its own local copy of ResNet-18
        self.model = get_model(device)
        print(f'[Client {client_id}] Ready!')

    # ── CALLED BY FLOWER ──────────────────
    def get_parameters(self, config):
        """
        Returns current model weights as a list of numpy arrays.
        Flower calls this to get weights to send to the server.
        """
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]

    # ── CALLED BY FLOWER ──────────────────
    def set_parameters(self, parameters):
        """
        Loads weights received from the server into local model.
        This is how the global model is distributed to each client.
        """
        params_dict = zip(
            self.model.state_dict().keys(), parameters
        )
        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }
        self.model.load_state_dict(state_dict, strict=True)

    # ── CALLED BY FLOWER EVERY ROUND ──────
    def fit(self, parameters, config):
        """
        The main training step — called every FL round.

        1. Load global weights from server
        2. Train on local data for 1 epoch
        3. Return updated weights to server

        Returns:
          - updated weights
          - number of training samples (for weighted averaging)
          - metrics dictionary
        """
        print(f'\n[Client {self.client_id}] '
              f'Starting local training...')

        # Step 1: Load global model weights
        self.set_parameters(parameters)

        # Step 2: Train locally for 1 epoch
        loss, acc = train(
            self.model,
            self.train_loader,
            self.device,
            epochs=1
        )

        print(f'[Client {self.client_id}] '
              f'Train Loss: {loss:.4f} | '
              f'Train Acc: {acc*100:.2f}%')

        # Step 3: Return updated weights + dataset size + metrics
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {'train_loss': loss, 'train_acc': acc}
        )

    # ── CALLED BY FLOWER EVERY ROUND ──────
    def evaluate(self, parameters, config):
        """
        Evaluates the global model on local test data.
        Called after aggregation each round.

        Returns:
          - loss (float)
          - number of test samples
          - metrics dictionary
        """
        print(f'\n[Client {self.client_id}] '
              f'Evaluating global model...')

        # Load the aggregated global weights
        self.set_parameters(parameters)

        # Test on the shared test set
        loss, acc = test(
            self.model,
            self.test_loader,
            self.device
        )

        print(f'[Client {self.client_id}] '
              f'Test Loss: {loss:.4f} | '
              f'Test Acc: {acc*100:.2f}%')

        return (
            float(loss),
            len(self.test_loader.dataset),
            {'test_loss': loss, 'test_acc': float(acc)}
        )


# ─────────────────────────────────────────
# ENTRY POINT
# Run as: python client.py 0
#         python client.py 1
#         python client.py 2
# The number is the client/hospital ID
# ─────────────────────────────────────────
if __name__ == '__main__':

    # Get client ID from command line argument
    if len(sys.argv) != 2:
        print('Usage: python client.py <client_id>')
        print('Example: python client.py 0')
        sys.exit(1)

    client_id = int(sys.argv[1])
    if client_id not in [0, 1, 2]:
        print('client_id must be 0, 1, or 2')
        sys.exit(1)

    # Use GPU if available
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Client {client_id} using device: {device}')

    # Create the client
    client = BrainTumorClient(client_id, device)

    # Connect to the Flower server and start FL
    # Server runs on localhost port 8080
    import os
    server_address = os.getenv('SERVER_ADDRESS', 'localhost:8080')
    print(f'Connecting to server at: {server_address}')
    fl.client.start_numpy_client(
    server_address=server_address,
    client=client
    )