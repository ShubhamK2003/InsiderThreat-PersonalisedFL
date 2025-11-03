# server.py
"""
Server orchestrates federated training: collects encoder updates and averages them (weighted) on CPU,
then broadcasts averaged encoder back to clients.
"""
import copy
from fed_utils import safe_cpu_average_state_dicts # Updated utility
import torch

class Server:
    def __init__(self, initial_encoder_state_dict):
        self.global_encoder = {k: v.cpu().float().clone() for k, v in initial_encoder_state_dict.items()}

    def aggregate_encoder_updates(self, encoder_updates_with_samples):
        """
        encoder_updates_with_samples: list of (encoder.state_dict(), num_samples) from clients
        Aggregate them (weighted) safely on CPU and set as new global encoder.
        """
        avg = safe_cpu_average_state_dicts(encoder_updates_with_samples)
        self.global_encoder = avg
        return avg

    def get_global_encoder(self):
        return {k: v.clone() for k, v in self.global_encoder.items()}