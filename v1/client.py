# client.py
"""
Client class: holds private decoder, local data, and local optimizer.
This is a FedPer (Federated Personalization) implementation:
The full model is trained locally, but only the encoder is sent to the server.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import config
from fed_utils import reset_optimizer_state_for_encoder

class Client:
    def __init__(self, user_id, client_obj, device=config.DEVICE):
        """
        client_obj: dict with keys 'train_df', 'test_df', 'scaler'
        """
        self.user_id = user_id
        self.train_df = client_obj['train_df']
        self.test_df = client_obj['test_df']
        self.scaler = client_obj['scaler']
        self.device = device

        from models import Encoder, Decoder, LocalAutoencoder
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.model = LocalAutoencoder(encoder=self.encoder, decoder=self.decoder).to(self.device)
        
        self.criterion = nn.MSELoss(reduction='none') 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        
        # prepare data tensors
        X_train = self.train_df[config.NUMERICAL_FEATURES].astype(float).values
        y_train = self.train_df['is_anomaly'].astype(int).values if 'is_anomaly' in self.train_df.columns else np.zeros(len(X_train), dtype=int)
        
        X_test = self.test_df[config.NUMERICAL_FEATURES].astype(float).values
        y_test = self.test_df['is_anomaly'].astype(int).values if 'is_anomaly' in self.test_df.columns else np.zeros(len(X_test), dtype=int)
        
        X_train_s = self.scaler.transform(X_train).astype(np.float32)
        X_test_s = self.scaler.transform(X_test).astype(np.float32)
        
        self.train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train)), batch_size=config.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(y_test)), batch_size=config.BATCH_SIZE, shuffle=False)

    def set_encoder_weights(self, encoder_state_dict):
        """
        Replace local encoder weights with server-provided global encoder.
        Then reset optimizer state for encoder parameters to avoid stale moments.
        """
        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        reset_optimizer_state_for_encoder(self.optimizer, self.model, encoder_prefix='encoder')

    def local_train(self, local_epochs=config.LOCAL_EPOCHS): # *** REMOVED encoder_update_steps ***
        """
        Local training (FedPer style): clients update both encoder and decoder.
        Returns:
            - num_samples_trained (int): Total samples seen during training.
            - avg_loss (float): Average loss across all batches.
        """
        self.model.train()
        num_samples_trained = 0
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            for X_batch, _ in self.train_loader:
                X_batch = X_batch.to(self.device).float()
                
                num_samples_trained += len(X_batch)
                num_batches += 1
                
                # Standard single optimizer over both encoder+decoder
                self.optimizer.zero_grad()
                X_hat = self.model(X_batch)
                
                loss_per_sample = self.criterion(X_hat, X_batch).mean(dim=1)
                loss = loss_per_sample.mean() 
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return num_samples_trained, avg_loss

    def get_encoder_state_dict(self):
        """
        Returns encoder.state_dict() (moved to CPU) to send to server.
        """
        sd = {k: v.cpu().clone().detach() for k, v in self.encoder.state_dict().items()}
        return sd

    def set_encoder_state_dict_from_server(self, sd):
        """
        Convenience to load state dict and reset optimizer for encoder params.
        """
        sd_cpu = {k: v.cpu().float() for k, v in sd.items()}
        self.set_encoder_weights(sd_cpu)

    def eval_reconstruction_errors(self, on='test'):
        """
        Returns per-sample reconstruction errors and labels for 'train' or 'test'.
        """
        self.model.eval()
        loader = self.test_loader if on == 'test' else self.train_loader
        errors = []
        labels = []
        
        with torch.no_grad():
            for X_batch, y in loader:
                X_batch = X_batch.to(self.device).float()
                X_hat = self.model(X_batch)
                per_sample_mse = self.criterion(X_hat, X_batch).mean(dim=1).cpu().numpy()
                errors.append(per_sample_mse)
                labels.append(y.numpy())
                
        if errors:
            errors = np.concatenate(errors)
            labels = np.concatenate(labels)
        else:
            errors = np.array([])
            labels = np.array([])
            
        return errors, labels