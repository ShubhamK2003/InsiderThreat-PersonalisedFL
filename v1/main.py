# main.py
"""
Main training driver that runs FedAvg and FedRep experiments.

FedPer implementation:
 - Server holds global encoder weights only.
 - Each client receives global encoder, sets local encoder (keeps private decoder),
   trains locally, and returns only encoder state_dict to server.
 - Server performs a weighted average of encoder weights and broadcasts them.

Evaluation:
 - Per-client thresholds are computed dynamically at evaluation time
   using the model's current state on the train data (mean + k*std).
"""
import copy
import torch
import numpy as np
import random
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from models import Encoder, Decoder, LocalAutoencoder # V1 models
import config
from data_loader import prepare_clients_for_training # V1 data loader
if config.ANOMALY_INJECTOR_VERSION == 'v1':
    from anomaly_injector_v1 import AnomalyInjector
    print("🚀 Using Anomaly Injector V1 (Loud)")
elif config.ANOMALY_INJECTOR_VERSION == 'v2':
    from anomaly_injector_v2 import AnomalyInjectorV2 as AnomalyInjector
    print("🚀 Using Anomaly Injector V2 (Subtle)")
else:
    raise ValueError(f"Unknown ANOMALY_INJECTOR_VERSION: {config.ANOMALY_INJECTOR_VERSION}")
# ---
from client import Client 
from server import Server
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from fed_utils import safe_cpu_average_state_dicts 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config.TORCH_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def build_clients(clients_dict):
    """Initializes Client objects from the client data map."""
    clients = {}
    for uid, cobj in clients_dict.items():
        clients[uid] = Client(uid, cobj, device=config.DEVICE)
    return clients

def initialize_global_encoder():
    """Creates an initial encoder and returns its state dict on CPU."""
    enc = Encoder().to(config.DEVICE)
    return {k: v.cpu().float().clone() for k, v in enc.state_dict().items()}

def evaluate_clients(clients):
    """
    Compute pooled metrics across clients using V1's dynamic threshold method.
    """
    all_preds = []
    all_labels = []
    all_scores = []
    
    for uid, client in clients.items():
        # 1. Get reconstruction errors on *train* data
        train_errors, _ = client.eval_reconstruction_errors(on='train')
        
        if len(train_errors) == 0:
            threshold = float('inf') # No train data, flag nothing
        else:
            # 2. Calculate the valid threshold for the *current* model
            threshold = float(np.mean(train_errors) + config.THRESHOLD_STD_MULT * np.std(train_errors))
            
        # 3. Get errors on *test* data
        errors, labels = client.eval_reconstruction_errors(on='test')
        if len(errors) == 0:
            continue
            
        # 4. Classify test data using the *new* threshold
        scores = errors
        preds = (scores > threshold).astype(int)
        
        all_preds.append(preds)
        all_labels.append(labels)
        all_scores.append(scores)

    if len(all_labels) == 0 or len(np.concatenate(all_labels)) == 0:
        return {'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0, 'AUROC': 0.0, 'AUPR': 0.0}
        
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_score = np.concatenate(all_scores)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
    else:
        auroc = 0.0
        aupr = 0.0

    return {
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1),
        'AUROC': float(auroc),
        'AUPR': float(aupr)
    }

# --- Experiment runners ---
def run_fedrep(clients, server, num_rounds=config.NUM_ROUNDS):
    """
    FedPer loop with weighted aggregation and V2-style logging.
    """
    history = {'loss': [], 'recall': []}
    for r in range(1, num_rounds + 1):
        encoder_updates = []
        round_losses = []
        total_samples = 0
        
        for uid, client in clients.items():
            client.set_encoder_state_dict_from_server(server.get_global_encoder())
            
            num_samples, avg_loss = client.local_train(
                local_epochs=config.LOCAL_EPOCHS
            )
            
            # Collect updates for weighted averaging
            encoder_updates.append((client.get_encoder_state_dict(), num_samples))
            round_losses.append(avg_loss * num_samples) # Store weighted loss
            total_samples += num_samples
            
        # server aggregation (WEIGHTED)
        if encoder_updates:
            new_encoder = server.aggregate_encoder_updates(encoder_updates)
            
        # evaluate
        metrics = evaluate_clients(clients)
        avg_round_loss = sum(round_losses) / total_samples if total_samples > 0 else 0.0
        
        history['loss'].append(avg_round_loss)
        history['recall'].append(metrics.get('Recall', 0.0))
        
        if config.VERBOSE:
             print(f"Mode [FEDREP] | Round [{r}/{num_rounds}] | Avg Train Loss: {avg_round_loss:.4f} | Test Recall: {metrics.get('Recall', 0.0):.4f}")
            
    return history

def run_local(clients, local_epochs=config.LOCAL_EPOCHS):
    """
    Baseline: each client trains locally. 
    """
    print("Running local-only training...")
    round_losses = []
    total_samples = 0
    
    for uid, client in clients.items():
        num_samples, avg_loss = client.local_train(
            local_epochs=local_epochs # Use combined epochs
        )
        round_losses.append(avg_loss * num_samples)
        total_samples += num_samples
    
    metrics = evaluate_clients(clients)
    avg_round_loss = sum(round_losses) / total_samples if total_samples > 0 else 0.0
    
    print(f"[LOCAL] Completed local-only training. Metrics: {metrics}")
    
    history = {
        'loss': [avg_round_loss] * config.NUM_ROUNDS,
        'recall': [metrics.get('Recall', 0.0)] * config.NUM_ROUNDS
    }
    return history, metrics

def run_fedavg(clients, num_rounds=config.NUM_ROUNDS):
    """
    FedAvg baseline with weighted aggregation.
    """
    history = {'loss': [], 'recall': []}
    
    avg = {k: v.cpu().clone().detach() for k, v in list(clients.values())[0].model.state_dict().items()}
    
    for r in range(1, num_rounds + 1):
        full_updates = []
        round_losses = []
        total_samples = 0
        
        for uid, client in clients.items():
            avg_device = {k: v.to(client.device) for k, v in avg.items()}
            client.model.load_state_dict(avg_device, strict=True)
            client.optimizer = torch.optim.Adam(client.model.parameters(), lr=config.LEARNING_RATE)

            num_samples, avg_loss = client.local_train(
                local_epochs=config.LOCAL_EPOCHS
            )
            
            sd = {k: v.cpu().clone().detach() for k, v in client.model.state_dict().items()}
            full_updates.append((sd, num_samples))
            round_losses.append(avg_loss * num_samples)
            total_samples += num_samples
        
        if full_updates:
            avg = safe_cpu_average_state_dicts(full_updates)
        
        metrics = evaluate_clients(clients)
        avg_round_loss = sum(round_losses) / total_samples if total_samples > 0 else 0.0
        
        history['loss'].append(avg_round_loss)
        history['recall'].append(metrics.get('Recall', 0.0))
        
        if config.VERBOSE:
            print(f"Mode [FEDAVG] | Round [{r}/{num_rounds}] | Avg Train Loss: {avg_round_loss:.4f} | Test Recall: {metrics.get('Recall', 0.0):.4f}")
            
    return history

# --- Main orchestration logic ---
def main():
    set_seed(config.RANDOM_SEED)
    os.makedirs('plots', exist_ok=True)
    
    client_data_map = prepare_clients_for_training(config.DATA_FILE)
    
    print("\n" + "="*40)
    print("💉 Injecting Anomalies per-client...")
    print("="*40)
    
    # This instance is created from the dynamically imported class
    injector = AnomalyInjector() 
    
    for client_id, data_obj in client_data_map.items():
        clean_test_df = data_obj['test_df']
        if len(clean_test_df) > 0:
            print(f"--- Client: {client_id} ---")
            injected_test_df = injector.inject(clean_test_df)
            data_obj['test_df'] = injected_test_df
        else:
             print(f"--- Client: {client_id} (No test data to inject) ---")
    print("="*40 + "\n")

    print("\n" + "="*40)
    print("📊 Per-client data (train/test counts)")
    print("="*40)
    train_total = 0
    test_total = 0
    test_anomaly_total = 0
    for uid, data_obj in client_data_map.items():
        train_df = data_obj['train_df']
        test_df = data_obj['test_df']
        
        train_n = len(train_df)
        test_n = len(test_df)
        test_a = 0
        if 'is_anomaly' in test_df.columns:
            test_a = int(test_df['is_anomaly'].sum())
            
        print(f" {uid}: train={train_n}, test={test_n} (anomalies={test_a})")
        train_total += train_n
        test_total += test_n
        test_anomaly_total += test_a
        
    print(f" TOTAL: train={train_total}, test={test_total} (anomalies={test_anomaly_total})")
    print("="*40 + "\n")
    
    MODES_TO_RUN = config.DEFAULT_MODES.copy()
    results = []
    plot_data = {m: {'loss': [], 'recall': []} for m in MODES_TO_RUN}
    
    for mode in MODES_TO_RUN:
        print(f"\n{'='*10} Running experiment: {mode.upper()} {'='*10}")
        
        set_seed(config.RANDOM_SEED)
        clients = build_clients(client_data_map)
        
        if not clients:
            print(f"⚠️ No clients created for mode {mode}. Skipping.")
            continue
            
        global_encoder_sd = initialize_global_encoder()
        server = Server(global_encoder_sd) 
        
        if mode != 'fedavg':
            for uid, c in clients.items():
                c.set_encoder_state_dict_from_server(global_encoder_sd)

        start_time = time.time()
        history = {} 
        final_metrics = {}
        
        if mode == 'local':
            history, final_metrics = run_local(
                clients, 
                local_epochs=config.LOCAL_EPOCHS * config.NUM_ROUNDS
            )
            
        elif mode == 'fedavg':
            history = run_fedavg(clients, num_rounds=config.NUM_ROUNDS)
            
        elif mode == 'fedrep':
            history = run_fedrep(clients, server, num_rounds=config.NUM_ROUNDS)
            
        end_time = time.time()
        total_time = end_time - start_time
        
        plot_data[mode] = history
        
        if mode != 'local':
            final_metrics = evaluate_clients(clients)
            
        print(f"\n🏁 Final metrics for {mode.upper()}: {final_metrics}")
        
        final_res = {'Method': mode.upper(), **final_metrics}
        final_res['Total Time (s)'] = total_time
        final_res['Avg Time/Round (s)'] = total_time / config.NUM_ROUNDS if config.NUM_ROUNDS > 0 else 0.0
        results.append(final_res)

    print("\n\n" + "="*30 + " SUMMARY " + "="*30)
    summary_df = pd.DataFrame(results).set_index('Method')
    cols = ['Recall', 'Precision', 'F1-Score', 'AUROC', 'AUPR', 'Total Time (s)', 'Avg Time/Round (s)']
    for c in cols:
        if c not in summary_df.columns:
            summary_df[c] = 0.0
    print(summary_df[cols].to_string(float_format="%.4f"))
    print("="*70)
    
    print(f"📊 Generating and saving plots...")
    rounds = np.arange(1, config.NUM_ROUNDS + 1)
    
    plt.figure(figsize=(10, 5))
    for mode in MODES_TO_RUN:
        values = plot_data[mode]['loss']
        plt.plot(rounds, values, label=mode.upper(), marker='o', markersize=4)
    plt.xlabel("Communication Round")
    plt.ylabel("Average Train Loss")
    plt.title("Average Training Loss vs. Communication Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/train_loss_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for mode in MODES_TO_RUN:
        values = plot_data[mode]['recall']
        plt.plot(rounds, values, label=mode.upper(), marker='o', markersize=4)
    plt.xlabel("Communication Round")
    plt.ylabel("Test Recall")
    plt.title("Test Recall vs. Communication Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/test_recall_comparison.png')
    plt.close()
    
    print(f"✅ Plots saved in ./plots (train_loss_comparison.png, test_recall_comparison.png)")
    print("Done.")

if __name__ == "__main__":
    main()