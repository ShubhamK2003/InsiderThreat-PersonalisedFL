# evaluation.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

import config

def evaluate_model(model, test_loader, labels):
    """
    Evaluates the model and returns a dictionary of key performance metrics.
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    all_scores = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.DEVICE)
            reconstructions = model(data)
            # Calculate reconstruction error per sample in the batch
            scores = criterion(reconstructions, data).mean(axis=1)
            all_scores.extend(scores.cpu().numpy())
    
    all_scores = np.array(all_scores)
    
    # Ensure labels and scores align if there's a batch size mismatch
    min_len = min(len(all_scores), len(labels))
    all_scores = all_scores[:min_len]
    labels = labels[:min_len]

    # -- Threshold-Independent Metrics --
    auroc = roc_auc_score(labels, all_scores)
    aupr = average_precision_score(labels, all_scores)

    # -- Threshold-Dependent Metrics --
    # Set a threshold to classify the top N% of scores as anomalies,
    # where N is the known injection rate. This is a standard way to
    # evaluate anomaly detectors without a separate validation set.
    threshold = np.percentile(all_scores, 100 * (1 - config.INJECTION_RATE))
    predictions = (all_scores > threshold).astype(int)
    
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        'AUROC': auroc, 
        'AUPR': aupr, 
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall
    }