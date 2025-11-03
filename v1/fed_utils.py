# fed_utils.py
"""
Helpers for federated ops:
- safe_cpu_average_state_dicts: aggregate state dicts on CPU/dtype float32 (WEIGHTED)
- reset_optimizer_state_for_encoder: clear Adam/SGD moments for encoder params 
                                     when encoder weights get replaced
"""
import torch
import copy
import re
from collections import OrderedDict

def safe_cpu_average_state_dicts(state_dicts_with_samples):
    """
    state_dicts_with_samples: list of tuples: (state_dict, num_samples)
    Returns weighted averaged state_dict moved to CPU with float32 tensors.
    """
    if not state_dicts_with_samples:
        return None
    
    avg = OrderedDict()
    
    # Get keys from the first client's update
    first_update_dict = state_dicts_with_samples[0][0]
    for key in first_update_dict.keys():
        avg[key] = torch.zeros_like(first_update_dict[key]).float().cpu()
        
    # Calculate the total number of samples
    total_samples = sum(num_samples for _, num_samples in state_dicts_with_samples)
    
    if total_samples == 0:
        # This should not happen if we check for empty client lists
        return first_update_dict 
        
    # Perform the weighted sum
    for update_dict, num_samples in state_dicts_with_samples:
        weight = num_samples / total_samples
        for key in avg.keys():
            if key in update_dict:
                 avg[key] += update_dict[key].cpu().float() * weight
            else:
                # This might happen if a key (e.g., decoder) isn't in an update (e.g., FedRep)
                # But for aggregation, all dicts should have the same keys.
                pass 
                
    return avg

def reset_optimizer_state_for_encoder(optimizer, model, encoder_prefix='encoder'):
    """
    When encoder parameters are replaced on a client, reset optimizer.state for those params
    so Adam/SGD don't use stale moments.
    """
    param_to_name = {p: n for n, p in model.named_parameters()}
    
    encoder_param_ids = set()
    for name, p in model.named_parameters():
        if name.startswith(encoder_prefix):
            encoder_param_ids.add(id(p))
            
    for param_tensor, state in list(optimizer.state.items()):
        if id(param_tensor) in encoder_param_ids:
            del optimizer.state[param_tensor]