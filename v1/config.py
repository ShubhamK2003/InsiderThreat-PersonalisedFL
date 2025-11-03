# config.py
import torch

# -- Data Configuration --
DATA_FILE = 'user_logs.csv'
# The features to be used for modeling. We exclude identifiers, dates, and raw text.
NUMERICAL_FEATURES = [
    'session_duration_min', 'login_success', 'login_fail', 'logout',
    'after_hours_login', 'weekend_login', 'device_link', 'device_unlink',
    'ip_change', 'location_change', 'file_upload', 'file_download', 'file_delete',
    'file_export', 'file_permanently_delete', 'file_move', 'file_copy',
    'bulk_operation', 'shared_link_create', 'shared_link_download', 'external_share',
    'share_permission_change', 'shared_folder_create', 'mass_sharing', 'password_change',
    'tfa_disable', 'tfa_add', 'security_setting_change', 'audit_log_access',
    'role_change', 'permission_escalation', 'policy_change'
]
INPUT_DIM = len(NUMERICAL_FEATURES)

# The column used to partition users into clients
CLIENT_PARTITION_COLUMN = 'user_role' # Department-wise 

# -- Preprocessing --
# Temporal split for train/test sets. 0.7 means 70% of time for training, 30% for testing.
TRAIN_SPLIT_RATIO = 0.7

# -- Anomaly Injection Configuration --
# The percentage of sessions in the test set to be turned into anomalies
INJECTION_RATE = 0.2

# Anomaly injector version
ANOMALY_INJECTOR_VERSION = 'v2'

# -- Model Hyperparameters --
# The dimensionality of the autoencoder's bottleneck layer
LATENT_DIM = 24

# -- Federated Learning Hyperparameters --
# Total number of communication rounds between server and clients
NUM_ROUNDS = 45
# Number of local training epochs on each client before sending an update to server
LOCAL_EPOCHS = 5
# Learning rate for the optimizers
LEARNING_RATE = 0.001
# Batch size
BATCH_SIZE = 64

# Number of standard deviations above train mean to call anomaly.
THRESHOLD_STD_MULT = 2 

# -- General Training Configuration --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
DEFAULT_MODES = ['fedavg', 'fedrep']
VERBOSE = True
TORCH_DETERMINISTIC = False