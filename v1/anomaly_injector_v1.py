# anomaly_injector.py
import numpy as np
import pandas as pd
import config

class AnomalyInjector:
    """
    Injects synthetic anomalies into the test set to create a ground truth for evaluation.
    This version operates *only* at the session level.
    """
    def __init__(self, random_seed=config.RANDOM_SEED):
        self.rng = np.random.default_rng(random_seed)

    # --- "LOUD" ANOMALY METHODS ---
    def _inject_loud_exfiltration(self, row):
        """Simulates large-scale, obvious data theft."""
        params = {'mult': (5, 10), 'add': (20, 40)} # Session params
        exfil_cols = ['file_export', 'external_share', 'bulk_operation', 'file_download']
        for col in exfil_cols:
            if col in row.index:
                row[col] *= self.rng.uniform(*params['mult'])
                row[col] += self.rng.integers(*params['add'])
        return row

    def _inject_loud_privilege_escalation(self, row):
        """Simulates a very noisy privilege abuse event."""
        params = {'add': (2, 3)} # Session params
        priv_cols = ['permission_escalation', 'role_change', 'policy_change', 'audit_log_access']
        for col in priv_cols:
            if col in row.index:
                row[col] += self.rng.integers(*params['add'])
        return row

    def _inject_loud_off_hours_activity(self, row):
        """Simulates very obvious suspicious work at unusual times with high activity."""
        params = {'login': (1, 2), 'mult': (10, 20)} # Session params
        time_cols = ['after_hours_login', 'weekend_login']
        for col in time_cols:
            if col in row.index:
                row[col] += self.rng.integers(*params['login'])
        
        activity_cols = ['file_upload', 'file_move', 'shared_link_create']
        for col in activity_cols:
            if col in row.index:
                row[col] *= self.rng.uniform(*params['mult'])
        return row

    # --- "SUBTLE" ANOMALY METHODS ---
    def _inject_subtle_exfiltration(self, row):
        """Simulates a 'low and slow' data theft."""
        params = {'mult': (1.5, 2), 'add': (2, 5)} # Session params
        exfil_cols = ['file_export', 'external_share', 'file_download']
        for col in exfil_cols:
             if col in row.index:
                row[col] *= self.rng.uniform(*params['mult'])
                row[col] += self.rng.integers(*params['add'])
        return row

    def _inject_behavior_shift(self, row):
        """Simulates an unusual *combination* of behaviors."""
        params = {'login': (1, 2), 'share': (1, 3), 'del': (1, 3)} # Session params
        if 'after_hours_login' in row.index:
            row['after_hours_login'] += self.rng.integers(*params['login'])
        if 'external_share' in row.index:
            row['external_share'] += self.rng.integers(*params['share'])
        if 'file_delete' in row.index:
            row['file_delete'] += self.rng.integers(*params['del'])
        return row

    def _inject_failed_login_spike(self, row):
        """Simulates a potential brute-force or password-guessing attempt."""
        params = {'add': (5, 15)} # Session params
        if 'login_fail' in row.index:
            row['login_fail'] += self.rng.integers(*params['add'])
        return row
        
    def inject(self, test_df):
        """
        Main method to inject anomalies into a copy of the test dataframe.
        Modifies the numerical feature columns directly.
        Returns:
            - test_df_injected: A copy of the DataFrame with anomalies.
            - labels: A numpy array of 0s and 1s.
        """
        if len(test_df) == 0:
            return test_df.copy(), np.array([])
            
        print(f"💉 Injecting anomalies in SESSION mode...")
        print(f" Injecting into {config.INJECTION_RATE*100:.2f}% of the {len(test_df)} test data points...")
        
        test_df_injected = test_df.copy()
        labels = np.zeros(len(test_df_injected))
        num_anomalies = int(len(test_df_injected) * config.INJECTION_RATE)
        
        if num_anomalies == 0 and len(test_df_injected) > 0:
             print(f"⚠️ Test set is too small ({len(test_df_injected)} samples) to inject {config.INJECTION_RATE*100}% anomalies. Injecting 1.")
             num_anomalies = 1
             
        anomaly_indices = self.rng.choice(test_df_injected.index, num_anomalies, replace=False)
        
        # Our list of methods includes both loud and subtle types for variety
        injection_methods = [
            self._inject_loud_exfiltration,
            self._inject_loud_privilege_escalation,
            self._inject_loud_off_hours_activity,
            self._inject_subtle_exfiltration,
            self._inject_behavior_shift,
            self._inject_failed_login_spike
        ]
        
        for idx in anomaly_indices:
            # Choose a random anomaly type to inject
            method_to_use = self.rng.choice(injection_methods)
            
            # Apply method to the row
            row_copy = test_df_injected.loc[idx].copy()
            injected_row = method_to_use(row_copy)
            test_df_injected.loc[idx] = injected_row
            
            labels[test_df_injected.index.get_loc(idx)] = 1
            
        print(f"✅ Injected {int(np.sum(labels))} anomalies.")
        
        # Add 'is_anomaly' column to the DataFrame for the client to use
        test_df_injected['is_anomaly'] = labels.astype(int)
        
        return test_df_injected