# anomaly_injector_v2.py
"""
V2 Injector: "Subtle" anomalies.
This injector creates much "tougher" anomalies by focusing on
low-magnitude, pattern-based changes rather than large spikes.

"""
import numpy as np
import pandas as pd
import config

class AnomalyInjectorV2:
    def __init__(self, random_seed=config.RANDOM_SEED):
        self.rng = np.random.default_rng(random_seed)

    # ANOMALY METHODS (HARD) ---

    def _inject_low_and_slow_exfil(self, row):
        """
        Simulates a "low and slow" data theft.
        Small increases in multiple exfiltration-related features.
        """
        if 'file_export' in row.index:
            row['file_export'] += self.rng.integers(3, 7) 
        if 'external_share' in row.index:
            row['external_share'] += self.rng.integers(2, 4) 
        if 'file_download' in row.index:
            row['file_download'] *= self.rng.uniform(1.5, 2.0) 
        return row

    def _inject_internal_recon(self, row):
        """
        Simulates internal reconnaissance.
        Small, unusual activity in security/admin logs, which are normally 0.
        """
        if 'audit_log_access' in row.index:
            row['audit_log_access'] += self.rng.integers(2, 4) 
        if 'security_setting_change' in row.index:
            row['security_setting_change'] += self.rng.integers(2, 4) 
        return row

    def _inject_suspicious_login_pattern(self, row):
        """
        Simulates a compromised account with unusual login behavior.
        Combines a single login failure with an off-hours login.
        """
        if 'login_fail' in row.index:
            row['login_fail'] += self.rng.integers(4, 8) 
        if 'after_hours_login' in row.index:
            row['after_hours_login'] += self.rng.integers(2, 4) 
        if 'ip_change' in row.index:
            row['ip_change'] += self.rng.integers(2, 4) 
        return row

    def _inject_data_staging(self, row):
        """
        Simulates data staging before exfiltration.
        Unusual amounts of file creation (uploads) and deletion (cleanup).
        """
        if 'file_upload' in row.index:
            row['file_upload'] = row['file_upload'] * self.rng.uniform(1.8, 2.5) + self.rng.integers(5, 10) 
        if 'file_delete' in row.index:
            row['file_delete'] = row['file_delete'] * self.rng.uniform(1.8, 2.5) + self.rng.integers(5, 10) 
        if 'file_move' in row.index:
            row['file_move'] += self.rng.integers(5, 10) 
        return row
        
    def _inject_policy_violation(self, row):
        """
        Simulates a user trying to bypass security policies.
        """
        if 'tfa_disable' in row.index:
            row['tfa_disable'] += self.rng.integers(1, 2) # Was (1, 3), 1 or 2 is enough
        if 'policy_change' in row.index:
            row['policy_change'] += self.rng.integers(1, 2) # Was (1, 3)
        return row

    def _inject_data_hoarding(self, row):
        """
        Simulates a disgruntled employee hoarding data before leaving.
        Not exfiltration, but unusual internal gathering.
        """
        if 'file_download' in row.index:
            row['file_download'] = row['file_download'] * self.rng.uniform(2.0, 2.5) + self.rng.integers(10, 20) 
        if 'file_copy' in row.index:
            row['file_copy'] = row['file_copy'] * self.rng.uniform(2.0, 2.5) + self.rng.integers(10, 20) 
        if 'bulk_operation' in row.index:
            row['bulk_operation'] += self.rng.integers(2, 4) 
        return row

    def _inject_subtle_off_hours_work(self, row):
        """
        Simulates a user logging in at a weird time *and* doing a
        single sensitive action, a difficult pattern to spot.
        """
        if 'after_hours_login' in row.index:
            row['after_hours_login'] += self.rng.integers(1, 3) 
        if 'file_export' in row.index:
            row['file_export'] += self.rng.integers(3, 8) 
        return row
        
    def _inject_privilege_seeking(self, row):
        """
        Simulates a user testing their boundaries, trying to access
        or change permissions. These are normally zero.
        """
        if 'permission_escalation' in row.index:
            row['permission_escalation'] += self.rng.integers(1, 3) 
        if 'role_change' in row.index:
            row['role_change'] += self.rng.integers(1, 3) 
        return row

    def _inject_cleanup_activity(self, row):
        """
        Simulates a user trying to cover their tracks.
        Focuses on deletion, especially permanent deletion.
        """
        if 'file_delete' in row.index:
            row['file_delete'] = row['file_delete'] * self.rng.uniform(3.0, 4.0) + self.rng.integers(5, 10) 
        if 'file_permanently_delete' in row.index:
            row['file_permanently_delete'] += self.rng.integers(3, 8) 
        return row


    def inject(self, test_df):
        """
        Main method to inject anomalies into a copy of the test dataframe.
        """
        if len(test_df) == 0:
            return test_df.copy(), np.array([])
            
        print(f"💉 Injecting anomalies in SESSION mode (V2 - Hard Difficulty)...") 
        print(f" Injecting into {config.INJECTION_RATE*100:.2f}% of the {len(test_df)} test data points...")
        
        test_df_injected = test_df.copy()
        labels = np.zeros(len(test_df_injected))
        num_anomalies = int(len(test_df_injected) * config.INJECTION_RATE)
        
        if num_anomalies == 0 and len(test_df_injected) > 0:
             print(f"⚠️ Test set is too small ({len(test_df_injected)} samples) to inject {config.INJECTION_RATE*100}% anomalies. Injecting 1.")
             num_anomalies = 1
             
        anomaly_indices = self.rng.choice(test_df_injected.index, num_anomalies, replace=False)
        
        # --- Full list of 9 subtle methods ---
        injection_methods = [
            self._inject_low_and_slow_exfil,
            self._inject_internal_recon,
            self._inject_suspicious_login_pattern,
            self._inject_data_staging,
            self._inject_policy_violation,
            self._inject_data_hoarding,
            self._inject_subtle_off_hours_work,
            self._inject_privilege_seeking,
            self._inject_cleanup_activity
        ]
        
        for idx in anomaly_indices:
            method_to_use = self.rng.choice(injection_methods)
            row_copy = test_df_injected.loc[idx].copy()
            injected_row = method_to_use(row_copy)
            test_df_injected.loc[idx] = injected_row
            
            labels[test_df_injected.index.get_loc(idx)] = 1
            
        print(f"✅ Injected {int(np.sum(labels))} anomalies.")
        
        test_df_injected['is_anomaly'] = labels.astype(int)
        
        return test_df_injected