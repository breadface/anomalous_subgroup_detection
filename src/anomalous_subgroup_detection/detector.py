import numpy as np
import pandas as pd
import itertools
from typing import Optional, List, Any
from .scan_statistics import gaussian_scan_statistic, bernoulli_scan_statistic

class SubsetScanDetector:
    """
    Detects anomalous subgroups in tabular data using scan statistics.
    Supports both Gaussian and Bernoulli scan statistics and can search for the most anomalous subgroup.
    """
    def __init__(
        self,
        statistic_type: str = 'gaussian',
        features: Optional[List[str]] = None,
        max_combination_size: int = 2,
        num_permutations: int = 100,
    ):
        """
        Args:
            statistic_type (str): 'gaussian' or 'bernoulli'.
            features (list, optional): List of feature columns to consider for subgrouping.
            max_combination_size (int): Maximum number of features to combine for subgroup search.
            num_permutations (int): Number of random subgroups to evaluate (for large feature sets).
        """
        if features is not None and len(features) == 0:
            raise ValueError("features list cannot be empty")
        self.statistic_type = statistic_type
        self.features = features
        self.max_combination_size = max_combination_size
        self.num_permutations = num_permutations
        self.data = None
        self.target_column = None
        self.null_params = None
        self._last_results = None
        if statistic_type == 'gaussian':
            self.scan_statistic = gaussian_scan_statistic
        elif statistic_type == 'bernoulli':
            self.scan_statistic = bernoulli_scan_statistic
        else:
            raise ValueError("statistic_type must be 'gaussian' or 'bernoulli'")

    def fit(self, data: pd.DataFrame, target_column: Optional[str] = None):
        """
        Fit the detector to the data (estimate null parameters).
        Args:
            data (pd.DataFrame): The input data.
            target_column (str, optional): Name of the target column.
        """
        self.data = data.copy()
        if self.features is None:
            self.features = [col for col in data.columns if col != target_column]
        if any(f not in data.columns for f in self.features):
            raise ValueError("One or more features are missing from the data.")
        self.target_column = target_column
        if target_column is not None:
            target = data[target_column]
            if self.statistic_type == 'bernoulli':
                self.null_params = {'proportion': target.mean()}
            else:
                self.null_params = {'mean': target.mean(), 'std': target.std()}
        else:
            self.null_params = {'mean': data.values.mean(), 'std': data.values.std()}

    def scan(self, subgroup_mask: np.ndarray) -> float:
        """
        Compute the scan statistic for a given subgroup.
        Args:
            subgroup_mask (np.ndarray): Boolean mask for the subgroup.
        Returns:
            float: Scan statistic value for the subgroup.
        """
        if self.data is None or self.null_params is None:
            raise ValueError("Detector must be fit before scanning.")
        subset = self.data[subgroup_mask]
        if self.target_column is not None:
            target = subset[self.target_column]
            if self.statistic_type == 'bernoulli':
                return self.scan_statistic(target.sum(), len(target), self.null_params['proportion'])
            else:
                return self.scan_statistic(target.values, self.null_params['mean'], self.null_params['std'])
        else:
            return self.scan_statistic(subset.values, self.null_params['mean'], self.null_params['std'])

    def fit_and_scan(self, data: pd.DataFrame, target_column: Optional[str] = None) -> dict:
        """
        Fit the detector and search for the most anomalous subgroup.
        Args:
            data (pd.DataFrame): The input data.
            target_column (str, optional): Name of the target column.
        Returns:
            dict: Information about the most anomalous subgroup found.
        """
        self.fit(data, target_column)
        best_score = -np.inf
        best_mask = None
        best_description = None
        for k in range(1, min(self.max_combination_size, len(self.features)) + 1):
            for feature_combo in itertools.combinations(self.features, k):
                unique_values = [self.data[feat].unique() for feat in feature_combo]
                for value_combo in itertools.product(*unique_values):
                    mask = np.ones(len(self.data), dtype=bool)
                    description = {}
                    for feat, val in zip(feature_combo, value_combo):
                        mask &= (self.data[feat] == val)
                        description[feat] = val
                    if mask.sum() == 0:
                        continue
                    score = self.scan(mask)
                    if score > best_score:
                        best_score = score
                        best_mask = mask.copy()
                        best_description = description.copy()
        # Compute p-value using permutation tests
        p_value = self._compute_p_value(best_score, best_mask)
        self._last_results = {
            'best_score': best_score,
            'mask': best_mask,
            'best_subset_definition': best_description,
            'p_value': p_value,
            'detected_subgroup_indices': np.where(best_mask)[0].tolist()
        }
        return self._last_results

    def _compute_p_value(self, observed_score: float, observed_mask: np.ndarray) -> float:
        """
        Compute p-value using permutation tests.
        Args:
            observed_score (float): The observed scan statistic score.
            observed_mask (np.ndarray): The mask for the observed subgroup.
        Returns:
            float: The p-value.
        """
        if self.num_permutations <= 0:
            return 0.0
        perm_scores = []
        for _ in range(self.num_permutations):
            perm_mask = np.random.permutation(observed_mask)
            perm_score = self.scan(perm_mask)
            perm_scores.append(perm_score)
        p_value = (np.sum(np.array(perm_scores) >= observed_score) + 1) / (self.num_permutations + 1)
        return p_value

    def get_results(self) -> Any:
        """
        Return the results from the last fit_and_scan call.
        Returns:
            dict or None: The last results dictionary, or None if not run yet.
        """
        return self._last_results 