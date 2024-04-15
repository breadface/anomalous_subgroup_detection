"""
Simple example demonstrating how to use the anomalous subgroup detection package.
This example shows both Gaussian and Bernoulli scan statistics in action.
"""

import pandas as pd
import numpy as np
from anomalous_subgroup_detection import SubsetScanDetector

def generate_synthetic_data(num_samples=1000, anomaly_size=50, anomaly_magnitude=5.0, anomaly_proportion=0.8):
    """
    Creates a synthetic dataset with a hidden anomalous subgroup.
    The anomaly is injected into a specific combination of feature values.
    """
    np.random.seed(42)  # for reproducibility

    # Base data
    data = pd.DataFrame({
        'feature_A': np.random.choice(['typeA', 'typeB', 'typeC'], num_samples),
        'feature_B': np.random.choice(['regionX', 'regionY', 'regionZ'], num_samples),
        'feature_C': np.random.choice(['source1', 'source2'], num_samples),
        'amount': np.random.normal(loc=100, scale=20, size=num_samples),
        'is_fraud': np.random.choice([0, 1], p=[0.98, 0.02], size=num_samples)  # Low base fraud rate
    })

    # Inject an anomalous subgroup
    anomalous_feature_A = 'typeB'
    anomalous_feature_B = 'regionY'
    anomalous_feature_C = 'source1'

    # Find indices for the anomalous subgroup
    anomalous_group_mask = (data['feature_A'] == anomalous_feature_A) & \
                          (data['feature_B'] == anomalous_feature_B) & \
                          (data['feature_C'] == anomalous_feature_C)

    # If we don't have enough samples for the anomaly, create more
    if anomalous_group_mask.sum() < anomaly_size:
        print(f"Not enough natural samples for anomaly group ({anomalous_group_mask.sum()}), adding more.")
        num_to_add = anomaly_size - anomalous_group_mask.sum()
        additional_anomalies = pd.DataFrame({
            'feature_A': [anomalous_feature_A] * num_to_add,
            'feature_B': [anomalous_feature_B] * num_to_add,
            'feature_C': [anomalous_feature_C] * num_to_add,
            'amount': np.random.normal(loc=100 + anomaly_magnitude, scale=10, size=num_to_add),
            'is_fraud': np.random.choice([0, 1], p=[1 - anomaly_proportion, anomaly_proportion], size=num_to_add)
        })
        data = pd.concat([data, additional_anomalies], ignore_index=True)
        anomalous_group_mask = (data['feature_A'] == anomalous_feature_A) & \
                              (data['feature_B'] == anomalous_feature_B) & \
                              (data['feature_C'] == anomalous_feature_C)

    # Modify 'amount' for the anomalous subgroup
    data.loc[anomalous_group_mask, 'amount'] = np.random.normal(
        loc=data.loc[anomalous_group_mask, 'amount'].mean() + anomaly_magnitude,
        scale=data.loc[anomalous_group_mask, 'amount'].std() * 0.5,  # Make it tighter
        size=anomalous_group_mask.sum()
    )

    # Modify 'is_fraud' for the anomalous subgroup
    data.loc[anomalous_group_mask, 'is_fraud'] = np.random.choice(
        [0, 1], p=[1 - anomaly_proportion, anomaly_proportion], size=anomalous_group_mask.sum()
    )

    print(f"Generated {len(data)} samples. Anomalous subgroup size: {anomalous_group_mask.sum()}")
    return data

def main():
    # --- Example 1: Gaussian Scan on 'amount' ---
    print("\n--- Running Gaussian Scan Example ---")
    data_gaussian = generate_synthetic_data(num_samples=2000, anomaly_size=70, anomaly_magnitude=50.0)

    # Features to define subgroups
    features_to_scan = ['feature_A', 'feature_B', 'feature_C']
    target_amount = 'amount'

    # Initialize the detector
    gaussian_detector = SubsetScanDetector(
        statistic_type='gaussian',
        features=features_to_scan,
        max_combination_size=3,
        num_permutations=200  # More permutations for better p-value accuracy
    )

    # Run the scan
    gaussian_detector.fit_and_scan(data_gaussian, target_amount)

    # Get results
    results_gaussian = gaussian_detector.get_results()

    print("\n--- Gaussian Scan Results ---")
    print(f"Best Score: {results_gaussian['best_score']:.4f}")
    print(f"Best Subset Definition: {results_gaussian['best_subset_definition']}")
    print(f"P-value: {results_gaussian['p_value']:.4f}")
    print(f"Detected Subgroup Size: {len(results_gaussian['detected_subgroup_indices'])}")
    if results_gaussian['p_value'] < 0.05:
        print("Interpretation: The detected subgroup is statistically significant (p < 0.05).")
    else:
        print("Interpretation: The detected subgroup is NOT statistically significant (p >= 0.05).")

    # --- Example 2: Bernoulli Scan on 'is_fraud' ---
    print("\n--- Running Bernoulli Scan Example ---")
    data_bernoulli = generate_synthetic_data(num_samples=2000, anomaly_size=70, anomaly_proportion=0.5)

    target_fraud = 'is_fraud'

    bernoulli_detector = SubsetScanDetector(
        statistic_type='bernoulli',
        features=features_to_scan,
        max_combination_size=3,
        num_permutations=200  # More permutations for better p-value accuracy
    )

    bernoulli_detector.fit_and_scan(data_bernoulli, target_fraud)

    results_bernoulli = bernoulli_detector.get_results()

    print("\n--- Bernoulli Scan Results ---")
    print(f"Best Score: {results_bernoulli['best_score']:.4f}")
    print(f"Best Subset Definition: {results_bernoulli['best_subset_definition']}")
    print(f"P-value: {results_bernoulli['p_value']:.4f}")
    print(f"Detected Subgroup Size: {len(results_bernoulli['detected_subgroup_indices'])}")
    if results_bernoulli['p_value'] < 0.05:
        print("Interpretation: The detected subgroup is statistically significant (p < 0.05).")
    else:
        print("Interpretation: The detected subgroup is NOT statistically significant (p >= 0.05).")

if __name__ == "__main__":
    main() 