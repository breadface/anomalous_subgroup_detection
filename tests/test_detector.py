"""
Tests for the anomalous subgroup detection package.
These tests verify that the detector and scan statistics work as expected.
"""

import pytest
import numpy as np
import pandas as pd
from anomalous_subgroup_detection import SubsetScanDetector
from anomalous_subgroup_detection.scan_statistics import gaussian_scan_statistic, bernoulli_scan_statistic

def test_gaussian_scan_statistic():
    """Test the Gaussian scan statistic function with various inputs."""
    # Test with normal data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    null_mean = 3.0
    null_std = 1.0
    score = gaussian_scan_statistic(data, null_mean, null_std)
    assert isinstance(score, float)
    assert not np.isinf(score)

    # Test with empty data
    empty_data = np.array([])
    score = gaussian_scan_statistic(empty_data, null_mean, null_std)
    assert np.isinf(score) and score < 0

    # Test with identical values
    identical_data = np.array([1.0, 1.0, 1.0])
    score = gaussian_scan_statistic(identical_data, 1.0, 0.0)
    assert score == 0.0

def test_bernoulli_scan_statistic():
    """Test the Bernoulli scan statistic function with various inputs."""
    # Test with normal data
    score = bernoulli_scan_statistic(5, 10, 0.5)
    assert isinstance(score, float)
    assert not np.isinf(score)

    # Test with empty data
    score = bernoulli_scan_statistic(0, 0, 0.5)
    assert np.isinf(score) and score < 0

    # Test with invalid inputs
    with pytest.raises(ValueError):
        bernoulli_scan_statistic(11, 10, 0.5)  # counts > n

def test_subset_scan_detector_initialization():
    """Test the initialization of the SubsetScanDetector class."""
    # Test valid initialization
    detector = SubsetScanDetector(
        statistic_type='gaussian',
        features=['feature_A', 'feature_B'],
        max_combination_size=2,
        num_permutations=100
    )
    assert detector.statistic_type == 'gaussian'
    assert detector.features == ['feature_A', 'feature_B']
    assert detector.max_combination_size == 2
    assert detector.num_permutations == 100

    # Test invalid statistic type
    with pytest.raises(ValueError):
        SubsetScanDetector(
            statistic_type='invalid',
            features=['feature_A'],
            max_combination_size=1,
            num_permutations=100
        )

    # Test empty features list
    with pytest.raises(ValueError):
        SubsetScanDetector(
            statistic_type='gaussian',
            features=[],
            max_combination_size=1,
            num_permutations=100
        )

def test_detector_fit_and_scan():
    """Test the fit_and_scan method of the SubsetScanDetector class."""
    # Create synthetic data
    data = pd.DataFrame({
        'feature_A': ['typeA', 'typeB', 'typeA', 'typeB'],
        'feature_B': ['regionX', 'regionY', 'regionX', 'regionY'],
        'amount': [100, 200, 100, 200],  # typeB/regionY has higher amounts
        'is_fraud': [0, 1, 0, 1]  # typeB/regionY has higher fraud rate
    })

    # Test Gaussian scan
    detector = SubsetScanDetector(
        statistic_type='gaussian',
        features=['feature_A', 'feature_B'],
        max_combination_size=2,
        num_permutations=10  # Small number for quick testing
    )
    detector.fit_and_scan(data, 'amount')
    results = detector.get_results()
    
    assert 'best_score' in results
    assert 'best_subset_definition' in results
    assert 'p_value' in results
    assert 'detected_subgroup_indices' in results

    # Test Bernoulli scan
    detector = SubsetScanDetector(
        statistic_type='bernoulli',
        features=['feature_A', 'feature_B'],
        max_combination_size=2,
        num_permutations=10
    )
    detector.fit_and_scan(data, 'is_fraud')
    results = detector.get_results()
    
    assert 'best_score' in results
    assert 'best_subset_definition' in results
    assert 'p_value' in results
    assert 'detected_subgroup_indices' in results

def test_detector_with_missing_features():
    """Test the detector's behavior with missing features in the data."""
    data = pd.DataFrame({
        'feature_A': ['typeA', 'typeB'],
        'amount': [100, 200]
    })

    detector = SubsetScanDetector(
        statistic_type='gaussian',
        features=['feature_A', 'feature_B'],  # feature_B is missing
        max_combination_size=2,
        num_permutations=10
    )

    with pytest.raises(ValueError):
        detector.fit_and_scan(data, 'amount') 