# Anomalous Subgroup Detection

A Python library for detecting anomalous subgroups in data using subset scanning techniques. This implementation is inspired by the paper ["Subset Scanning Over Neural Network Activations"](https://arxiv.org/pdf/1810.08676) but generalized for broader applications.

## Overview

This library implements a statistical approach to detect anomalous subgroups in data by scanning through potential subsets defined by combinations of feature values. It uses likelihood ratio tests to identify subgroups that significantly deviate from the expected distribution.

Key features:
- Supports both Gaussian (continuous) and Bernoulli (binary) data distributions
- Identifies statistically significant anomalous subgroups
- Provides p-values through permutation testing
- Handles categorical features for subgroup definition
- Configurable search depth for feature combinations

## Installation

```bash
pip install anomalous-subgroup-detection
```

## Quick Start

```python
from anomalous_subgroup_detection import SubsetScanDetector

# Initialize detector
detector = SubsetScanDetector(
    statistic_type='gaussian',  # or 'bernoulli'
    features=['feature_A', 'feature_B'],
    max_combination_size=2,
    num_permutations=100
)

# Run the scan
detector.fit_and_scan(data, target_column='amount')

# Get results
results = detector.get_results()
print(f"Best Score: {results['best_score']}")
print(f"Best Subset: {results['best_subset_definition']}")
print(f"P-value: {results['p_value']}")
```

## Use Cases

- Fraud detection in financial transactions
- Anomaly detection in healthcare data
- Quality control in manufacturing
- Security monitoring
- Any scenario where you need to identify unusual patterns in subgroups of data

## How It Works

1. The detector scans through potential subgroups defined by combinations of feature values
2. For each subgroup, it calculates a likelihood ratio test statistic comparing:
   - H0: The subgroup follows the overall data distribution
   - H1: The subgroup follows a different distribution
3. The subgroup with the highest score is identified as the most anomalous
4. Statistical significance is assessed through permutation testing

## Documentation

For detailed documentation and examples, see the [examples](examples/) directory.

## License

MIT License

## Citation

If you use this software in your research, please cite:

```
@article{speakman2018subset,
  title={Subset Scanning Over Neural Network Activations},
  author={Speakman, Skyler and Sridharan, Srihari and Remy, Sekou and Weldemariam, Komminist and McFowland III, Edward},
  journal={arXiv preprint arXiv:1810.08676},
  year={2018}
}
``` 