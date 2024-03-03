import numpy as np
from scipy.stats import norm

def gaussian_scan_statistic(subset_data: np.ndarray, null_mean: float, null_std: float) -> float:
    """
    Calculates the Gaussian scan statistic (likelihood ratio test statistic) for a given subset.
    Assumes data follows a Gaussian distribution.

    H0 (Null Hypothesis): The data in the subset comes from the overall null distribution.
    H1 (Alternative Hypothesis): The data in the subset comes from a different Gaussian distribution.

    Args:
        subset_data (np.ndarray): Numerical data points belonging to the current subset.
        null_mean (float): Mean of the data under the null hypothesis (overall data mean).
        null_std (float): Standard deviation of the data under the null hypothesis (overall data std).

    Returns:
        float: The log-likelihood ratio test statistic. A higher value indicates
               that the subset is more anomalous compared to the null distribution.
               Returns -np.inf if subset_data is empty or too small for calculation.

    Note:
        This function is great for spotting weird stuff in continuous data.
        Just be careful with tiny subsets—they might not give you reliable results.
    """
    n_subset = len(subset_data)
    if n_subset == 0:
        return -np.inf  # No data, no score

    subset_mean = np.mean(subset_data)
    # Using a pooled standard deviation or a small constant to avoid zero std if subset is too uniform
    # For simplicity, we'll use null_std directly for the alternative hypothesis if subset_std is zero.
    subset_std = np.std(subset_data) if np.std(subset_data) > 1e-9 else null_std  # Add a small epsilon or use null_std

    if subset_std == 0 and null_std == 0:  # All values are identical
        return 0.0  # No deviation, not anomalous

    # Calculate log-likelihood under H1 (subset distribution)
    # Using subset's own estimated mean and std
    log_likelihood_h1 = np.sum(norm.logpdf(subset_data, loc=subset_mean, scale=subset_std))

    # Calculate log-likelihood under H0 (null distribution)
    log_likelihood_h0 = np.sum(norm.logpdf(subset_data, loc=null_mean, scale=null_std))

    # Return the log likelihood ratio
    return log_likelihood_h1 - log_likelihood_h0

# You could add other scan statistics here, e.g., for Bernoulli data:
def bernoulli_scan_statistic(subset_counts: int, subset_n: int, null_proportion: float) -> float:
    """
    Calculates the Bernoulli scan statistic for a given subset.
    Assumes data follows a Bernoulli distribution (binary outcomes).

    Args:
        subset_counts (int): Number of 'positive' events in the subset.
        subset_n (int): Total number of events in the subset.
        null_proportion (float): Proportion of 'positive' events under the null hypothesis.

    Returns:
        float: The log-likelihood ratio test statistic.
               Returns -np.inf if subset_n is 0 or subset_counts is invalid.

    Note:
        This function is perfect for binary data, like checking for fraud or weird patterns in categories.
        Just make sure your subset is big enough to trust the results!
    """
    if subset_n == 0:
        return -np.inf  # No data, no score
    if not (0 <= subset_counts <= subset_n):
        raise ValueError("subset_counts must be between 0 and subset_n")

    # Proportion in the subset
    p_subset = subset_counts / subset_n

    # Avoid log(0) issues for probabilities
    p_subset_clamped = np.clip(p_subset, 1e-9, 1 - 1e-9)
    null_proportion_clamped = np.clip(null_proportion, 1e-9, 1 - 1e-9)

    # Log-likelihood under H1 (subset distribution)
    ll_h1 = subset_counts * np.log(p_subset_clamped) + \
            (subset_n - subset_counts) * np.log(1 - p_subset_clamped)

    # Log-likelihood under H0 (null distribution)
    ll_h0 = subset_counts * np.log(null_proportion_clamped) + \
            (subset_n - subset_counts) * np.log(1 - null_proportion_clamped)

    return ll_h1 - ll_h0 