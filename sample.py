import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats


# A. without replacement
# population size = len(df)
def simple_random_sampling(df, n, seed=None):
    """
    Performs Simple Random Sampling (SRS) by selecting n random rows.
    Ensures each row has an equal chance of being picked.
    """
    return df.sample(n=n, random_state=seed)


def systematic_random_sampling(df, sample_size):
    """
    Performs Systematic Random Sampling by selecting rows at a regular interval.
    Only use if there is no pattern in the data ordering.
    """
    interval = len(df) // sample_size
    return df.iloc[::interval]


def stratified_sampling(df, group_col, n, seed=None):
    """
    Performs Stratified Sampling by splitting the population into subgroups
    and performing random sampling within each subgroup while maintaining proportions.
    """
    return df.groupby(group_col).sample(n=n, random_state=seed)


def weighted_random_sampling(df, column, value, int_condition_true, int_else, frac, seed=None):
    """
    Performs Weighted Random Sampling when categories in the population are not evenly representative.
    """
    df['weight'] = np.where(df[column] == value, int_condition_true, int_else)
    return df.sample(frac=frac, weights='weight', random_state=seed)


def cluster_sampling(df, column, k, n, seed=None):
    """
    Performs Cluster Sampling by selecting some subgroups randomly and sampling only from them.
    """
    col_values = list(df[column].unique())
    sampled_clusters = random.sample(col_values, k)
    clustered_df = df[df[column].isin(sampled_clusters)]
    return clustered_df.groupby(column).sample(n=n, random_state=seed)


def calculate_relative_error(population_mean, sample_mean):
    """
    Calculates relative error between population mean and sample mean.
    """
    return 100 * abs(population_mean - sample_mean) / population_mean


def plot_relative_error(errors_df):
    """
    Plots relative error against sample size to visualize sampling error.
    """
    errors_df.plot(x='sample_size', y='relative_error', kind='line')
    plt.show()


def sample_distribution(df, column, n, iterations=1000):
    """
    Simulates a Sample Distribution by taking multiple sample means.
    """
    mean_values = [df.sample(n=n)[column].mean() for _ in range(iterations)]
    plt.hist(mean_values, bins=30)
    plt.show()


def approximate_sampling_distribution(start, end_exclude, size, iterations=1000):
    """
    Approximates a Sampling Distribution using random choice method.
    """
    mean_values = [np.random.choice(list(range(start, end_exclude)), size=size, replace=True).mean() for _ in
                   range(iterations)]
    plt.hist(mean_values, bins=30)
    plt.show()


def calculate_standard_error(sample_means):
    """
    Calculates the Standard Error, which decreases as sample size increases.
    """
    return np.std(sample_means, ddof=1)


# Significance of Gaussian (normal) distribution in statistics, through the lens of the Central Limit Theorem (CLT),
# demonstrating that means of independent samples from any distribution will approximate a normal distribution
# as the sample size increases, showing how larger sample sizes lead to more accurate estimates of the population
# parameter and narrower sampling distributions.
# The relationship between sample size and the accuracy of estimates: Larger sample sizes yield narrower
# distributions and more accurate estimates of population parameters.
# The impact of sample size on the shape of the distribution: As sample size increases, the distribution of sample
# means becomes more symmetric and closely approximates a normal distribution.
# The concept of standard error: The standard deviation of the sampling distribution, known as the standard error,
# decreases as the sample size increases. This is crucial for estimating the variability of sample means and
# understanding the precision of your estimates.
# calculate the standard deviation of sampling distribution using NumPy, specific focus on the difference between
# population and sample standard deviation calculations. For example, when using pandas
# np.std() population standard deviation set ddof=0, sample standard deviation np.std() ddof=1

# B. Bootstrapping with replacement (resampling)
# Sampling: treat dataset as the population and move to a smaller sample
# Bootstrapping: treat dataset as sample and build up theoretical population from sample
# understand sampling variability with single sample, unable to sample population multiple times for sample distribution
# resampling from our observed data to approximate the sampling distribution of a statistic,
# like the mean or median. Helpful if the resulting bootstrap distribution approximates a normal distribution
# because this allows us to make statistical inferences, such as calculating confidence intervals,
# more easily and accurately.
def bootstrap_resampling(df, col_to_agg, no_iteration=1000):
    """
    Perform bootstrapping with replacement to approximate the sampling distribution of
    a statistic.

    Parameters:
    df (pd.DataFrame): The dataset containing the column to aggregate.
    col_to_agg (str): The column name for which to compute the bootstrap mean.
    no_iteration (int): The number of bootstrap samples to generate.

    Returns:
    list: A list containing the bootstrapped means.
    """
    mean_bootstrap_samples = []  # Store computed means for each bootstrap sample

    for _ in range(no_iteration):
        mean_bootstrap_samples.append(
            np.mean(
                df.sample(frac=1, replace=True)[col_to_agg]  # Resample with replacement and compute mean
            )
        )

    return mean_bootstrap_samples


# Confidence Intervals
# account for uncertainty in our estimate of a population parameter by providing a range of possible values.
# We are confident that the true value lies somewhere in the interval specified by that range.
# confidence interval typically will not include some values that the sample could take;
# only those close to the statistic of interest
# Inverse Cumulative Distribution (Quantile Method)
def inverse_cumulative_confidence_interval(sample, confidence_level=0.95):
    """
    Calculate the confidence interval using the inverse cumulative distribution method.

    Parameters:
    sample: np.array, the sample data to calculate the confidence interval.
    confidence_level: float, confidence level (e.g., 0.95 for 95% confidence)

    Returns:
    tuple: (lower_bound, upper_bound)
    """
    # Step 1: Calculate point estimate (mean of the sample)
    point_estimate = np.mean(sample)

    # Step 2: Calculate standard error (standard deviation of the sample divided by sqrt(sample size))
    std_error = np.std(sample, ddof=1) / np.sqrt(len(sample))

    # Step 3: Calculate the critical value for the given confidence level
    # The quantiles are determined based on the confidence level.
    # For example, for a 95% confidence level, 2.5% of the distribution will be in each tail.
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - lower_quantile

    # Calculate the lower and upper bounds using the norm.ppf function
    lower = stats.norm.ppf(lower_quantile, loc=point_estimate, scale=std_error)
    upper = stats.norm.ppf(upper_quantile, loc=point_estimate, scale=std_error)

    return lower, upper


# Standard Error Method (assuming bootstrap distribution is approximately normal)
def standard_error_confidence_interval(boot_distn, confidence_level=0.95):
    """
    Calculate the confidence interval using the standard error method.

    Parameters:
    boot_distn: np.array, the bootstrap distribution.
    confidence_level: float, confidence level (e.g., 0.95 for 95% confidence)

    Returns:
    tuple: (lower_bound, upper_bound)
    """
    # Step 1: Point estimate is the mean of the bootstrap distribution
    point_estimate = np.mean(boot_distn)

    # Step 2: Standard error is the standard deviation of the bootstrap distribution
    std_error = np.std(boot_distn, ddof=1)

    # Step 3: Calculate the critical value for the given confidence level
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - lower_quantile

    # Calculate the lower and upper bounds using the norm.ppf function
    lower = stats.norm.ppf(lower_quantile, loc=point_estimate, scale=std_error)
    upper = stats.norm.ppf(upper_quantile, loc=point_estimate, scale=std_error)

    return lower, upper
