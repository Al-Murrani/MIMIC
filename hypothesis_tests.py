import numpy as np
from scipy.stats import norm, t


# Steps:
# identify population parameter that is hypothesized about
# specify null and alternative hypothesis
# determine test statistic and corresponding null distribution
# conduct hypothesis test in python
# measure evidence against null hypothesis and compare to significance level
# interpret the result in the context of the original problem


def calculate_standard_error(boot_dist):
    """
    Calculate the standard error from the bootstrap distribution.
    """
    return np.std(boot_dist, ddof=1)


# 1. Single Variable
# z-score standardized test statistic
# As variables has arbitrary unit, we need to standardize the value
# standard normal (z) distribution mean=0 + std=1
def calculate_z_score(proportion_sample_mean, prop_hyp_mean, std_error):
    """
    Compute the z-score for hypothesis testing.
    """
    return (proportion_sample_mean - prop_hyp_mean) / std_error


# p-value (quantify evidence) probability of obtaining a result assuming the null hypothesis is true
# p-value <= alpha then reject the null hypothesis, strong evidence for the alternative hypothesis
# large p-value - fail to reject the null hypothesis
# small p-value - reject null hypothesis
# significant level - cut off point between small and large
# if p <= significant level then reject null hypothesis, else fail to reject the null hypothesis
# decide significant level prior to running the test
# confidence interval of alpha 95% - np.quantile(0.025 or 0.975)
# type of error:
# support alternative hypothesis when null was correct - false positive (type 1)
# support null hypothesis when alternative was correct - false negative error (type 2)
def calculate_p_value(z_score, alternative='two-tailed'):
    """
    Compute the p-value based on the z-score and test type.
    - 'two-tailed': Alternative hypothesis is different from null.
    - 'right-tailed': Alternative hypothesis is greater than null.
    - 'left-tailed': Alternative hypothesis is less than null.
    """
    if alternative == 'two-tailed':
        return 2 * (1 - norm.cdf(abs(z_score), loc=0, scale=1))
    elif alternative == 'right-tailed':
        return 1 - norm.cdf(z_score, loc=0, scale=1)
    elif alternative == 'left-tailed':
        return norm.cdf(z_score, loc=0, scale=1)
    else:
        raise ValueError("Invalid alternative hypothesis type")


# Hypothesis testing - determine whether sample statistics are close to or far away from expected(or hypothesised
# values) hypothesis - statement about unknown population parameter (not knowing the true value, make inferring from
# the data) compare two competing hypothesis (null hypothesis & alternative hypothesis (challenges what is known))
# initially null is true rejecting null hypothesis only if evidence from sample is significant hypothesis tests check
# if sample statics lie in the tails of the null distribution
def hypothesis_test(df, col, condition, prop_hyp_mean, alpha=0.05, alternative='two-tailed'):
    """
    Perform a hypothesis test on the given dataset.
    - df: DataFrame containing the data.
    - col: Column related to hypothesis.
    - condition: Condition to test.
    - prop_hyp_mean: Hypothesized population proportion.
    - alpha: Significance level.
    - alternative: Type of alternative hypothesis ('two-tailed', 'right-tailed', 'left-tailed').
    """
    proportion_sample_mean = (df[col] == condition).mean()
    boot_dist = np.random.choice([0, 1], size=1000, p=[1 - proportion_sample_mean, proportion_sample_mean])
    std_error = calculate_standard_error(boot_dist)
    z_score = calculate_z_score(proportion_sample_mean, prop_hyp_mean, std_error)
    p_value = calculate_p_value(z_score, alternative)

    result = "Reject null hypothesis" if p_value <= alpha else "Fail to reject null hypothesis"
    return {
        'z_score': z_score,
        'p_value': p_value,
        'result': result
    }


# 2. Two sample problem
# Using sample standard deviation to estimate the standard error is computationally easier than using bootstrapping.
# to correct for the approximation, you need to use a t-distribution when transforming the test statistic to
# get the p-value.
# compare sample statistics (such as mean) across groups of a variable
# numerical variable vs categorical
# null hypothesis is that mean is the same in both group
# alternative is that mean greater in group1 when compare to group2
# null hypothesis: mean_group1 - mean_group2 = 0
# alternative hypothesis: mean_group1 - mean_group2 > 0
def calculate_group_stats(df, group_col, agg_col):
    """
    Calculate mean, standard deviation, and sample size for each group.
    """
    group_means = df.groupby(group_col)[agg_col].mean()
    group_stds = df.groupby(group_col)[agg_col].std()
    group_sizes = df.groupby(group_col)[agg_col].count()

    return group_means, group_stds, group_sizes


# A. t-test (independent variable)
# t = (sample_mean_gr1 - sample_mean_gr2) - (pop_mean_gr1 - pop_mean_gr2)/std_error(sample_mean_gr1 - sample_mean_gr2)
# although for std_error can use bootstrap, there is an easier way
# assumption pop_mean_gr1 - pop_mean_gr2 = 0
# std_error = square_root((std_var1/sample_size) + (std_var2/sample_size))

# t = (sample_mean_gr1 - sample_mean_gr2)/square_root((sqr(std_var1)/sample_size) + (sqr(std_var2)/sample_size))
# to calculate t:
# mean for each group
# df.groupby('col')['col to agg'].mean()
# std for each group
# df.groupby('col')['col to agg'].std()
# sample size
# df.groupby('col')['col to agg'].count()
# numerator = grp1_mean - grp2_mean
# denominator = np.sqrt((grp1_std ** 2 / sample size grp1) + (grp2_std ** 2 / sample size grp2))
# t_stat = numerator/denominator

# calculating degree of freedom, p-value, t_stat,
# t distribution has a degree of freedom df, larger df the more closely to normal distribution
# degree of freedom - maximum number of logically independent value in data sample
# df = n_grp1 + n_grp2 - 2
def calculate_degrees_of_freedom(n1, n2):
    """
    Compute degrees of freedom for an independent t-test.
    """
    return n1 + n2 - 2


# 1 - t.cdf(t_stat, df=degree_of_freedom)
# When the standard error is estimated from sample standard deviation and sample size,
# test statistic is transformed into a p-value using the t-distribution
def calculate_cumulative_distribution_p_value(t_stat, df):
    """
    Compute the p-value from the t-distribution.
    """
    p_value = 1 - t.cdf(t_stat, df=df)
    return p_value


def calculate_t_statistic(mean1, mean2, std1, std2, n1, n2):
    """
    Compute the t-statistic for an independent two-sample t-test.
    """
    standard_error = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
    t_stat = (mean1 - mean2) / standard_error
    return t_stat


def perform_t_test(df, group_col, agg_col, group1, group2):
    """
    Perform an independent two-sample t-test given a DataFrame.
    """
    group_means, group_stds, group_sizes = calculate_group_stats(df, group_col, agg_col)

    mean1, mean2 = group_means[group1], group_means[group2]
    std1, std2 = group_stds[group1], group_stds[group2]
    n1, n2 = group_sizes[group1], group_sizes[group2]

    t_stat = calculate_t_statistic(mean1, mean2, std1, std2, n1, n2)
    df = calculate_degrees_of_freedom(n1, n2)
    p_value = calculate_cumulative_distribution_p_value(t_stat, df)

    return {'t_statistic': t_stat, 'degrees_of_freedom': df, 'p_value': p_value}

# B. Paired (not independent) t-tests
# an examples measuring a same variable on different years
# sample_df['diff'] = sample_df['dependent_var1'] - sample_df['dependent_var2']
# sample_df['diff'].hist(bins=)
# sample_df['diff'].mean()
# null hypothesis: mean diff = 0
# alternative hypothesis: mean diff < 0
# t = mean_diff - population_diff / sqrt(std_diff ** 2/n_diff)
# degree_freedom = n_diff - 1
# n_diff = len(sample_df)
# std_diff = sample_df['diff'].std()
# t_stat = (mean_diff - 0) / np.sqrt(std_diff**2)/n_diff
# degree_freedom = n_diff - 1
# p_value = t.cdf(t_stat, degree_freedom)
# alternatively use pingouin package
# pingouin.ttest(x=sample_df['dependent_var1'],
#                y=sample_df['dependent_var2'],
#                paired=True,
#                alternative='less')

# 3. ANOVA - more than two categories for a variable
# sns.boxplot(x= numerical_variable, y= categorical_variable, data=)
# differences between the groups
# pingouin.anova(data= , dv=numerical_variable, between=categorical_variable)
# at least two categories have significantly different compensation
# if there is significant differences, anova test will not tell which pair then you have to preform pairwise_tests()
# pingouin.pairwise(data= , dv= , between= ,padjust='bonf'), p-corr is the corrected p-value as we run more
# comparison the probability of false positive (support alternative hypothesis when null was correct) increases

# 4. One sample proportion test
# While bootstrapping can be used to estimate the standard error of any statistic, it is computationally intensive.
# For proportions, using a simple equation of the hypothesized proportion and sample size is easier to compute.
# z = sample statistic - P0 (unknown population parameter)/ SE(sample statistic)
# SE sample statistic = sqrt(P0 * (1-P0)/n)
# df['new_col'].value_counts(normalized=True)
# sample statistic = (df['col'] == 'condition').mean()
# p_0 = 0.5 according to null hypothesis
# n = len(df)
# z_score = (sample statistic - p_0) / (np.sqrt(p_0 * (1- p_0) / n)
# left-tailed p_value = norm.cdf(z_score)
# two-tailed p_value = norm.cdf(-z-score) + 1 - norm.cdf(z_score)
# p_value = 2 * (1 - norm.cdf(z_score))

# 5. Two sample proportion tests
# df.groupby('col')['col to agg'].value_counts()
# import statsmodels.stats.proportion import proportions_ztest
# z_score, p_value = proportions_ztest(count=n_hobbyists, nobs=n_rows, alternative='two-sided')
# two-sided as testing for difference

# Chi-square test of independent
# to compare proportions of successes in a categorical variable across groups of another categorical variable
# extend proportion tests to more than two groups
# statistical independent - proportion of successes in the response variable is the same across all categories
# of explanatory variable
# props = df.groupby('col')['col to agg'].value_counts(normalize=True)
# wide_props = props.unstack()
# wide_props.plot(kind='bar', stacked=True)
# if the above bars are roughly the same then use that test
# expected, observed, stats = pingouin.chi2_independence(data=df, x='categorical variable', y='categorical variable')
# degree of freedom = (no of response categories-1) x (no of explanatory categories-1)
# no alternative argument
# observed and expected counts squared must be non-negative
# chi-square tests are almost always right-tailed
# chi-square distribution has degrees of freedom and non-centrality parameters.
# When these numbers are large, the chi-square distribution can be approximated by a normal distribution.

# chi-square goodness of fit tests
# single categorical variable to hypothesised distribution
# col_counts = df['col'].value_counts()
# col_counts = df.rename_axis('col').reset_index(name='n').sort_values('col')
# how far the observed sample distribution from hypothesised distribution
# n_total = len(df)
# hypothesized['n'] = hypothesized['prop'] * n_total
# plt.bar(col_counts['col'], col_counts['n'], color='red', label='')
# plt.bar(hypothesized['col'], hypothesized['n'], alpha=0.5, color='blue', label='')
# plt.legend()
# chisquare(f_obs=col_counts['n'], f_exp=hypothesized['n'])

# assumption in hypothesis testing:
# randomly sourced from its population
# observations are independent (before data collection)
# sample are large enough that the central limit theorem applies
# (small sample result in sample distribution is not normally distributed)
# there are standard sample size for different tests

# Assumption are not meet - parametric test
# Willcoxon-signed rank test
# Willcoxon-Mann-Whitney
# Kruskal-Wallis
