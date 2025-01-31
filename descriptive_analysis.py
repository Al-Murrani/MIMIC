import pandas as pd

# Descriptive analysis: measure of frequency, central tendency, dispersion or variation and position
# for categorical data, use count and proportion

# https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.plot.html
# bar graph for categorical and count or sum
# line graph changing values over time
# scatter graph for numerical values
# df[["var1", "var2"]].hist()

# df.merge()
# df.concat()
# df.filter()
# df.isin()
# pd.merge_ordered() for filling NA values
# pd.merge_asof() left-join, for time series, fuzzy match of date
# df.query('')
# pd.melt()

# statistic can be descriptive or inferential
# data can be numerical (continues or discrete) or categorical (ordinal or nominal)
# df['col'].describe()
# 1. Center tendency of data includes mean, median, mode
# plot the distribution and if the data is left or right skewed then the median more suitable
# mode for categorical data
# 2. measure of spread included variance, standard variation, mean absolute deviation
# 3. quantiles (box plot), Inter quantile range(IQR)
# np.quantile
# 4. outliers (data < Q1 - 1.5 x IQR or data > Q3 + 1.5 x IQR)
# Finding outliers using IQR
# Outliers can have big effects on statistics like mean, as well as statistics that rely on the mean,
# such as variance and standard deviation. Interquartile range, or IQR, is another way of measuring spread
# that's less influenced by outliers. IQR is also often used to find outliers. If a value is less than Q1-(1.5xIQR)
# or greater than Q3+(1.5xIQR)
# , it's considered an outlier. In fact, this is how the lengths of the whiskers in a matplotlib box plot
# are calculated.
# series = df.groupby('col to group')['col to agg'].aggfunc()
# Compute the first and third quantiles and IQR of series
# q1 = np.quantile(series, 0.25)
# q3 = np.quantile(series, 0.75)
# iqr = q3 - q1
# # Calculate the lower and upper cutoffs for outliers
# lower = q1 - 1.5 * iqr
# upper = q3 + 1.5 * iqr
# Subset series to find outliers
# outliers = series.loc[(series > upper) | (series < lower)]

# 1. Discrete uniform distribution
# expected value can be calculated by multiplying each possible outcome with its corresponding probability and
# taking the sum
# Expected value = mean of distribution
# Discrete probability for discrete outcome probability = area, fair = discrete uniform distribution
# Create probability distribution
# size_dist = df.groupby('col group').agg('count') / df.shape[0]
# Reset index and rename columns
# size_dist = size_dist.reset_index()
# size_dist.columns = ['group_size', 'prob']
# Expected value
# expected_value = np.sum(size_dist['group_size'] * size_dist['prob'])
# Subset groups of required categories
# groups = size_dist[size_dist['group_size'] > required categories no]
# Sum the probabilities of groups
# prob = groups['prob'].sum()

# 2. Continuous uniform distribution
# outcome probability = area
# from scipy.stats import uniform.cdf(value, min, max)
# The uniform.rvs() function takes in the minimum and maximum of the distribution you're working with, followed by
# number of wait times you want to generate.

# 3. Binomial distribution, binary outcomes, trails MUST be independent
# n (total number of trials), p (probability of success)
# rvs - Generates random samples (simulated data) from a binomial distribution
# binom.rvs(number of trials, probability of success, size= number of random samples to generate)
# pmf - Gives the exact probability of observing exactly number of success in number of trials
# binom.pmf(equal to no of success, no trials, prob of success)
# cdf - Gives the cumulative probability of observing up to and including number of success in number of trials
# binom.cdf(fewer or equal to no of success, no trials, prob of success)
# expected value = n * p

# 4. Normal distribution
# standard normal distribution mean = 0 & std = 1, 68% area with 1std, 95% area with 2std, 99.7 area with 3std
# norm.cdf, norm.ppf
# central limit theorem (CLT), sample (random and independent) distribution becomes closer to norma as the number of
# trial increases no matter the original distribution being sampled from
# series.sample()

# np.random.seed(321)
# sample_means = []
# Loop 30 times to take 30 means
# for i in range(30):
# Take sample of size 20 from col of df with replacement
# cur_sample = df['col'].sample(sample_size, replace=True)
# Take mean of cur_sample
# cur_mean = cur_sample.mean()
# Append cur_mean to sample_means
# sample_means.append(cur_mean)
# Print mean of sample_means
# print((pd.Series(sample_means)).mean())
# # Print mean of col in df
# print(df['col'].mean())

# Poisson's distribution - discrete
# Events happen at a certain rate, but completely random
# Probability of some number of events (countable) occurring over a fixed period of time
# Described by lambda = average number of events per time interval (frequency rate or count), also expected
# value of distribution (peak at lambda value)
# Package to use scipy.stats
# pmf, cdf
# Distribution of sample means from Poisson distribution, looks normal with a large number of samples

# Exponential Distribution - continues
# Probability of time between Poisson events (interval between the event is consistent)
# lambda (rate)
# expected value is a measure of frequency in terms of time between events = 1/lambda
# expon

# Student t-test
# similar to normal distribution
# parameter degrees of freedom (df), effect thickness of tails
# lower df = thicker tails and higher std
# higher df = close to normal distribution

# Log-normal distribution
# variable whose logarithm is normally distributed, distribution skewed

# correlation coefficient, linear relationship, strength of relationship between two variables
# scatter plot seaborn package
# .corr
# pearson product moment correlation (r)
# sns.scatterplot without trend line
# sns.lmplot with trend line
# in order to use corr transform data, as variable have a linear relationship
# highly skewed data - log transformation
# square root - sqrt()
# reciprocal - 1/variable
# correlation does not mean causation
# confounding

# Create log_column for variables with non-linear relationship
# df['new_column'] = np.log(df['col_to_log'])
# Scatterplot
# sns.scatterplot(data=df, x='new_col', y='variable to correlate')
# plt.show()
# Calculate correlation
# cor = (df['new_column']).corr(df['variable to correlate'])

# Visualisation:
# import these libraries seaborn and matplotlib.pyplot
# choose appropriate plot type
# 1. two quantitative variable called relational plot
# sns scatter plot
# relplot (relational plot) (sub-plot in a single figure), use instead of scatterplot, kind= scatter or line
# hue (sub-group), hue_order(), palette, hex
# col or/and row for sub-plot categories to plot
# line plot, tacking the same thing typically over time, multiple observation per x value, errorbar instead of ci

# 2. categorical plot
# sns.countplot()
# catplot() categorical plot, kind= bar, count and order, whis, sym ""
# bar plot mean quantitative per category
# boxplot compare quantitative variable distribution against group
# point plot
# points show mean of quantitative variable for each category
# row or col to create separate sub-plot for each category
# instead of the mean use median, less suspendable to outlier
# similar to line plot, but point plot has categorical variable usually over x
# sns.catplot , estimator, linestyles, capsize

# sns.set_style (background)
# sns.set_palette, sequential palettes (main element colors)
# sns.set_context (scale)

# sns plots create two different types of objects, FacetGrid (relplot, catplot) and AxesSubplot (scatterplot, countplot)
# variable.fig.suptitle() - FacetGrid
# to add a title to sub-plot variable.set_titles() - AxesSubplot
# variable.set(xlabel="", ylabel="")

# EXPLORATION DATA ANALYSIS (EDA):
# To review and clean data
# Derive insights - descriptive analysis and correlation
# Generate hypotheses
# Prepare data for machine learning
# 1.
# pd.info() - index dtype, columns, non-null values and memory usage
# categorical data - value_counts()
# numerical data - .describe()
# visualising numerical data - sns.histplot() x=numerical data, with binwidth
# 2. Data Validation:
# change data type with df['col'].astype()
# to check data type .dtypes
# validating categorical value df['categorical col'].isin(['categories as series or df']) use ~ to invert the result
# to exclude data - df[~ df['categorical col'].isin(['categories as series or df'])]
# validating numerical data df.select_dtypes('number').head() , .min , .max
# 3. Visualization:
# the distribution numerical data - boxplot df (numerical x-axis),
# broken down by categories (y-axis) sns.boxplot
# sns.boxplot(data=, x='numerical variable', y='categorical variable to group by')
# plt.show()
# 4. Exploring Groups Of Data:
# .groupby() - groups data by category and used with aggregating functions (min, max, count, sum, var, std)
# to summarize grouped data
# .agg across rows for given column or df, apply to numerical columns,
# to apply more than one function df.agg(['mean', 'std'])
# df.agg({'col1':['agg fun 1', agg fun 2'], 'col2':['agg fun 3']})
# df.groupby('categorical col').agg(out_col1_name=('col agg', 'agg fuc), out_col2_name=('col agg', 'agg fuc))
# sns.barplot, vertical bar indicating the 95% confidence interval for the categorical mean.
# Since confidence intervals are calculated using both the number of values and the variability of those values,
# they give a helpful indication of how much data can be relied upon.
# 4. Missing Data
# missing_values_count = df.isna().sum()
# Strategies for dealing with missing values:
# Drop missing values (= or < 5%)
# threshold=len(df)*0.05 , cols_drop=df.columns[df.isna().sum()<=threshold, df.dropna(subset=cols_drop, inplace=True)
# Impute mean, median and mode (depends on context and distribution)
# cols_with_missing_values = df.columns[df.isna().sum() > 0]
# for col in cols_with_missing_values[:-1]: df[col].fillna(df[col].mode()[0])
# Impute by subtype for mean, median or mode, depending on boxplot distribution
# df_dict = df.groupby('col to group by')['col to agg'].median().to_dict()
# df['col to impute'] = df['col to impute'].fillna(df['col to group'].map(df_dict))
# 5. Converting & Analyzing Categorical Data:
# non_numeric = df.select_dtypes("type to select")
# for col in non_numeric.columns:
# print(f"Number of unique values in {col} column: ", non_numeric[col].nunique())
# Extracting value from categories, pandas.series.str.contains('condition'), | or strs, ^ start with, regexp
# Create a new [] with the req categories
# Variables containing the filter (condition)
# Create boolean lists with these conditions conditions = [(pandas.series.str.contains('condition'))]
# Create a categorical column
# df['new col'] = np.select(condlist_conditions, choicelist_req_categories, default='other')
# visualizing category frequency sns.countplot(data=data, x='')
# 6. Numerical Data
# pd.Series.str.replace('chr to remove', 'char to replace them with')
# .astype()
# adding summary statistic into dataframe:
# df['new_col'] = df.groupby('col to group by')['col to agg'].transform(lambda x: x.agg_fun())
# df[['col1 to select', 'col2 to select']].value_counts()
# 7. Outliers
# .describe()
# using inter quartile range (IQR) = 75th -25th percentile
# upper outlier > 75th percentile + (1.5 * IQR)
# lower outlier < 25th percentile + (1.5 * IQR)
# iqr = .quantile(0.75) - .quantile(0.25)
# Find the 75th and 25th percentiles
# seventy_fifth = df["col"].quantile(0.75)
# twenty_fifth = df["col"].quantile(0.25)
# Calculate iqr
# iqr = seventy_fifth - twenty_fifth
# Calculate the thresholds
# upper = seventy_fifth + (1.5 * iqr)
# lower = twenty_fifth - (1.5 * iqr)
# Subset the data
# df = df[(df["col"] > lower) & (df["col"] < upper)]
# print(df["col"].describe())
# 8. Date Data - pattern over time
# converting col to date type
# upon reading data - pd.read_csv('file.csv', parse_dates=['col'])
# pd.to_datetime()
# col time or hours object type
# Convert object to datetime then extract the required part such as hour or time
# pd.to_datetime(df['col'], format='%H%M').dt.hour/dt.time
# combining cols - pd.to_datetime(pd.to_datetime(pd[['col_month', 'col_day', 'col_year']])
# extract part of the date df['col'].dt.month/day/year
# visualizing pattern using lineplot(), agg y values at each value of x and show estimated mean and 95%
# confidence interval
# 9. Correlation - Numerical
# direction and strength
# pd.corr(), pearson correlation, linear relationship
# sns.heatmap(pd.corr(), annot=True)
# visualizing correlation is important as the correlation might not be liner and therefore pearson method is
# not applicable, compliment the correlation calculation with scatterplot, pairplot (all numerical in one visualization)
# 10. Categorical
# .value_counts()
# sns.histplot(data= , x='', hue= 'col to group by',binwidth=1)
# sns.kdeplot(data=, x='', hue='', cut=0) for visualizing distribution, especially when multiple distribution
# sns.scatterplot with hue as group by for categorical
# 11. Class Frequency (Categorical Data)
# Samples must be representative of the population
# classes imbalance
# .value_Counts(normalize=True for relative frequency)
# cross tabulation pd.crosstab(df['col for index'], df['col name'], values=df['col to agg'], aggfunc='agg func')
# 12. Generating New Feature
# Correlation sns.heatmap(df.corr(), annot=True)
# such as extracting time or the day from a datetime variable pd.series.dt.weekday
# group numerical data and group them as classes, using as an example descriptive statistics
# bins = [0, quantile 0.25, median, 0.75, max]
# create labels as a list of categories
# df['new_col'] = pd.cut(df['cont to cat col'], labels=cat_list, bins=bins)
# np.inf (infinity)
# sns.heatmap(df.corr(), annot=True)
# sns.countplot(data=df, x="col to count", hue="cat to group by")
# 13. Generating Hypothesis After Exploratory Data Analysis
# what is true
# hypothesis testing prior to collecting data
# generate hypothesis or question
# decision on what statistical test to use
# Summary: check for data type, aggregation and summary statistic, check missing values and identify strategies
# to deal with it, create categories from strings, use lambda to calculate summary statistic for categorical
# values and add them to df, deals with outliers, pattern over time, correlation between variables and examining
# distribution using kdeplot(), cross tabulation and generating new features using cut

# transform works on just one Series at a time and apply works on the entire DataFrame at once.

# SAMPLING
# A. without replacement
# population size = len(df)
# 1. simple random sampling (SRS)
# picking rows at random one at time, where rows has equal chance of being picked
# df/series.sample(n= , random_state=seed)
# population parameter is a calculation made on the population dataset
# point estimate or sample statistic is a calculation made on the sample dataset
# selection bias use histogram series.hist(bins=np.arange(star, end_exclusive, interval))
# pseudo-random number generation, seed =
# np.random, np.random.beta and others check the documentation
# 2. systematic random sampling - at regular interval
# sampling_interval = population size // sample_size
# row_to_select = df.iloc[::interval]
# df.plot(x= , y= , kind='scatter')
# use systematic sampling only if the scatter plotted data has no pattern or meaning behind the row order
# to make systematic sampling safe (shuffling will make it same as simple sampling)
# shuffled = df.sample(frac=1) frac - randomly shuffles the rows
# shuffled = shuffled.reset_index(drop=True).reset_index()
# shuffled.plot(x='index', y='col to sample', kind='scatter')
# 3. Stratified sampling
# split population into subgroup, random sampling of each subgroup
# to return a count for each subgroup series.value_counts()
# df[df.isin([selection_of_cols])]
# proportion of each category in the population the same as in sample
# df.groupby().sample(n= , random_state=)
# 4. weighted random sampling - use if the categories within a population is not evenly representative
# condition = df['col'] == 'value to be samples more frequently'
# df['weight'] = np.where(condition, int_condition_true, int_else)
# df.sample(frac= , weight='weight')
# cluster sampling
# random sampling to pick some subgroups
# random sample on only those subgroups
# col_values = list(df['col'].unique()
# sample = random.sample(col_values, k=no_of_subgroup)
# condition = df['col'].isin(sample)
# cluster = df[condition]
# df['col'].cat.remove_unused_categories()
# cluster.groupby('col').sample(n= , random_state)

# Sample Size and Point Of Estimate
# Relative error = 100 * abs(population_mean - sample_mean) / population_mean
# errors.plot(x='sample_size', y='relative_error', kind='line')
# Sample Distribution - distribution of replicates of point estimates
# mean_col_n = []
# for i in range(1000):
#     mean_col_n.append(df.sample(n=)['col to agg'].mean()
# )
# plt.hist(mean_col_n, bins=30)
# increasing no of replica did not have an effect on relative error of sample mean, more consistent distribution shape
# Approximate Sampling Distribution
# we don't have access to entire population, can't calculate sample distribution and therefore apprx sample distribution
# expand_grid
# Add a column of mean rolls and convert to a categorical
# np.random.choice(list(range(start,end_exclude)), size=, replace=True).mean(), run this a for loop
# standard error and central limit theorem
# mean of independent samples have apprx normal distribution
# mean of the samples mean - np.mean()
# std dev sample mean ddof=0 (population) and ddof=1 (sample)
# standard deviation of the sampling distribution is approximately equal to the population standard
# deviation divided by the square root of the sample size
# increase sample size the distribution of the mean closer to normal distribution
# and width of sample distribution becomes narrower
# standard error - std of the sampling distribution

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
# process (result bootstrap distribution):
# 1. resample of same size as original sample
# 2. calculate statistic of interest for bootstrap sample
# 3. repeat step 1 & 2 many times

# mean_col_no_iteration = []
# 3. repeat step 1 & 2 many times and append to list
# for i in range(no_iteration):
#     mean_col_no_iteration.append(
#         # 2. calculate point estimate
#         np.mean(
#             # 1. resample for bootstrap
#             df.sample(frac=1, replace=True)['col_to_agg']
#         )
#     )

# plt.hist(mean_col_no_iteration)
# Bootstrapping cannot correct biases from sampling
# estimated_standard_error = np.std(bootstrap_distn, ddof=1), is the standard deviation of the static of interest
# population_std = standard_error * np.sqrt(sample_size)
# not good for estimating population mean, good for estimating population standard deviation

# Calculate the population std dev popularity
# pop_sd = df['col'].std(ddof=0)
# Calculate the original sample std dev popularity
# samp_sd = df['col'].std()
# Calculate the sampling dist'n estimate of std dev popularity
# samp_distn_sd = np.std(sampling_distribution, ddof=1) * np.sqrt(5000)
# Calculate the bootstrap dist'n estimate of std dev popularity
# boot_distn_sd = np.std(bootstrap_distribution, ddof=1) * np.sqrt(5000)
# Print the standard deviations
# print([pop_sd, samp_sd, samp_distn_sd, boot_distn_sd])

# Confidence Intervals
# account for uncertainty in our estimate of a population parameter by providing a range of possible values.
# We are confident that the true value lies somewhere in the interval specified by that range.
# confidence interval typically will not include some values that the sample could take;
# only those close to the statistic of interest
# sampling vs bootstrap distribution
# symmetrical, point of estimate -/+ confidence intervals
# to include 95%
# Confidence interval:
# 1. Inverse cumulative distribution
# Bell curve
# integrate to get area under the bell curve
# inverse the above, flip x and y axes
# norm.ppf(quantile, loc=0, scale=1)
# 2. standard error method
# assumes bootstrap distribution is normal. if sample size and number of replicates are sufficiently large
# point_estimate = np.mean(boot_distn)
# std_error = np.std(boot_distn, ddof=1)
# lower = norm.ppf(0.025, loc=point_estimate, scale=std_error)
# upper = norm.ppf(0.975, loc=point_estimate, scale=std_error)
# bootstrap distribution is not perfectly normal

# Generate a 95% confidence interval using the quantile method on the bootstrap distribution, setting the 0.025
# quantile as lower_quant and the 0.975 quantile as upper_quant
# q1 = np.quantile(bootstrap_distribution, 0.025)
# q3 = np.quantile(bootstrap_distribution, 0.975)
# Generate a 95% confidence interval using the standard error method from the bootstrap distribution.
# Calculate point_estimate as the mean of bootstrap_distribution, standard_error as the std of bootstrap_distribution.
# Calculate lower_se as the 0.025 quantile of an inv. CDF from a normal distribution with mean point_estimate
# and standard deviation standard_error.
# Calculate upper_se as the 0.975 quantile of that same inv. CDF.

# standard deviation of a bootstrap statistic is a good approximation of standard error of sampling distribution
# standard error method (mean & std boot_distn) vs quantile (0.025/0.975), similar result, means normal
# distribution is a good approximate for bootstrap distribution

# HYPOTHESIS TESTING
# Steps:
# identify population parameter that is hypothesized about
# specify null and alternative hypothesis
# determine test statistic and corresponding null distribution
# conduct hypothesis test in python
# measure evidence against null hypothesis and compare to significance level
# interpret the result in the context of the original problem

# 1. z-score standardized test statistic - single variable
# variables has arbitrary unit, we need to standardize the value
# z-score = sample stat  - hypoth.param.value / standard error
# Hypothesis testing - determine whether sample statistics are close to or far away from expected
# (or hypothesised values)
# standard normal (z) distribution mean=0 + std=1
# p-value
# hypothesis - statement about unknown population parameter (not knowing the true value, make inferring
# from the data)
# compare two competing hypothesis (null hypothesis & alternative hypothesis (challenges what is known))
# initially null is true
# rejecting null hypothesis only if evidence from sample is significant
# hypothesis tests check if sample statics lie in the tails of the null distribution:
# alternative different from null (two-tailed)
# alternative greater than null (right-tailed) norm.cdf(z_score, loc=0, scale=1)
# alternative less than null (left-tailed) 1 - norm.cdf(z_score, loc=0, scale=1)
# p-value (quantify evidence) probability of obtaining a result assuming the null hypothesis is true
# calculate
# set alpha (significant level)
# proportion_sample_mean = (df.['col related to hypothesis'] == 'condition to test').mean()
# prop_hyp_mean from studies or literature
# std_error = np.std(boot_dist, ddof=1)
# z_score = (proportion_sample_mean - prop_hyp_mean) / std_error
# p-value = 1 - norm.cdf(z_score, loc=0, scale=1)
# p-value <= alpha then reject the null hypothesis, strong evidence for the alternative hypothesis
# large p-value - fail to reject the null hypothesis
# small p-value - reject null hypothesis
# significant level - cut off point between small and large
# if p <= significant level then reject null hypothesis, else fail to reject the null hypothesis
# decide significant level prior to running the test
# confidence interval of alpha 95% - np.quantile(0.025 or 0.975)
# type of error
# support alternative hypothesis when null was correct - false positive (type 1)
# support null hypothesis when alternative was correct - false negative error (type 2)

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
# steps:
# df.groupby('col')['col to agg'].mean()
# if the means of the two group different then ask
# Question is the difference statistically significant or can it be explained by sample variability
# population mean unkown, can be estimated from sample mean
# sample_mean_gr1 - sample_mean_gr2 = test statistic
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

# t distribution has a degree of freedom df, larger df the more closely to normal distribution
# degree of freedom - maximum number of logically independent value in data sample
# df = n_grp1 + n_grp2 - 2
# calculating p-value, t_stat, degree of freedom
# 1 - t.cdf(t_stat, df=degree_of_freedom)
# When the standard error is estimated from sample standard deviation and sample size,
# test statistic is transformed into a p-value using the t-distribution

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

