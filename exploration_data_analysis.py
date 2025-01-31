import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)


# Each data type requires different EDA methods to extract meaningful insights, leading to a well-rounded analysis.
# Summary Table
# | Data Type     | Key Techniques                                                                      |
# |---------------|-------------------------------------------------------------------------------------|
# | Numerical     | Descriptive stats, histograms, box plots, scatter plots, outlier detection          |
# | Categorical   | Frequency counts, bar plots, pie charts, cross-tabulation                           |
# | Temporal      | Time series plots, seasonal analysis, trend analysis, rolling statistics            |
# | Text          | Word clouds, frequency distribution, n-grams, sentiment analysis, topic modeling    |
# | Mixed         | Correlation analysis, multi-variable plots, encoding techniques                     |


def read_files_to_dataframe(path_folder, file_format):
    # Define the folder containing the CSV files
    folder_path = path_folder

    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter to get only CSV files
    csv_files = [f for f in all_files if f.endswith(file_format)]

    # Initialize an empty list to hold DataFrames
    df_list = []

    # Loop through the list of CSV files and read each into a DataFrame
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df


# To check for data type
# dataframe summary including data type df.info()
# subset of df columns based on specified column type df.select_dtype(incl_column_type)
# cast pandas object to specified data type .astype()

def data_type_conversion(df, columns_list, target_type):
    for column in columns_list:
        if target_type == 'datetime':
            df[column] = pd.to_datetime(df[column])
        elif target_type == 'string':
            df[column] = df[column].astype('string')
        elif target_type == 'category':
            df[column] = df[column].astype('category')
    return df.dtypes


def filter_categorical_data(df, col_name, categories, exclude=False):
    """
    Filters the DataFrame based on categories for a categorical column.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        col_name (str): The column name to filter.
        categories (list): The list of categories to filter by.
        exclude (bool, optional): Whether to exclude the categories. Defaults to False.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")

    col_dtype = df[col_name].dtypes

    if col_dtype not in ['category', 'object']:
        raise ValueError(f"Column '{col_name}' must be categorical for filtering.")

    if categories is None:
        raise ValueError("Categories must be provided for categorical column filtering.")

    if exclude:
        return df[~df[col_name].isin(categories)]
    else:
        return df[df[col_name].isin(categories)]


def summarize_categorical_counts(df):
    categorical_counts = {}

    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        # Get value counts for the column
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']

        # Store the value counts DataFrame in the dictionary
        categorical_counts[col] = value_counts

    return categorical_counts


def aggregate_data(df, group_by=None, agg_columns=None, agg_funcs=None, rename_aggs=None, transform_col=None,
                   new_col_name=None):
    """
    Aggregate data for categorical and numerical columns with flexible options.
    Groups data by category with aggregating functions (min, max, count, sum, var, std)
    to summarize grouped data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    group_by (str or list of str): Column(s) to group by. Default is None.
    agg_columns (list of str): Columns to aggregate. Default is None.
    agg_funcs (dict or list): Aggregation functions. For `.agg`, can be a list or dict.
    example: df.agg(['mean', 'std'])
             df.agg({'col1':['agg fun 1', 'agg fun 2'], 'col2':['agg fun 3']})
             df.groupby('categorical col').agg(out_col1_name=('col agg', 'agg fuc),
             out_col2_name=('col agg', 'agg fuc'))
    rename_aggs (dict): Rename aggregation columns as a dictionary mapping. Default is None.
    transform_col (str): Column to transform within groups. Default is None.
    new_col_name (str): New column name for transformed data. Default is None.

    Returns:
    pd.DataFrame: The resulting DataFrame after applying aggregation.
    """
    if group_by and agg_columns and agg_funcs:
        if isinstance(agg_funcs, list):
            # Standard groupby aggregation with multiple functions
            return df.groupby(group_by)[agg_columns].agg(agg_funcs)
        elif isinstance(agg_funcs, dict):
            # Named aggregation with .agg
            return df.groupby(group_by).agg(**rename_aggs)
    elif group_by and transform_col and new_col_name:
        # Apply transformation and add a new column
        df[new_col_name] = df.groupby(group_by)[transform_col].transform(lambda x: x.agg(agg_funcs))
        return df
    elif not group_by and agg_funcs:
        # Aggregation for numerical data without grouping
        # .agg across rows for given column or df
        return df.agg(agg_funcs)
    else:
        raise ValueError("Invalid parameters: ensure group_by, agg_columns, and agg_funcs are provided as needed.")


def check_missing_values(df):
    missing_values_count = df.isna().sum()
    return missing_values_count


def remove_missing_values(df):
    # Drop missing values (= or < 5%)
    threshold = len(df) * 0.05
    cols_drop = df.columns[df.isna().sum() <= threshold]
    df.dropna(subset=cols_drop, inplace=True)


def impute_missing_values(df):
    # Impute mean, median and mode (depends on context and distribution)
    cols_with_missing_values = df.columns[df.isna().sum() > 0]
    for col in cols_with_missing_values[:-1]:
        df[col].fillna(df[col].mode()[0])


def impute_missing_values_subtype(df, group_by_col, agg_col, col_to_impute):
    # Impute by subtype for mean, median or mode, depending on boxplot distribution
    df_dict = df.groupby(group_by_col)[agg_col].median().to_dict()
    df[col_to_impute] = df[col_to_impute].fillna(df[group_by_col].map(df_dict))


def create_categories_from_string(df, col, patterns, categories, new_col_name, default_category='other'):
    """
    Create a categorical column in a pandas DataFrame based on string patterns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col: string column
        patterns (list of str): List of string patterns to filter.
        categories (list of str): List of categories corresponding to the patterns.
        new_col_name (str): Name of the new column to create.
        default_category (str): Default category if no pattern matches.

    Returns:
        pd.DataFrame: DataFrame with the new categorical column added.
    """
    if len(patterns) != len(categories):
        raise ValueError("The length of patterns and categories must be the same.")

    # Create conditions based on patterns
    # Possibly a for loop for the len(pattern)
    conditions = [df[col].str.contains(pattern, na=False) for pattern in patterns]

    # Use np.select to create the categorical column
    df[new_col_name] = np.select(conditions, categories, default=default_category)

    return df


def outlier_iqr(dataframe, column, filter_outliers):
    series = dataframe[column]
    first_quantiles = np.quantile(series, 0.25)
    third_quantiles = np.quantile(series, 0.75)
    # iqr = q3 - q1
    inter_quartile_range = third_quantiles - first_quantiles
    # # Calculate the lower and upper cutoffs for outliers
    # lower = q1 - (1.5 * iqr)
    lower_cutoffs = first_quantiles - (1.5 * inter_quartile_range)
    # upper = q3 + (1.5 * iqr)
    upper_cutoffs = third_quantiles + (1.5 * inter_quartile_range)

    if filter_outliers:
        # Filter the DataFrame to exclude outliers
        df_wo_outliers = dataframe[(series >= lower_cutoffs)
                                   & (series <= upper_cutoffs)]
        return df_wo_outliers
    else:
        # Subset series to find outlier
        # outliers = series.loc[(series > upper) | (series < lower)]
        outliers = series.loc[
            (series > upper_cutoffs) |
            (series < lower_cutoffs)]
        return outliers


# plot function like a flow diagram

def plot_data(df, x, y=None, data_type=None, **kwargs):
    """
    Plots data based on the provided type using Seaborn.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The column to plot on the x-axis.
        y (str, optional): The column to plot on the y-axis. Defaults to None.
        data_type (str): Type of the data ('numerical', 'datetime', 'categorical', 'correlation'). Defaults to None.
        **kwargs: Additional keyword arguments for the Seaborn plotting functions.
    """
    if data_type == 'numerical':
        if y is None:
            # Numerical distribution (histogram)
            sns.histplot(data=df, x=x, **kwargs)
            plt.title(f'Distribution of {x}')
        else:
            # Numerical distribution grouped by a categorical column (boxplot)
            sns.boxplot(data=df, x=x, y=y, **kwargs)
            plt.title(f'Distribution of {x} with respect to {y}')
        plt.show()

    elif data_type == 'datetime':
        if y is None:
            raise ValueError("Datetime plots require both x and y parameters.")
        # Line plot for datetime data
        sns.lineplot(data=df, x=x, y=y, **kwargs)
        plt.title(f'{y} Over Time ({x})')
        plt.show()

    elif data_type == 'categorical':
        if y is not None:
            raise ValueError("Categorical plots typically do not require a y parameter.")
        # Count plot for categorical data
        sns.countplot(data=df, x=x, **kwargs)
        plt.title(f'Category Frequency: {x}')
        plt.show()

    elif data_type == 'correlation':
        # Heatmap for correlation matrix
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", **kwargs)
        plt.title("Correlation Matrix Heatmap")
        plt.show()
    else:
        raise ValueError(
            "Invalid data type specified. Must be 'numerical', 'datetime', 'categorical', or 'correlation'.")


# pattern over time
# converting col to date type
# upon reading data:
# pd.read_csv('file.csv', parse_dates=['col'])
# pd.to_datetime()
# col time or hours object type
# Convert object to datetime then extract the required part such as hour or time
# pd.to_datetime(df['col'], format='%H%M').dt.hour/dt.time
# combining cols - pd.to_datetime(pd.to_datetime(pd[['col_month', 'col_day', 'col_year']])
# extract part of the date df['col'].dt.month/day/year
# visualizing pattern using lineplot(), agg y values at each value of x and show estimated mean and 95%
# confidence interval


# correlation between variables
# Correlation - Numerical
# direction and strength
# pd.corr(), pearson correlation, linear relationship
# sns.heatmap(pd.corr(), annot=True)
# visualizing correlation is important as the correlation might not be liner and therefore pearson method is
# not applicable, compliment the correlation calculation with scatterplot, pairplot (all numerical in one visualization)
# Categorical
# .value_counts()
# sns.histplot(data= , x='', hue= 'col to group by',binwidth=1)
# sns.kdeplot(data=, x='', hue='', cut=0) for visualizing distribution, especially when multiple distribution
# sns.scatterplot with hue as group by for categorical


# examining distribution using kdeplot()
# sns.kdeplot(data=, x='', hue='', cut=0) for visualizing distribution, especially when multiple distribution

def class_frequency(df, column_name, normalize=False, index_col=None, agg_col=None, agg_func=None):
    """
    Calculate class frequency or cross-tabulation for categorical data.
    Samples must be representative of the population.
    Otherwise, classes imbalance

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column for which to calculate frequency.
    normalize (bool): Whether to return relative frequency. Default is False (absolute frequency).
    index_col (str): The column to use as the index for cross-tabulation. Default is None.
    agg_col (str): The column to aggregate in cross-tabulation. Default is None.
    agg_func (str): The aggregation function for cross-tabulation (e.g., 'sum', 'mean'). Default is None.

    Returns:
    pd.Series or pd.DataFrame: Frequency or cross-tabulation table.
    """
    if index_col and agg_col and agg_func:
        # Perform cross-tabulation with aggregation
        return pd.crosstab(
            df[index_col],
            df[column_name],
            values=df[agg_col],
            aggfunc=agg_func
        )
    else:
        # Calculate class frequency
        return df[column_name].value_counts(normalize=normalize)


# generating new features using cut
# Correlation sns.heatmap(df.corr(), annot=True)
# such as extracting time or the day from a datetime variable pd.series.dt.weekday
# group numerical data and group them as classes, using as an example descriptive statistics
# bins = [0, quantile 0.25, median, 0.75, max]
# create labels as a list of categories
# df['new_col'] = pd.cut(df['cont to cat col'], labels=cat_list, bins=bins)
# np.inf (infinity)
# sns.heatmap(df.corr(), annot=True)
# sns.countplot(data=df, x="col to count", hue="cat to group by")

# generate hypothesis or question
# hypothesis testing prior to collecting data
# generate hypothesis or question
# decision on what statistical test to use

# decision on what statistical test to use
