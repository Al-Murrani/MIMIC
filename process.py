import pandas as pd


class Process:
    """
    A class for performing various data processing operations on pandas DataFrame.

    This class provides methods to convert data types of specified columns, count missing values,
    identify duplicate rows, and count unique values in columns of the DataFrame.
    """
    def __init__(self, dataframe):
        self.df = dataframe

    def data_type_conversion(self, columns_list, target_type):
        for column in columns_list:
            if target_type == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column])
            elif target_type == 'string':
                self.df[column] = self.df[column].astype(str)
        return self.df.dtypes

    def count_na(self):
        na_counts = self.df.isna().sum().to_dict()
        return na_counts

    def find_duplicate(self, mark_duplicate, subset_list):
        try:
            if len(subset_list) != 0:
                duplicate = self.df[self.df.duplicated(subset_list, keep=mark_duplicate)]
            else:
                duplicate = self.df[self.df.duplicated(keep=mark_duplicate)]
            duplicate_count = duplicate.count()
            return duplicate_count
        except ValueError:
            f"Column {subset_list} not found in the dataframe."

    def count_unique_values(self, column, out_put_column):
        try:
            return self.df.groupby(column).size().reset_index(name=out_put_column)
        except ValueError:
            f"Column {column} not found in the dataframe."
