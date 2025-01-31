import pandas as pd


# .head() returns the first few rows (the “head” of the DataFrame).
# .info() shows information on each of the columns, such as the data type and number of missing values.
# .shape returns the number of rows and columns of the DataFrame.
# .describe() calculates a few summary statistics for each column.
# drop duplicate then value_counts
# WHERE clause is df[df['col'] == condition]
# dataframe[[columns list]].agg([function list]) #per group per column
# summary stat per group for each column
# dataframe.groupby(['col1', 'col2'] to group by)[['col1, col2]] to calculate function].agg([fun1, fun2,..])
# df.groupby('col to group on').agg({'col to agg':'func'})
# pivot table same as above df.pivot_table(value="value to aggregate", index="group by and displays in rows"
# columns="group by and display in columns", aggfunc=summary func). Result in sorted df on indexes
# Object is series, method is to count the number of elements in the underlying data
# df.idxmax Return index of first occurrence of maximum over requested axis.
# pivot table mean
# pivot table is sorted df on indexes so can use .loc
# .set_index, .sort_index, .loc[]
# dates in ISO 8601 format, "yyyy-mm-dd", "yyyy-mm", and "yyyy"
# df.isna().any() or df.isna().any().sum(), df.isna().sum().plot(kinds='bar'), .dropna, .fillna()


# new_dict = {
#   "key1": [],
#   "skey2": [],
#   "key3": []
# }
#
# # Convert dictionary into DataFrame
# df = pd.DataFrame(new_dict)
