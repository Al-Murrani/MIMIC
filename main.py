import pandas as pd
import plotly.express as px
import read
from process import Process
from plots import PlotlyPlots

pd.set_option('display.max_columns', None)


def calculate_date_difference(row, second_date, first_date):
    if row[first_date].date() == row[second_date].date():
        return 0
    else:
        return (row[second_date] - row[first_date]).days


# 1. read the admission data into dataframe
df_mimic_admission = read.read_file_to_dataframe('C:\\Users\\amela\\mimic\\ADMISSIONS.csv')

admission_columns = df_mimic_admission.columns
admission_shape = df_mimic_admission.shape
columns_datatype = df_mimic_admission.dtypes

# 2. initialise
df_mimic_to_process = Process(df_mimic_admission)

admission_object_datetime_columns = df_mimic_to_process.data_type_conversion(['ADMITTIME', 'DISCHTIME', 'DEATHTIME'],
                                                                             'datetime')

admission_int_string_columns = df_mimic_to_process.data_type_conversion(['SUBJECT_ID'], 'string')

admission_na_counts = df_mimic_to_process.count_na()

admission_number_per_subject = df_mimic_to_process.count_unique_values('SUBJECT_ID', 'admission_number')

# 3. initialise
df_admission_no_per_subject_process = Process(admission_number_per_subject)

number_of_admissions = df_admission_no_per_subject_process.count_unique_values('admission_number', 'number_subjects')

# 4. plot Data
scatter = PlotlyPlots(admission_number_per_subject)
scatter.plot('scatter',
             y='admission_number',
             x='SUBJECT_ID',
             title='Admission Number Per Subject',
             labels={'SUBJECT_ID': 'Subject ID', 'admission_number': 'Admission Number'})

bar = px.bar(number_of_admissions,
             x='admission_number',
             y='number_subjects',
             title='Number Of Patients Per Admission Number',
             labels={'admission_number': 'Admission Number', 'number_subjects': 'Log Number Of Subjects'})
bar.update_yaxes(type='log')
bar.update_layout(title_x=0.5)
bar.show()

# 5. Distribution of the number of admission days ('ADMITTIME', 'DISCHTIME')
df_mimic_admission['_NUMBEROFDAYSSTAY'] = (df_mimic_admission.apply(calculate_date_difference,
                                                                    args=('DISCHTIME', 'ADMITTIME'),
                                                                    axis=1))

group_of_days_admission = df_mimic_admission.groupby(['_NUMBEROFDAYSSTAY']).size().reset_index(name='counts')
print(group_of_days_admission)

# plot number of admission days
bar_days_stay = PlotlyPlots(group_of_days_admission)
bar_days_stay.plot('bar',
                   x='_NUMBEROFDAYSSTAY',
                   y='counts',
                   title='Number Of Admission Days',
                   labels={'_NUMBEROFDAYSSTAY': 'Number Of Admission Days', 'counts': 'Counts'})

