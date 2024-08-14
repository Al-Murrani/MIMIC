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

# plot number of admission days
bar_days_stay = PlotlyPlots(group_of_days_admission)
bar_days_stay.plot('bar',
                   x='_NUMBEROFDAYSSTAY',
                   y='counts',
                   title='Number Of Admission Days',
                   labels={'_NUMBEROFDAYSSTAY': 'Number Of Admission Days', 'counts': 'Counts'})

# what is the most common reason (disease) for admission
admission_number_per_diagnosis = (df_mimic_to_process.count_unique_values('DIAGNOSIS', 'admission_number').sort_values
                                  (by='admission_number', ascending=False))

# what disease has the highest number of re-admission
# count the number of admission per patient
# select patients with count more one admission
patients_readmission = admission_number_per_subject[admission_number_per_subject['admission_number'] > 1]
# group by diagnosis and count and return diagnosis with the top count
patients_readmission_details = pd.merge(df_mimic_admission, patients_readmission, on='SUBJECT_ID', how='inner')
patients_readmission_details_to_process = Process(patients_readmission_details)
diagnosis_admission_readmission_number = (patients_readmission_details_to_process.count_unique_values
                                          ('DIAGNOSIS', 'readmission_number')
                                          .sort_values(by='readmission_number', ascending=False))
print(diagnosis_admission_readmission_number)

