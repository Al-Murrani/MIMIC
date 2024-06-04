import read
from process import Process
import plotly.express as px

# 1. read the admission data into dataframe
df_mimic_admission = read.read_file_to_dataframe('PathToAdmissionCSVFile')

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
scatter = px.scatter(admission_number_per_subject,
                     y='admission_number',
                     x='SUBJECT_ID',
                     title='Admission Number Per Subject',
                     labels={'SUBJECT_ID': 'Subject ID', 'admission_number': 'Admission Number'})
scatter.update_layout(title_x=0.5)
scatter.show()

bar = px.bar(number_of_admissions,
             x='admission_number',
             y='number_subjects',
             title='Number Of Patients Per Admission Number',
             labels={'admission_number': 'Admission Number', 'number_subjects': 'Log Number Of Subjects'})
bar.update_yaxes(type='log')
bar.update_layout(title_x=0.5)
bar.show()
