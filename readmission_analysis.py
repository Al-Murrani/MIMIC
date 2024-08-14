import pandas as pd


class ReadmissionAnalysis:
    def __init__(self, processed_admission_df, original_admission_df):
        self.processed_admission_df = processed_admission_df
        self.original_admission_df = original_admission_df

    def get_patients_with_multiple_admissions(self):
        """
        Select patients with more than one admission.
        """
        return self.processed_admission_df[self.processed_admission_df['admission_number'] > 1]

    def merge_admission_details(self, patients_readmission, merge_on, merge_method):
        """
        Merge admission details with patients having multiple admissions.
        """
        return pd.merge(self.original_admission_df, patients_readmission, on=merge_on, how=merge_method)

    def get_top_diagnosis_by_readmission(self, merged_df, diagnosis_col, count_col):
        """
        Group by diagnosis and count, then return the diagnosis with the highest readmission count.
        """
        diagnosis_counts = merged_df.groupby(diagnosis_col).size().reset_index(name=count_col)
        return diagnosis_counts.sort_values(by=count_col, ascending=False)

    def run_analysis(self):
        """
        Run the entire analysis process.
        """
        patients_readmission = self.get_patients_with_multiple_admissions()
        patients_readmission_details = self.merge_admission_details(patients_readmission,
                                                                    'SUBJECT_ID',
                                                                    'inner')
        top_diagnosis = self.get_top_diagnosis_by_readmission(patients_readmission_details,
                                                              'DIAGNOSIS',
                                                              'readmission_number')
        return top_diagnosis
