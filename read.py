import pandas as pd
import os


def read_file_to_dataframe(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file path {file_path} does not exists.")

    file_extension = file_path.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            return pd.read_csv(file_path)
        elif file_extension == 'json':
            return pd.read_json(file_path)
        elif file_extension in ['xls', 'xlsx']:
            return pd.read_excel(file_path)
    except ValueError:
        f"Unsupported file extension: {file_extension}"


def read_files_to_dataframe(directory_containing_files, ignore_index):
    for files in os.walk(directory_containing_files):
        concat_files = pd.concat(
            map(pd.read_csv, files), ignore_index=ignore_index)
        return concat_files
