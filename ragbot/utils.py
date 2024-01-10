import os
import pickle
import pandas as pd
import parquet
from config import GOOGLE_MANUAL_URL


def get_tenant_info_from_df(col_name, tenant_name):
    df = pd.read_csv(GOOGLE_MANUAL_URL)
    matching_rows = df[df['tenant_name'] == tenant_name].iloc[0]
    if not matching_rows.empty:
        return matching_rows[col_name]
    else:
        return "No matching tenant found"

def read_pickle(pickle_file_name):
    try:
        with open(pickle_file_name, 'rb') as file:
            print('in open')
            pkl_data = pickle.load(file)
    except FileNotFoundError:
        print(f"The file {pickle_file_name} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return pkl_data


def read_all_pickles(pickle_dir):
    df = pd.DataFrame()
    pickle_names = os.listdir(pickle_dir)
    for file in pickle_names:
        temp = read_pickle(f'{pickle_dir}/{file}')
        temp['filename'] = file.split('.')[0]
        df = pd.concat([df, temp], ignore_index=True)
    df.rename(columns={'values': 'embedding'}, inplace=True)
    return df


def save_object_to_pickle(obj, filepath, output_filename):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print(f"Created directory: {filepath}")
    destination_path = os.path.join(filepath, output_filename)
    with open(destination_path, 'wb') as file:
        pickle.dump(obj, file)
    print(f"File written to {destination_path}")

    import os

def write_parquet_file(filename, df):
    df.to_parquet(filename, engine='pyarrow')

def read_parquet_file(filename):
    return pd.read_parquet(filename, engine='pyarrow')

def print_tree(directory, 
               excluded_folders=None, 
               excluded_files=None, 
               prefix=''):
    if excluded_folders is None:
        excluded_folders = []
    if excluded_files is None:
        excluded_files = []

    if prefix == '':  # Only print this line at the topmost level
        print(os.path.basename(directory) + '/')

    files = [f for f in os.listdir(directory) 
             if f not in excluded_folders 
             and f not in excluded_files]
    for index, file in enumerate(files):
        path = os.path.join(directory, file)
        is_last = index == len(files) - 1
        print(prefix + '└── ' if is_last else prefix + '├── ', file)
        if os.path.isdir(path) and file not in excluded_folders:
            extension = '    ' if is_last else '│   '
            print_tree(
                path, 
                excluded_folders, 
                excluded_files, 
                prefix=prefix+extension
            )
    
