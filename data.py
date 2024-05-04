import os
import requests
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np


def download_data(file_path):
    url = 'https://archive.ics.uci.edu/static/public/1/abalone.zip'
    dir_path, file_name = os.path.split(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    r = requests.get(url)
    file = open(file_path, 'wb')
    file.write(r.content)
    file.close()


def unzip_data(file_path):
    with ZipFile(file_path, 'r') as zObject:
        zObject.extractall(path='Data')


def add_column_names(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                  'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    return df


def data_split(df, directory):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv(os.path.join(directory, 'train.csv'), index=False)
    test.to_csv(os.path.join(directory, 'test.csv'), index=False)


def encode_categorical_to_int(df):
    cat_list = ["I", "F", "M"]
    num_list = [0, 1, 2]
    df['Sex'] = df['Sex'].replace(to_replace=cat_list, value=num_list)
    return df


def clean_data(data_path):
    df = pd.read_csv(data_path)
    print("Original number of rows:", df.shape[0])
    df = df.dropna()
    print("Number of rows after removing NaNs:", df.shape[0])
    # Remove zero values, except for categorical and predicted column.
    df = df.loc[~(df[df.columns.difference(['Rings', 'Sex'])] == 0.0).any(axis=1)]
    print("Number of rows after removing zero-values:", df.shape[0])
    # Remove extreme outliers (z-score > 3), except for categorical and predicted column.
    df = df[(np.abs(stats.zscore(df[df.columns.difference(['Rings', 'Sex'])])) < 3).all(axis=1)]
    print("Number of rows after removing outliers:", df.shape[0])
    df.to_csv(data_path, index=False)


def prepare_data():
    data_path = 'Data/abalone.data'
    if not os.path.isfile(data_path):
        zip_path = 'Data/abalone.zip'
        print("Download and extract data set.")
        download_data(zip_path)
        unzip_data(zip_path)

    df = add_column_names(data_path)
    #df = replace_categorical_with_ints(df)
    csv_path = 'Data/abalone.csv'
    df.to_csv(csv_path, index=False)
    data_split(df, 'Data')
