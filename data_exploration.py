import os.path
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from data import encode_categorical_to_int


def plot_data():
    file_path = 'Data/train.csv'
    df = pd.read_csv(file_path)

    palette = {"I": 'green', "F": 'red', "M": 'blue'}

    # df.plot.scatter(x='Whole weight', y='Rings')
    # To get a quick overview, plot all columns against each other.
    # sns.pairplot(df, hue='Sex', plot_kws={'alpha':0.2}, palette=palette)
    # Findings:
    # - Length, diameter and height are approx. linearly related.
    # - Whole weight, shucked weight, viscera weight, and shell weight are
    # approx. linearly related.
    # - Weight increases faster than each length/diameter/height dimension.
    # Weight probably nearly linearly correlated with volume.
    # Infant is smaller and has fewer rings, male and female don't differ.

    # Approximate volume as cuboid:
    volume = df.Length * df.Diameter * df.Height
    df.insert(4, "Volume", volume)
    # sns.pairplot(df, hue='Sex', plot_kws={'alpha':0.2}, palette=palette)
    # Confirms that weight linearly related to volume.

    # Cube root of volume -> average 1D length/height/diameter dimension.
    x = np.cbrt(volume) # cube root.
    df.insert(5, "X", x)
    # sns.pairplot(df, hue='Sex', plot_kws={'alpha': 0.2}, palette=palette)

    # Look at approximated abalone density:
    density = df['Whole weight'] / df.Volume
    df.insert(6, "Density", density)
    sns.pairplot(df, hue='Sex', plot_kws={'alpha':0.2}, palette=palette)
    if not os.path.exists('Results'):
        os.makedirs('Results')
    plt.savefig('Results/scatter_matrix_seaborn.png')
    # Density is mostly unrelated to all other variables and the number of rings.


def statistics(data_path):
    df = pd.read_csv(data_path)
    stats_df = df.describe()
    directory, filename = os.path.split(data_path)
    filename = os.path.splitext(filename)[0]
    html_path = os.path.join(directory, 'stats_' + filename + '.html')
    stats_df.to_html(html_path)


def zero_analysis():
    file_path = 'Data/abalone.csv'
    df = pd.read_csv(file_path)
    df_zero = df[df['Height'] == 0]
    print("Number of rows where Height=0:", df_zero.shape[0])
    # any column:
    df_zero = df.loc[(df == 0).any(axis=1)]
    print("Number of rows where any measurement=0:", df_zero.shape[0])


def outlier_analysis(column):
    file_path = 'Data/abalone.csv'
    df = pd.read_csv(file_path)
    # sns.boxplot(df[column])
    # plt.title(f'Box Plot of {column}')
    # plt.show()

    # Z-score
    df_zscore = (df[column] - df[column].mean()) / df[column].std()
    df_high = df[column].loc[df_zscore > 3]
    print(df_high)


def correlation_analysis(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(['Sex'], axis=1)  # remove categorical variables
    # df = replace_categorical_with_ints(df)
    corr_matrix = df.corr()
    fig = plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, vmin=0, vmax=1, cmap="coolwarm", square=True)
    if not os.path.exists('Results'):
        os.makedirs('Results')
    plt.savefig('Results/correlation_heatmap.svg')