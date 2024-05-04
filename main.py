from data import prepare_data, clean_data
from data_exploration import statistics, plot_data, zero_analysis, outlier_analysis, correlation_analysis
from analysis import ml_analysis
import os
from scipy import stats


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prepare_data()
    statistics('Data/abalone.csv')
    # # Findings:
    # # - For all variables, mean and median are close together.
    # # - There are abalones with height = 0.0 --> potential error
    # # - Height has some outliers with values factor ~10 larger than the 75th percentile.
    zero_analysis()
    # # Findings: there are two rows with Height=0.
    outlier_analysis('Height')
    # # Findings: there are two extreme Height outliers.
    # # Remove NaNs, zero values and outliers from training data:
    clean_data('Data/train.csv')
    statistics('Data/train.csv')

    correlation_analysis('Data/train.csv')
    # plot_data()
    # Findings:
    # - Length, diameter and height are approx. linearly related.
    # - Whole weight, shucked weight, viscera weight, and shell weight are approx. linearly related.
    # - Weight and approximated volume (l*d*h) are linearly related.
    ml_analysis()
    # Tried multiple models for regression:
    # Decision tree, linear reg., mixed linear and polynomial reg, sequential neural network.
    # The NN showed the best results: R²=0.65, MAE=1.3, RMSE=2.0

