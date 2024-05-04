import os
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import tensorflow as tf

from data import encode_categorical_to_int


def decision_tree(X_train, y_train, X_test, y_test, results_path):
    """
    Use a decision tree regressor to predict.
    :param X_train: Dataset to be trained on.
    :param y_train: target to be trained on.
    :param X_test: Dataset to be tested on.
    :param y_test: correct result to compare test prediction to.
    :param results_path:
    """
    # Replace categorical values:
    print("--- Replace categorical values. ---")
    cat_list = ["I", "F", "M"]
    num_list = [0, 1, 2]
    X_train['Sex'] = X_train['Sex'].replace(to_replace=cat_list, value=num_list)
    X_test['Sex'] = X_test['Sex'].replace(to_replace=cat_list, value=num_list)

    reg = tree.DecisionTreeRegressor()
    print("--- Started decision tree training. ---")
    reg = reg.fit(X_train, y_train)
    print("--- Finished decision tree training. ---")
    y_pred = reg.predict(X_test)

    evaluate_regression(y_test, y_pred, 'decision tree')


def linear_regression(X_train, y_train, X_test, y_test, results_path):
    """
    Use a linear regression to predict.
    :param X_train: Dataset to be trained on.
    :param y_train: target to be trained on.
    :param X_test: Dataset to be tested on.
    :param y_test: correct result to compare test prediction to.
    """
    # Remove categorical columns
    X_train = X_train.drop(['Sex'], axis=1)
    X_test = X_test.drop(['Sex'], axis=1)

    reg = LinearRegression()
    reg = reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    evaluate_regression(y_test, y_pred, 'linear regression')


def polyfit_regression(X_train, y_train, X_test, y_test, results_path):
    """
    # Use 3rd degree polynomials for weights and linear functions for length dimensions.
    :param X_train: Dataset to be trained on.
    :param y_train: target to be trained on.
    :param X_test: Dataset to be tested on.
    :param y_test: correct result to compare test prediction to.
    """
    # Remove categorical columns
    X_train = X_train.drop(['Sex'], axis=1)
    X_test = X_test.drop(['Sex'], axis=1)

    X_train = transform_df_for_partial_polynomial_reg(X_train)
    X_test = transform_df_for_partial_polynomial_reg(X_test)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    evaluate_regression(y_test, y_pred, 'polynomial regression')


def transform_df_for_partial_polynomial_reg(x):
    x_weight = x[['Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']]
    x_others = x[['Length', 'Diameter', 'Height']]
    poly = PolynomialFeatures(degree=3)  # polynomial feature combination matrix
    x_weight = pd.DataFrame(poly.fit_transform(x_weight))
    x = pd.concat([x_weight, x_others], axis=1)
    x.columns = x.columns.astype(str)
    return x


def neural_network(X_train, y_train, X_test, y_test, results_path, retrain=True):
    nn_path = os.path.join(results_path, 'nn_weights')

    # Train the network either if no checkpoints exist or if the 'retrain' option is set.
    if retrain or not os.path.exists(nn_path) or not os.listdir(nn_path):
        train_neural_network(X_train, y_train, nn_path)
    # Always evaluate the network performance.
    evaluate_best_neural_network(X_test, y_test, nn_path)


def train_neural_network(X_train, y_train, nn_path):
    X_train = encode_categorical_to_int(X_train)
    number_columns = X_train.shape[1]
    nn_model = create_seq_nn(number_columns)

    if not os.path.exists(nn_path):
        os.makedirs(nn_path)
    else:
        try:
            files = os.listdir(nn_path)
            for file in files:
                file_path = os.path.join(nn_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Old network checkpoints deleted successfully.")
        except OSError:
            print("Error occurred while deleting old checkpoint files.")

    checkpoint_name = os.path.join(nn_path, 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    nn_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=callbacks_list)


def evaluate_best_neural_network(X_test, y_test, nn_path):
    X_test = encode_categorical_to_int(X_test)
    files = os.listdir(nn_path)
    files.sort()
    best_weights_file = os.path.join(nn_path, files[-1])  # Latest checkpoint performed best.
    nn_model = create_seq_nn(X_test.shape[1])
    nn_model.load_weights(best_weights_file)  # load checkpoint
    y_pred = nn_model.predict(X_test)
    evaluate_regression(y_test, y_pred, 'sequential nn')


def create_seq_nn(input_dim):
    """
    Creates a simple sequential neural network for regression (1 linear output layer), with three hidden layers.
    :param input_dim: Number of variables in the input data.
    :return: the neural network.
    """
    nn = Sequential()

    # The Input Layer:
    nn.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))

    # Three Hidden Layers:
    nn.add(Dense(256, kernel_initializer='normal', activation='relu'))
    nn.add(Dense(256, kernel_initializer='normal', activation='relu'))
    nn.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer:
    nn.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network:
    nn.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    # nn.summary()
    return nn


def evaluate_regression(y_true, y_pred, model='model'):
    print(f"{model:-^28}")
    mse = mean_squared_error(y_true, y_pred)
    print(f"mean squared error:\t\t{mse:.2f}")  # 9.1
    print(f"root mean sq. error:\t{np.sqrt(mse):.2f}")  # 3.0
    mae = mean_absolute_error(y_true, y_pred)
    print(f"mean absolute error:\t{mae:.2f}")  # 2.1
    r2 = r2_score(y_true, y_pred)
    print(f"r2-score:\t\t\t\t{r2:.2f}")  # 0.1
    print("-" * 28 + "\n")


def ml_analysis():
    results_path = 'Results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train_path = 'Data/train.csv'
    train = pd.read_csv(train_path)
    X_train = train.drop('Rings', axis=1)
    y_train = train['Rings']

    test_path = 'Data/test.csv'
    test = pd.read_csv(test_path)
    X_test = test.drop('Rings', axis=1)
    y_test = test['Rings']

    decision_tree(X_train, y_train, X_test, y_test, results_path)  # rmse: 3.0, mae: 2.1, r²: 0.17
    linear_regression(X_train, y_train, X_test, y_test, results_path)  # rmse: 2.2, mae: 1.6, r²: 0.57
    polyfit_regression(X_train, y_train, X_test, y_test, results_path)  # rmse: 2.1, mae: 1.5, r²: 0.60
    neural_network(X_train, y_train, X_test, y_test, results_path, retrain=True)  # rmse: 2.0, mae: 1.3, r²: 0.65
