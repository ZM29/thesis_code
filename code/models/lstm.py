## geen rolling window bij tuning


"""
This script performs the parameter tuning of the LSTM.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class LSTMModel(nn.Module):
    """
    An LSTM model for time series forecasting. This model consists of an LSTM layer 
    followed by a dropout layer and a linear layer for prediction.
    """
    def __init__(self, num_features, hidden_size, num_layers, output_size, probability):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first = True)
        self.drop = nn.Dropout(probability)
        self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out)
        out = self.layer(out)
        return out
    
def load_data(file_path):
    """
    Function to change load data and change format.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index(['Date'])
    return data

def stationary(data):
    """
    Function to make the data stationary by using the ADF test. If a feature (series) is non-stationary, 
    the function is repeated until the series is stationary.
    """
    stationary_data = pd.DataFrame()
    
    for column in data.columns:
        series = data[column]

        #NAs need to be dropped to ensure results 
        result = adfuller(series.dropna())
        
        #results[1] is the P-value of the test
        if result[1] > 0.05:
            diff_series = stationary(pd.DataFrame(series.diff().dropna(), columns = [column]))
            stationary_data = pd.concat([stationary_data, diff_series], axis = 1)
        else:
            stationary_data = pd.concat([stationary_data, series], axis = 1)

    return stationary_data


def objective(trial, X_train, y_train):
    """
    Function for Optuna to optimize hyperparameters of the model.
    """

    #defining the optimizing range of the hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 1, 512)
    num_layers = trial.suggest_int('num_layers', 1, 6)
    probability = trial.suggest_float('probability', 0.1, 0.5)
    max_epochs = trial.suggest_int('max_epochs', 100, 500)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam'])
    
    #initializing the model with the hyperparameters
    model = NeuralNetRegressor(
        LSTMModel,
        module__num_features = X_train.shape[1],
        module__hidden_size = hidden_size,
        module__num_layers = num_layers,
        module__output_size = 1,
        module__probability = probability,
        max_epochs = max_epochs,
        lr = lr,
        batch_size = batch_size,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        optimizer = torch.optim.__dict__[optimizer_name],
        optimizer__weight_decay = 1e-5,
        verbose = 0
    )

    #setting up the splits
    splits = TimeSeriesSplit(n_splits = 3)
    mse_scores = []

    #time series cross-validation
    for train_index, val_index in splits.split(X_train):

        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        #scaling the data and converting to tensors
        scaler_features = StandardScaler()
        X_train_fold = scaler_features.fit_transform(X_train_fold)
        X_val_fold = scaler_features.transform(X_val_fold)

        scaler_target = StandardScaler()
        y_train_fold = scaler_target.fit_transform(y_train_fold)
        y_val_fold = scaler_target.transform(y_val_fold)

        X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32)
        X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32)
        y_train_fold = torch.tensor(y_train_fold, dtype=torch.float32)
        y_val_fold = torch.tensor(y_val_fold, dtype=torch.float32)

        model.fit(X_train_fold, y_train_fold)
        y_prediction = model.predict(X_val_fold)

        #inversing forecasts and calculating MSE
        prediction_descaled = scaler_target.inverse_transform(y_prediction.reshape(-1, 1))
        fold_descaled = scaler_target.inverse_transform(y_val_fold.numpy().reshape(-1, 1))
        mse = nn.MSELoss()(torch.tensor(prediction_descaled), torch.tensor(fold_descaled))
        mse_scores.append(mse.item())

    return np.mean(mse_scores)

def analysis(file_path, study_name, storage):
    """
    Function that conducts the analysis by checking for stationarity and tuning the hyperparameters of the model.
    """
    data = load_data(file_path)
    index_names = data['Index'].unique()

    #looping over all the indices
    for index_name in index_names:
        print(f"Evaluating index: {index_name}")

        #filtering the data
        index_data = data[data['Index'] == index_name]
        index_data = index_data.drop("Index", axis = 1)
        
        #making the data stationary and adding lags as a feature
        stationary_data = stationary(index_data)
        stationary_data = stationary_data.reset_index()
        stationary_data = stationary_data.rename(columns={'index': 'Date'})
        stationary_data = stationary_data.iloc[:-1]
        stationary_data['Close_Lagged'] = stationary_data['Close'].shift(1).fillna(method='bfill')
        lstm_data = stationary_data.copy()

        #setting up Optuna's storage and study
        index_storage = storage + '_' + index_name + '_study.db'
        index_study = study_name + '_' + index_name

        features = lstm_data.drop(['Close', 'Date'], axis = 1)
        targets = lstm_data[['Close']]

        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, shuffle = False)

        #loading or creating the corresponding Optuna study
        try:
            study = optuna.load_study(study_name = index_study, storage = index_storage)
        except KeyError:
            study = optuna.create_study(direction = 'minimize', study_name = index_study, storage = index_storage)

        ##tuning the hyperparameters by specifying the amount of trials
        #study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials = 50)

        print("Best parameters: ", study.best_params)
        print("Best score: ", study.best_value)
        
        #initializing the model with the best found parameters
        best_params = study.best_params
        best_model = NeuralNetRegressor(
            LSTMModel,
            module__num_features = X_train.shape[1],
            module__hidden_size = best_params['hidden_size'],
            module__num_layers = best_params['num_layers'],
            module__output_size = 1,
            module__probability = best_params['probability'],
            max_epochs = best_params['max_epochs'],
            lr = best_params['lr'],
            batch_size = best_params['batch_size'],
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
            optimizer = torch.optim.__dict__[best_params['optimizer']],
            optimizer__weight_decay = 1e-5,
            verbose = 0
        )

        #scaling the data and converting to tensors
        scaler_features = StandardScaler()
        X_train = scaler_features.fit_transform(X_train)
        X_test = scaler_features.transform(X_test)

        scaler_target = StandardScaler()
        y_train = scaler_target.fit_transform(y_train)
        y_test = scaler_target.transform(y_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32) 

        best_model.fit(X_train, y_train)
        forecasts = []
        train_set = X_test[0].unsqueeze(0)

        #making rolling window predictions
        for i in range(len(X_test)):
            forecast = best_model.predict(train_set)
            forecasts.append(forecast.item())

            if i < len(X_test) - 1:
                train_set = X_test[i + 1].unsqueeze(0)

        #inversing forecasts and calculating MSE
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = scaler_target.inverse_transform(forecasts)
        actual_values = scaler_target.inverse_transform(y_test.numpy())
        mse = mean_squared_error(actual_values, forecasts)

file_path = r'datasets\sentiment_numeric.csv'
study_name = 'lstm_optimization'
storage = 'sqlite:///datasets/optimize/lstm/lstm'
analysis(file_path, study_name, storage)   

