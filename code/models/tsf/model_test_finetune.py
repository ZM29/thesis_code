## Why only change learning rate at validation?

"""
This script performs the parameter tuning of the TSF.
"""

import torch
from TSF import TSF
import pandas as pd
import tqdm as tqdm
import numpy as np
from statsmodels.tsa.stattools import adfuller
from torch.optim import AdamW
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

#seed and setting device to GPU or CPU
torch.manual_seed(29)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(file_path):
    """
    Function to change load data and change format.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
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



def tensor_maker(data):
    """
    Function to create appropriate tensors that are used in the TSF. This is achieved by converting all columns to numeric type.
    After this, the function adds an extra dimensions for compatibility reasons.  
    """
    numeric_columns = data.columns
    data.loc[:, numeric_columns] = data.loc[:, numeric_columns].apply(pd.to_numeric, errors = 'coerce')

    #extra dimension for compatibility reasons
    data_tensor = np.expand_dims(data.values.T, axis = 2)
    data_tensor = torch.tensor(data_tensor, dtype = torch.float32, requires_grad = True)
    return data_tensor



def closest_divisible(model_dim, num_heads):
    """
    Function to determine the closest divisible number due to the design of the GPT2 blocks.
    The design evenly splits the subspace for the heads. Therefore, the closest divisible number needs to be found. 
    """
    quotient = model_dim // num_heads
    closest = num_heads * quotient
    if model_dim - closest > num_heads // 2:
        closest += num_heads
    return closest



def objective(trial, train_data):
    """
    Function for Optuna to optimize hyperparameters of the model.
    """

    #defining the optimizing range of the hyperparameters
    input_dim = 1
    model_dim = trial.suggest_int('model_dim', 8, 128)
    patch_size = trial.suggest_int('patch_size', 10, 50)
    max_len = 10000
    num_layers = trial.suggest_int('num_layers', 2, 20)
    num_heads = trial.suggest_int('num_heads', 1, 8)
    hidden_dim = trial.suggest_int('hidden_dim', 100, 1000)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 100)
    patience = trial.suggest_int('patience', 3, 10)

    #ensuring model_dim is divisible by num_heads which is required for GPT-2 block
    model_dim = closest_divisible(model_dim, num_heads)

    #initializing the model with the hyperparameters
    output_horizon = 1
    model = TSF(input_dim, model_dim, patch_size, max_len, num_layers,
                num_heads, hidden_dim, dropout, output_horizon).to(device)

    optimizer = AdamW([
        {'params': model.transformer.parameters()},
    ], lr = learning_rate, weight_decay = 0.01)
    

    loss_function = nn.MSELoss()
    best_val_loss = float('inf')

    #setting up the splits
    n_splits = 3
    splits = TimeSeriesSplit(n_splits = n_splits)
    losses = []

    #time series cross-validation
    for train_index, val_index in splits.split(train_data):
        train_set = train_data.iloc[train_index]
        val_set = train_data.iloc[val_index]

        #making tensors
        train_tensor = tensor_maker(train_set).to(device)
        val_tensor = tensor_maker(val_set).to(device)

        #initialize early stopping
        no_improvement = 0
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = patience, factor = 0.5)
        val_losses = []

        #training loop
        for epoch in range(num_epochs):
            # print(f"Starting Epoch {epoch}")
            model.train()
            train_loss = 0
            window = 0

            #looping over time steps for training
            for t in range(1, train_tensor.shape[1] - 1):
                #zero out gradients
                optimizer.zero_grad()

                #getting the input values for the model
                input_data = train_tensor[:, :t+1, :]
                output = model(input_data)

                #setting the target value for the model and computing loss
                target = train_tensor[:, t+1, :][0]
                loss = loss_function(output, target)
                train_loss += loss.item()
                window += 1

                #compute gradients
                loss.backward()
                #update model parameters based on the gradients
                optimizer.step()

            train_loss /= window

            #evaluation on validation data
            model.eval()
            val_loss = 0
            window = 0

            with torch.no_grad(): #this disables gradient computation and essentially freezes the model's knowledge
                for t in range(1, val_tensor.shape[1] - 1):
                    #getting the input values for the model
                    input_data = val_tensor[:, :t+1, :]
                    output = model(input_data)

                    #setting the target value for the model and computing loss
                    target = val_tensor[:, t+1, :][0]
                    loss = loss_function(output, target)
                    val_loss += loss.item()
                    window += 1

            val_loss /= window
            val_losses.append(val_loss)

            #adjusting the learning rate based on the loss
            scheduler.step(val_loss)

            #early stopping if it does not yield better performance (after several epochs) 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break

            #pruning unpromising trials
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        losses.append(np.array(val_losses).mean())

    return (np.array(losses).mean())

def analysis(file_path, storage, study_name):
    """
    Function that conducts the analysis by checking for stationarity and tuning the hyperparameters of the model.
    """
    data = load_data(file_path)
    index_names = data['Index'].unique()

    #looping over all the indices
    for index_name in tqdm.tqdm(index_names, desc=f"Analyzing indices:"):
        print(f'Analyzing: {index_name}')

        #filtering the data
        index_data = data[data["Index"] == index_name]
        index_data = index_data.drop(columns=['Index'])
        index_data = index_data.drop(columns='Date')

        #making the data stationary
        stationary_data = stationary(index_data)
        stationary_data = stationary_data.iloc[:-1]
        tsf_data = stationary_data.copy()

        #setting up Optuna's storage and study
        index_storage = storage + '_' + index_name + '_study.db'
        index_study = study_name + '_' + index_name

        train_data, test_data = train_test_split(tsf_data, test_size = 0.2, shuffle = False)
        
        #loading or creating the corresponding Optuna study
        try:
            study = optuna.load_study(study_name=index_study, storage=index_storage)
        except KeyError:
            study = optuna.create_study(direction='minimize', study_name=index_study, storage=index_storage)

        ##tuning the hyperparameters by specifying the amount of trials
        # study.optimize(lambda trial: objective(trial, train_data), n_trials = 40)

        print("Best parameters: ", study.best_params)
        print("Best score: ", study.best_value)

        #initializing the model with the best found parameters
        best_params = study.best_params
        model_dim = closest_divisible(best_params['model_dim'], best_params['num_heads'])
        best_model = TSF(1, #input_dim
                    model_dim ,
                    best_params['patch_size'], 
                    10000, #max_len positional encoding
                    best_params['num_layers'],
                    best_params['num_heads'], 
                    best_params['hidden_dim'], 
                    best_params['dropout'], 
                    1).to(device) #output_horizon

        optimizer = AdamW([
            {'params': best_model.transformer.parameters()},
        ], lr = best_params['learning_rate'], weight_decay = 0.01)

        #defining the training parameters
        loss_function = nn.MSELoss()
        num_epochs = best_params['num_epochs']
        patience = best_params['patience']  
        best_loss = float('inf')

        #initialize early stopping
        no_improvement = 0
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = patience, factor = 0.5) 

        #making tensors
        train_tensor = tensor_maker(train_data).to(device)
        test_tensor = tensor_maker(test_data).to(device)

        #training loop
        for epoch in range(num_epochs):
            best_model.train()
            train_loss = 0
            window = 0

            #looping over time steps for training
            for t in range(1, train_tensor.shape[1] - 1):
                #zero out gradients 
                optimizer.zero_grad()

                #getting the input values for the model
                input_data = train_tensor[:, :t+1, :]
                output = best_model(input_data)

                #setting the target value for the model and computing loss
                target = train_tensor[:, t+1, :][0]
                loss = loss_function(output, target)
                train_loss += loss.item()
                window += 1

                #compute gradients
                loss.backward()
                #update model parameters based on the gradients
                optimizer.step()

            train_loss /= window
            #adjusting the learning rate based on the loss
            scheduler.step(train_loss)

            if train_loss < best_loss:
                best_loss = train_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break

        print(f'Epoch {epoch}: DONE')

        #evaluation on test data
        best_model.eval()
        test_loss = 0
        window = 0

        with torch.no_grad(): #this disables gradient computation and essentially freezes the model's knowledge
            for t in range(1, test_tensor.shape[1] - 1):
                #getting the input values for the model
                input_data = test_tensor[:, :t+1, :]
                output = best_model(input_data)

                #setting the target value for the model and computing loss
                target = test_tensor[:, t+1, :][0]
                loss = loss_function(output, target)
                test_loss += loss.item()
                window += 1

            test_loss /= window

file_path = r'datasets\sentiment_numeric.csv'
study_name = 'TSF_optimization'
storage = 'sqlite:///datasets/optimize/TSF/TSF'
analysis(file_path, storage, study_name)


