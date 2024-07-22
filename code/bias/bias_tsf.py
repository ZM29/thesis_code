"""
This script performs the feature importance analysis of the TSF.
"""

import torch
from TSF import TSF
import pandas as pd
import tqdm as tqdm
import numpy as np
from statsmodels.tsa.stattools import adfuller
from torch.optim import AdamW
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from sklearn.model_selection import train_test_split

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



def permutation_importance(model, test_tensor, baseline_score, feature_names, num_permutations = 5):
    """
    Function that calculates the importance for each feature in this study. This is achieved by measuring the change in 
    performance after permuting the feature's values randomly. 
    """
    feature_importance = {}
    loss_function = nn.MSELoss().to(device)

    #looping over all the features
    for feature in range(test_tensor.shape[0]):
        permuted_scores = []

        for _ in range(num_permutations):
            X_test_permuted = test_tensor.clone().detach()
            window = 0
            test_loss = 0

            #looping over time steps and permuting the feature values
            for t in range(1, X_test_permuted.shape[1] - 1):

                #permuting the feature values
                X_test_permuted[feature, :t+1, :] = X_test_permuted[feature, torch.randperm(t+1).to(device), :]

                #getting the input values for the model
                input_data = X_test_permuted[:, :t+1, :].to(device)
                output = model(input_data)
                
                #setting the target value for the model and computing loss
                target = X_test_permuted[:, t+1, :][0].to(device)
                permuted_score = loss_function(output, target)
                test_loss += permuted_score.item()
                window += 1

            test_loss /= window
            permuted_scores.append(test_loss)

        #calculating the importance score by measuring the difference compared to the original score after taking the average of the permuted scores
        importance_score = baseline_score - np.mean(permuted_scores)
        feature_name = feature_names[feature]
        feature_importance[feature_name] = importance_score

    return feature_importance



def analysis(file_path, storage, study_name):
    """
    Function that conduct the analysis by checking for stationarity, retrieving the best model configuration (per index) from
    Optuna's database, training, forecasting and visualizing the feature importance.
    """
    data = load_data(file_path)
    index_names = data['Index'].unique()

    #looping over all the indices
    for index_name in tqdm.tqdm(index_names, desc=f"Analyzing indices:"):
        print(f'Analyzing: {index_name}')

        #filtering the data and saving the column names for the feature importance
        index_data = data[data["Index"] == index_name]
        index_data = index_data.drop(columns = ['Index'])
        index_data = index_data.drop(columns = 'Date')
        column_names = list(index_data.columns)

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
        num_epochs = epochs_index.get(index_name, 100)
        patience = best_params['patience']  
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
            
            #calculating feature importance and preparing the data for visualization
            feature_importance = permutation_importance(best_model, test_tensor, test_loss, column_names)
            feature_importance = pd.DataFrame(feature_importance.items(), columns = ['Feature', 'Permutation Importance'])
            name_mapping = {'ICS_ALL': 'ICS', 'Close': 'Returns Lagged'}
            feature_importance['Feature'] = feature_importance['Feature'].replace(name_mapping)
            
            #sorting the features by importance
            feature_names = feature_importance['Feature']
            importance_scores = feature_importance['Permutation Importance']
            feature_importance_list = list(zip(feature_names, importance_scores))
            sorted_importance = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
            sorted_feature_names = [x[0] for x in sorted_importance]
            sorted_importance_scores = [x[1] for x in sorted_importance]

            #visualizing feature importance
            plt.figure(figsize=(24, 12))
            bar_colors = ['red' if feature in [] else 'blue' for feature in sorted_feature_names]
            bars = plt.bar(range(len(sorted_importance)), sorted_importance_scores, color=bar_colors)
            plt.xticks(range(len(sorted_importance)), sorted_feature_names, rotation = 45, ha = 'right')
            plt.xlabel('Features', fontsize = 24)
            plt.ylabel('Permutation Importance', fontsize = 24)
            plt.xticks(fontsize = 24)
            plt.yticks(fontsize = 24)

            ax = plt.gca()
            xticklabels = ax.get_xticklabels()
            for label in xticklabels:
                if label.get_text() in ['ICS', 'Bullish', 'Bearish', 'Returns Lagged']:
                    label.set_fontweight('bold')

            plt.axhline(y = 0, color = 'black', linestyle = '-', linewidth = 0.8)
            plt.tight_layout()
            #plt.savefig(f'plots/tsf/{index_name}_feature_importance.png')
            plt.show()


file_path = r'datasets\sentiment_numeric.csv'
study_name = 'TSF_optimization'
storage = 'sqlite:///datasets/optimize/TSF/TSF'
epochs_index = {
    '^GSPC': 12, 
    '^FTSE': 4, 
    '^RUT': 7, 
    '^DJI': 7, 
    '^IXIC': 4 
}
analysis(file_path, storage, study_name)