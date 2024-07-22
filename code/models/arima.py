#Only comment code and rolling window decision with TVAR

"""
This script performs the parameter tuning of the ARIMA.
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from statsmodels.tsa.stattools import adfuller
import numpy as np
import optuna
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.sandwich_covariance import cov_hac
from tabulate import tabulate
from scipy import stats

def load_data(file_path):
    """
    Function to change load data and change format.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index(['Date'])
    closing_data = data.iloc[:,:2]
    
    return closing_data


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

def objective(trial, train_data, p_max, q_max):
    """
    Function for Optuna to optimize hyperparameters of the model.
    """
    try:
        #defining the optimizing range of the hyperparameters
        p = trial.suggest_int('p', 1, p_max)
        d = 0
        q = trial.suggest_int('q', 1, q_max)
        trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])
        order = (p, d, q)
        
        #setting up the splits
        splits = TimeSeriesSplit(n_splits = 3)
        mse_scores = []
        
        #time series cross-validation
        for train_index, val_index in splits.split(train_data):
            train_set = train_data.iloc[train_index]
            val_set = train_data.iloc[val_index]
            
            forecasts = []
            actual_values = []

            #rolling window forecast
            for i in range(len(val_set)):
                train_fold = pd.concat([train_set, val_set.iloc[:i]])
                forecast, _ = forecast_values(train_fold, order, trend)
                forecasts.append(forecast)
                actual_values.append(val_set.iloc[i]['Close'])
            
            #compute loss
            mse = mean_squared_error(actual_values, forecasts)
            mse_scores.append(mse)
        
        return np.mean(mse_scores)
    
    #any exception due to incompatibility is pruned and the trial is redone
    except Exception as e:
        print(f"Error occurred in trial: {e}")
        raise optuna.TrialPruned()


def forecast_values(train_data, order, trend):
    """
    Function for forecasting values.
    """
    p, d, q = order

    #determing the number of trend parameters
    k_trend = 1
    if trend == 'c':
        k_trend = 2
    elif trend == 't':
        k_trend = 2
    elif trend == 'ct':
        k_trend = 3
    
    arima_model = SARIMAX(train_data['Close'], order = order, trend = trend) 

    #initializing start parameters for model fitting for stability reasons
    start_params = (p + d + q + k_trend) * [0]
    start_params[-1] = 1

    arima_results = arima_model.fit(method = 'powell', start_params = start_params, disp = False, maxiter = 100)
    forecast = arima_results.forecast(steps = 1).iloc[0]

    return forecast, arima_results


def white_noise(model_results):
    """
    Function to check whether the model results are white noise and apply Newey-West standard errors.
    """

    #getting the MA and AR orders
    p = model_results.model.k_ar
    q = model_results.model.k_ma
    max_lag = max(p, q)
    residuals = model_results.resid

    #performing t-test
    t_stat, t_pvalue = stats.ttest_1samp(residuals, 0)

    flag = False 

    #performing Ljung-Box test for autocorrelation
    lb_results = pd.DataFrame(columns = ['lb_test_stat', 'lb_pvalue'])
    lb_results = acorr_ljungbox(residuals, lags = [max_lag], return_df = True)
    lb_results = [lb_results['lb_stat'].iloc[0], lb_results['lb_pvalue'].iloc[0]]

    #performing Breusch-Pagan test for heteroskedasticity
    bp_results = pd.DataFrame(columns = ['bp_test_stat', 'bp_pvalue'])
    constant = np.ones_like(residuals)
    exogenous = np.column_stack((constant, residuals))
    bp_results = het_breuschpagan(residuals, exogenous)
    bp_results = [bp_results[-2], bp_results[-1]]

    ##this prints the tests in the table format used in the study
    # headers = ['Variable', 'Ljung-Box Test Statistic', 'Ljung-Box P-value', 'Breusch-Pagan Test Statistic', 'Breusch-Pagan P-value', 'T-test Test Statistic' , 'T-test P-value']
    # table_data = []
    # table_data.append([f'Close', lb_results[0], lb_results[1],
    #                     bp_results[0], bp_results[1], t_stat, t_pvalue])
        
    # table = tabulate(table_data, headers, tablefmt = 'grid', floatfmt = '.3f')
    #print(table)


    #checking for any violated white noise assumption
    if lb_results[1] <= 0.05:
        #print(f"\nThe residuals are serially correlated (Ljung-Box test up to lag {max_lag}).")
        flag = True

    if bp_results[-1] <= 0.05:
        #print(f"The residuals are heteroskedastic (Breusch-Pagan test up to lag {max_lag}).")
        flag = True

    if t_pvalue <= 0.05:
        #print(f"The residuals have a mean significantly different from zero (T-test).")
        flag = True

    #if residuals are not white noise, the Newey-West standard errors are used
    if flag:
        #print("\nThe residuals are not white noise.")   
        nw_cov = cov_hac(model_results)
        nw_se = np.sqrt(np.diag(nw_cov))
        
        model_results.cov_params_default = nw_cov
        model_results.bse = nw_se
        
    return model_results
    

def analysis(file_path, storage, study_name):
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
        
        #making the data stationary
        stationary_data = stationary(index_data)
        stationary_data = stationary_data.reset_index()
        stationary_data = stationary_data.rename(columns={'index': 'Date'})

        p_max = q_max = 30

        train_data, test_data = train_test_split(stationary_data, test_size = 0.2, shuffle = False)

        #setting up Optuna's storage and study
        index_storage = storage + '_' + index_name + '_study.db'
        index_study = study_name + '_' + index_name

        #loading or creating the corresponding Optuna study
        try:
            study = optuna.load_study(study_name=index_study, storage=index_storage)
        except KeyError:
            study = optuna.create_study(direction='minimize', study_name=index_study, storage=index_storage)
        
        ##tuning the hyperparameters by specifying the amount of trials
        #study.optimize(lambda trial: objective(trial, train_data, p_max, q_max), n_trials=50)
        
        print("Best parameters: ", study.best_params)
        print("Best score: ", study.best_value)
        
        #initializing the model with the best found parameters
        best_params = study.best_params
        best_order = (best_params['p'], 0, best_params['q'])
        best_trend = best_params['trend']

        #checking whether the model is white noise
        _, arima_results = forecast_values(train_data, best_order, best_trend)
        arima_results = white_noise(arima_results)

        forecasts = []
        actual_values = []

        #making rolling window predictions
        for i in range(len(test_data)):
            train_set = pd.concat([train_data, test_data.iloc[:i]])
            _, arima_results = forecast_values(train_set, best_order, best_trend)

            #checking whether the white noise assumptions are not violated
            arima_results = white_noise(arima_results)
            forecast = arima_results.forecast(steps = 1).iloc[0]
    
            forecasts.append(forecast)
            actual_values.append(test_data.iloc[i]['Close'])
        
        mse = mean_squared_error(actual_values, forecasts)
        print(f"INDEX: {index_name}. Best order: {best_order}. Best trend: {best_trend}")


file_path = r'datasets\numeric_data.csv'
study_name = 'arima_optimization'
storage = 'sqlite:///datasets/optimize/arima/arima'
analysis(file_path, storage, study_name)

