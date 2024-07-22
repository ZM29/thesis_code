#Only comment code and the forecast issue of using Newey West that gives better results

"""
This script performs the parameter tuning of the TVAR.
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, bds
import numpy as np
import optuna
from statsmodels.tsa.api import VAR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from tabulate import tabulate
from scipy import stats

import warnings
warnings.filterwarnings("ignore", "An unsupported index was provided and will be ignored when e.g. forecasting.")


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
    results = []
    
    for column in data.columns:
        series = data[column]

        #NAs need to be dropped to ensure results 
        result = adfuller(series.dropna())

        #saving the results for the tables used in the study
        results.append([column, result[0], result[1]])
        
        #results[1] is the P-value of the test
        if result[1] > 0.05:
            diff_series = stationary(pd.DataFrame(series.diff().dropna(), columns = [column]))
            stationary_data = pd.concat([stationary_data, diff_series], axis = 1)

            #saving the results for the tables used in the study
            diff_result = adfuller(diff_series.dropna())
            results.append([f"{column} (differenced)", diff_result[0], diff_result[1]])
        else:
            stationary_data = pd.concat([stationary_data, series], axis = 1)

    ##this prints the tests in the table format used in the study
    # headers = ["Column", "Test Statistic", "P-value"]
    # table = tabulate(results, headers, tablefmt = "latex", floatfmt = ".3f")
    #print(table)

    return stationary_data

def linearity_test(data):
    """
    Function to check whether the data is linear by using the BDS test.
    """
    linear_columns = []
    non_linear_columns = []
    results = []

    #looping over the columns and performing the BDS test
    for column in data.columns:
        series = data[column]

        #perform BDS test and save the results
        bds_result = bds(series)
        results.append([column, bds_result[0], bds_result[1]])

        #results[1] is the P-value of the test
        if bds_result[1] > 0.05:
            linear_columns.append(column)
        else:
            non_linear_columns.append(column)

    # ##this prints the tests in the table format used in the study
    # headers = ["Column", "Test Statistic", "P-value"]
    # table = tabulate(results, headers, tablefmt = "latex", floatfmt = ".3f")
    #print(table)


def newey_west_covariance(residuals, lags):
    """
    Function to compute the Newey-West covariance matrix for the HAC standard errors.
    """
    first = residuals.shape[0]
    num_vars = residuals.shape[1]
    cov_matrix = np.zeros((num_vars, num_vars))

    for i in range(num_vars):
        for j in range(num_vars):
            for l in range(lags + 1):
                if l == 0:
                    cov_matrix[i, j] += np.sum(residuals.iloc[:, i] * residuals.iloc[:, j])
                else:
                    #Bartlett kernel weight is used due to its simplicity
                    weight = 1 - l / (lags + 1)  
                    cov_matrix[i, j] += np.sum(residuals.iloc[l:, i] * residuals.iloc[:-l, j]) * weight
                    cov_matrix[i, j] += np.sum(residuals.iloc[:-l, i] * residuals.iloc[l:, j]) * weight

    cov_matrix /= first
    return cov_matrix



def regimes(data, threshold_var, max_regimes = 10):
    """
    Function to determine the number of regimes by using K-means.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[threshold_var])
    
    inertia = []
    #computing the inertia for every cluster
    for k in range(2, max_regimes + 2):
        kmeans = KMeans(n_clusters = k, n_init = 10, random_state = 29)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    #plotting the elbow plot
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, max_regimes + 1), inertia, marker = 'o')
    plt.xlabel('Number of Clusters', fontsize = 24) 
    plt.ylabel('Inertia', fontsize = 24) 
    plt.xticks(range(1, max_regimes + 1), fontsize = 24) 
    plt.yticks(fontsize = 24)  
    plt.tight_layout()
    plt.show()


TREND_CHOICES = ["n","c"]

def objective(trial, train_data):
    """
    Function for Optuna to optimize hyperparameters of the model.
    """
    try:
        #defining the optimizing range of the hyperparameters
        trend_index = trial.suggest_int('trend_index', 0, len(TREND_CHOICES) - 1)
        trend = TREND_CHOICES[trend_index]
        threshold = trial.suggest_float('threshold', 0.235, 0.48)

        #setting up the splits
        splits = TimeSeriesSplit(n_splits = 3)

        for train_index, val_index in splits.split(train_data):
            train_set = train_data.iloc[train_index]
            break

        #breaking up the training data into 2 regimes
        regime1_data = train_set[train_set['Bullish'] <= threshold]
        regime2_data = train_set[train_set['Bullish'] > threshold]

        #ensuring that the p hyperparameter is not too high
        p = trial.suggest_int('p', 1, min(len(regime1_data), len(regime2_data)) - 1)

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
                forecast, _, _ = forecast_values(train_fold, p, trend, threshold)
                forecasts.append(forecast)
                actual_values.append(val_set.iloc[i]['Close'])

            #compute loss
            mse = mean_squared_error(actual_values, forecasts)
            mse_scores.append(mse)

        return np.mean(mse_scores)

    #any exception due to incompatibility is pruned and the trial is redone
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise optuna.TrialPruned()

def forecast_values(train_data, p, trend, threshold, threshold_var = 'Bullish'):
    """
    Function for forecasting values.
    """

    #splitting data into 2 regimes
    regime1_data = train_data[train_data[threshold_var] <= threshold]
    regime2_data = train_data[train_data[threshold_var] > threshold]

    model1 = VAR(regime1_data)
    results1 = model1.fit(maxlags = p, trend = trend)

    model2 = VAR(regime2_data)
    results2 = model2.fit(maxlags = p, trend = trend)

    #determining the current regime and using the corresponding model for prediction
    last_observation = train_data.iloc[-1]

    if last_observation[threshold_var] <= threshold:
        forecast = results1.forecast(regime1_data.values[-p:], steps = 1)
    else:
        forecast = results2.forecast(regime2_data.values[-p:], steps = 1) 

    return forecast[0, 0], results1, results2


def white_noise(model_fit, lag):
    """
    Function to check whether the model results are white noise and apply Newey-West standard errors.
    """
    residuals = model_fit.resid
    flag = False 

    lb_results = pd.DataFrame(columns = ['lb_test_stat', 'lb_pvalue'])
    t_test_results = pd.DataFrame(columns = ['t_test_stat', 't_test_pvalue'])

    #performing Ljung-Box test for autocorrelation and the t-test
    for column in range(residuals.shape[1]):
        lb_test_stat = acorr_ljungbox(residuals.iloc[:, column], lags = [lag], return_df = False)
        lb_results.loc[column] = [lb_test_stat['lb_stat'].iloc[0], lb_test_stat['lb_pvalue'].iloc[0]]

        t_stat, t_pvalue = stats.ttest_1samp(residuals.iloc[:, column], 0)
        t_test_results.loc[column] = [t_stat, t_pvalue]

    #performing Breusch-Pagan test for heteroskedasticity
    bp_results = pd.DataFrame(columns = ['bp_test_stat', 'bp_pvalue'])
    for i in range(model_fit.resid.shape[1]):
        bp_test_stat = het_breuschpagan(residuals.iloc[:, i], np.column_stack((np.ones(len(residuals)), residuals.iloc[:, i])))
        bp_results.loc[i] = [bp_test_stat[-2], bp_test_stat[-1]]

    ##this prints the tests in the table format used in the study
    # headers = ['Variable', 'Ljung-Box Test Statistic', 'Ljung-Box P-value', 'Breusch-Pagan Test Statistic', 'Breusch-Pagan P-value', 'T-test Test Statistic','T-test P-value']
    # table_data = []
    # for i in range(residuals.shape[1]):
    #     table_data.append([f'{columns[0][i]}', lb_results.loc[i, 'lb_test_stat'], lb_results.loc[i, 'lb_pvalue'],
    #                        bp_results.loc[i, 'bp_test_stat'], bp_results.loc[i, 'bp_pvalue'], t_test_results.loc[i, 't_test_stat'], t_test_results.loc[i, 't_test_pvalue']])
        
    # table = tabulate(table_data, headers, tablefmt = 'grid', floatfmt = '.3f')
    #print(table)

    #processing the results to check if any assumptions are violated
    lb_p_value = lb_results['lb_pvalue'].min()
    bp_p_values = bp_results['bp_pvalue'].tolist()
    t_p_values = t_test_results['t_test_pvalue'].tolist()

    #checking for any violated white noise assumption
    if lb_p_value <= 0.05:
        #print(f"\nThe residuals are serially correlated (Ljung-Box test up to lag {lag}).")
        flag = True

    if any(p <= 0.05 for p in bp_p_values):
        #print(f"\nThe residuals are heteroskedastic (Breusch-Pagan test).")
        flag = True

    if any(p <= 0.05 for p in t_p_values):
        #print(f"\nThe residuals have a mean significantly different from zero (T-test).")
        flag = True

    #if residuals are not white noise, the Newey-West standard errors are used
    if flag:
        # print("\nThe residuals are not white noise.")
        nw_cov = newey_west_covariance(residuals, lags = lag)
        model_fit.cov_params_default = nw_cov
        model_fit.stderr = np.sqrt(np.diag(nw_cov))

    return model_fit

def analysis(file_path):
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
        index_data = index_data.drop("Index", axis=1)

        ##determining the regimes by K-means clustering
        #regimes(index_data, threshold_var=['Bullish'])

        #making the data stationary
        stationary_data = stationary(index_data)
        stationary_data = stationary_data[:-1]
        stationary_data = stationary_data.reset_index()
        stationary_data = stationary_data.rename(columns={'index': 'Date'}).iloc[:,1:]
        tvar_data = stationary_data.copy()
        
        #linearity_test(stationary_data)
        train_data, test_data = train_test_split(tvar_data, test_size = 0.2, shuffle = False)

        #setting up Optuna's storage and study
        index_storage = storage + '_' + index_name + '_study.db'
        index_study = study_name + '_' + index_name

        #loading or creating the corresponding Optuna study
        try:
            study = optuna.load_study(study_name=index_study, storage=index_storage)
        except KeyError:
            study = optuna.create_study(direction='minimize', study_name=index_study, storage=index_storage)

        n_trials = 50
        n_completed_trials = 0

        ##making sure that the amount of specified trials is achieved due to stability issues
        # while n_completed_trials < n_trials:
        #     try:
        #         study.optimize(lambda trial: objective(trial, train_data), n_trials = n_trials - n_completed_trials)
        #     except optuna.TrialPruned:
        #         pass
        #     n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        # print(f"Completed {n_completed_trials} trials for index: {index_name}")
        print("Best parameters: ", study.best_params)
        print("Best score: ", study.best_value)

        #initializing the model with the best found parameters
        best_params = study.best_params
        best_p = best_params['p']
        trend_index = best_params['trend_index']
        best_trend = TREND_CHOICES[trend_index]
        best_threshold = best_params['threshold']

        #checking whether the models are white noise
        _, var_model1, var_model2 = forecast_values(train_data, best_p, best_trend, best_threshold)
        var_model1 = white_noise(var_model1, best_p)
        var_model2 = white_noise(var_model2, best_p)

        forecasts = []
        actual_values = []

        #making rolling window predictions
        for i in range(len(test_data)):
            train_set = pd.concat([train_data, test_data.iloc[:i]])
            forecast, var_model1, var_model2 = forecast_values(train_set, best_p, best_trend, best_threshold)

            #checking whether the white noise assumptions are not violated
            var_model1 = white_noise(var_model1, best_p)
            var_model2 = white_noise(var_model2, best_p)

            regime1_data = train_data[train_data['Bullish'] <= best_threshold]
            regime2_data = train_data[train_data['Bullish'] > best_threshold]

            #determining the current regime and using the corresponding model for prediction
            last_observation = train_set.iloc[-1]

            if last_observation['Bullish'] <= best_threshold:
                forecast = var_model1.forecast(regime1_data.values[-best_p:], steps = 1)
            else:
                forecast = var_model2.forecast(regime2_data.values[-best_p:], steps = 1)

            forecasts.append(forecast[0, 0])
            actual_values.append(test_data.iloc[i]['Close'])

        mse = mean_squared_error(actual_values, forecasts)

        print(f"INDEX: {index_name}. Best p: {best_p}. Best trend: {best_trend}")


file_path = r'datasets\sentiment_numeric.csv'
study_name = 'tvar_optimization'
storage = 'sqlite:///datasets/optimize/tvar/tvar'
analysis(file_path)