import pandas as pd
import numpy as np
import collections
import hts
from pmdarima import auto_arima
from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from arch import arch_model
import datetime as dt
from datetime import timedelta
from datetime import datetime
from collections import OrderedDict

funct = lambda x: np.floor(abs(x)) if ((abs(x) - np.floor(abs(x))) <= 0.5) else np.ceil(abs(x))
funct1 = lambda x: 1 if x == 0 else x

def train_test_data(df: pd.DataFrame):
    """ This function splits the data into train and test datasets in the ratio of 80:20
    
    Args : 
        df( Pandas DataFrame ) :  it's a Pandas Dataframe of the time series with all the columns of the hierarchial time series

    Returns :
        train_data(Pandas DataFrame) : Time series data of all the nodes of hierarchial time series with 80 percent of total data
        test_data(Pandas DataFrame) : Time series data of all the nodes of hierarchial time series with 20 percent of total data

    Erros :
        Value Error : The Input pandas DataFrame should have atleast 12 rows, if not it will raise an error
    """
    if df.shape[0] >= 12 :
        train_sample_size = np.int(np.floor(df.shape[0]*0.8))
        train_data = df.iloc[:train_sample_size,: ]
        test_data = df.iloc[train_sample_size : df.shape[0], :]
        return train_data, test_data
    else :
        raise ValueError(" The number of rows of the input dataframe should be atleast 12 ")

def define_tree(temp_train: pd.DataFrame, hier : OrderedDict()):
    """This Function creates key variables for forecasting  hierarchial time series  like summarising matrix.

    Args :
        temp_train (Pandas DataFrame) : its the train data derived in the train_test_data function 
        hier(Ordered Dictionary) : its the heirarchial structure of the heirarchical time series

    Returns :
        tree( hierarchy tree) : it prints the hierarchical tree of the time series data
        sum_mat(numpy array) : its summarising matrix
        sum_mat_labels(list ) : its a list of the columns of the hierarchical time series data

    """
    tree = hts.hierarchy.HierarchyTree.from_nodes(hier, temp_train, root='total')
    sum_mat, _ = hts.functions.to_sum_mat(tree)
    sum_mat_labels = temp_train.columns
    return tree, sum_mat, sum_mat_labels
    
def time_series_scores(forecast, actual):
    """This Function Calculates the MAPE , RMSE, MAE, FIRST WEEK MAPE.
    
    Args :
        forecast(Pandas Series/ numpy array/ list) : The forecasting Values of a time series 
        actual(Pandas Series/ numpy array/ list) : The actual Values of a time series 

    Returns :
        mape(int) : Mean Absolute Percentage Error calculated using forecast and actual
        rmse(float) :  Root Mean Square Error  calculated using forecast and actual
        mae(int) : Mean Absolute Percenatage Error calculated using forecast and actual
        first_wk_mape : Mean Absolute Percentage Error calculated using forecast and actual of first value(n+1 value)
    """
    mape = np.floor(np.mean(np.abs(forecast - actual) / np.abs(actual.apply(funct1))) * 100)
    rmse = np.sqrt(((forecast - actual) ** 2).mean())
    mae = np.mean(np.abs(forecast - actual))
    first_wk_mape = 100 * np.abs(forecast[0] - actual[0]) / funct1(np.abs(actual[0]))
    return mape, rmse, mae, first_wk_mape


def forecast_models(temp_train, temp_test, tree, sum_mat, sum_mat_labels, exogenus_variables,m):

    # With ARIMA Model
    print("starting ARIM Model")
    forecasts_ARIM = pd.DataFrame()
    forecasts_ARIM_1 = pd.DataFrame()
    for col in sum_mat_labels:
        try:
            try:
                stepwise_model = auto_arima(temp_train[col].values, exogenous=temp_train[exogenus_variables], stepwise=True,error_action='ignore', seasonal=True,  start_P=1, D=None, start_Q=1,   max_p= 2, max_q= 2, max_d= 2, max_P=2, max_D=2, max_Q=2, n_fits= 5).fit(temp_train[col].values)                                        
                # print("with exogeneous-m-12")
                fcst = stepwise_model.predict(n_periods= len(temp_test), start=temp_test.index[0], end=temp_test.index[-1], exog=temp_test[exogenus_variables])
            except:
                stepwise_model = auto_arima(temp_train[col].values, stepwise=True, m=m, error_action='ignore', seasonal=True,   start_P=1, D=0, start_Q=1, max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2, random= True, n_fits= 10).fit(temp_train[col].values)
                # print(col , "without exogeneous-m-12")
                fcst = stepwise_model.predict(n_periods= len(temp_test))
        except:
            try:
                try:
                    stepwise_model = auto_arima(temp_train[col].values, exogenous=temp_train[exogenus_variables], stepwise=True,  error_action='ignore', seasonal=True, m=12, start_P=1, D=None, start_Q=1,   max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1,max_Q=2, n_fits= 10).fit(temp_train[col].values)

                    # print("with exogeneous-m-6")
                    fcst = stepwise_model.predict(n_periods=len(temp_test), start=temp_test.index[0],end=temp_test.index[-1], exog=temp_test[exogenus_variables])
                except:
                    stepwise_model = auto_arima(temp_train[col].values, stepwise=True, m=12, error_action='ignore', seasonal=True, start_P=1, D=None, start_Q=1, max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2, random= True, n_fits= 10).fit(temp_train[col].values)
                    # print("without exogeneous-m-6")
                    fcst = stepwise_model.predict(n_periods= len(temp_test))
            except:
                try:
                    try:
                        stepwise_model = auto_arima(temp_train[col].values, exogenous=temp_train[exogenus_variables], stepwise=True, error_action='ignore', seasonal=True,  m= 4, start_P=1, D=None, start_Q=1,   max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2,  n_fits= 10).fit(temp_train[col].values)

                        # print("with exogeneous-m-1")
                        fcst = stepwise_model.predict(n_periods=len(temp_test), start=temp_test.index[0],end=temp_test.index[-1], exog=temp_test[exogenus_variables])
                    except:
                        stepwise_model = auto_arima(temp_train[col].values, stepwise=True, error_action='ignore', seasonal=True, m=4,  start_P=1, D=None, start_Q=1, max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2, random= True, n_fits= 10).fit(temp_train[col].values)
                        # print("without exogeneous-m-1")
                        fcst = stepwise_model.predict(n_periods= len(temp_test))
                except:
                    fcst = np.random.randint(low = 0,high=2,size=len(temp_test))
        forecasts_ARIM[col] = fcst
        # print("ending ARIMA:", col)
    forecasts_ARIM.index = temp_test.index
    pred_dict_ARIM = collections.OrderedDict()
    for label in forecasts_ARIM.columns:
        if np.all(np.array(forecasts_ARIM[label].values) != 0):
            forecasts_ARIM_1[label] = forecasts_ARIM[label]
        else :
            forecasts_ARIM_1[label] = np.random.randint(low = 0,high=2,size=len(temp_test))
        pred_dict_ARIM[label] = pd.DataFrame(data=abs(forecasts_ARIM_1[label].values), columns=['yhat'])
    revised_ARIM = hts.functions.optimal_combination(pred_dict_ARIM, sum_mat, method='OLS', mse={})
    revised_forecasts_ARIM = pd.DataFrame(data=revised_ARIM[0:, 0:], index=forecasts_ARIM.index,
                                          columns=forecasts_ARIM.columns)
    mape_ARIM = pd.DataFrame(columns=forecasts_ARIM.columns)
    rmse_ARIM = pd.DataFrame(columns=temp_train.columns)
    mae_ARIM = pd.DataFrame(columns=temp_train.columns)
    first_wk_mape_ARIM = pd.DataFrame(columns=temp_train.columns)
    for col in revised_forecasts_ARIM.columns:
        revised_forecasts_ARIM[col] = revised_forecasts_ARIM[col].apply(funct)
        mape_ARIM.loc[0, col], rmse_ARIM.loc[0, col], mae_ARIM.loc[0, col], first_wk_mape_ARIM.loc[0, col] = time_series_scores(revised_forecasts_ARIM[col], temp_test[col])   



    # With Holt's Winter Smooth Exponential Model
    print("starting HWSE Model")
    HWSE_model = pd.DataFrame(columns=sum_mat_labels)
    forecasts_HWSE = pd.DataFrame(columns=sum_mat_labels)
    forecasts_HWSE_1 = pd.DataFrame(columns=sum_mat_labels)
    for col in sum_mat_labels:
        # print('Starting HWSE:', col)
        try:
            HWSE_model = HWES(temp_train[col], seasonal_periods=52, trend='add', seasonal='add').fit()
            forecasts_HWSE[col]=HWSE_model.forecast(steps=len(temp_test))
        except:
            try:
                HWSE_model = HWES(temp_train[col], seasonal_periods=12, trend='add', seasonal='add').fit()
                forecasts_HWSE[col]=HWSE_model.forecast(steps=len(temp_test))
            except:
                try:
                    HWSE_model = HWES(temp_train[col], seasonal_periods=4, trend='add', seasonal='add').fit()
                    forecasts_HWSE[col]=HWSE_model.forecast(steps=len(temp_test))
                except:
                    forecasts_HWSE[col] = np.random.randint(low = 0,high=2,size=len(temp_test))

        forecasts_HWSE[col] = forecasts_HWSE[col].fillna(0)
        # print('Ending HWSE:', col)
    forecasts_HWSE.index = temp_test.index
    pred_dict_HWSE = collections.OrderedDict()
    for label in forecasts_HWSE.columns:
        if  np.all(np.array(forecasts_HWSE[label].values))  :
            forecasts_HWSE_1[label] = forecasts_HWSE[label]
        else :
            forecasts_HWSE_1[label]  =  np.random.randint(low = 0,high=2,size=len(temp_test))
        pred_dict_HWSE[label] = pd.DataFrame(data=abs(forecasts_HWSE_1[label].values), columns=['yhat'])
    revised_HWSE = hts.functions.optimal_combination(pred_dict_HWSE, sum_mat, method='OLS', mse={})
    revised_forecasts_HWSE = pd.DataFrame(data=revised_HWSE[0:, 0:], index=forecasts_HWSE.index, columns=sum_mat_labels)
    mape_HWSE = pd.DataFrame(columns=forecasts_HWSE.columns)
    rmse_HWSE = pd.DataFrame(columns=forecasts_HWSE.columns)
    mae_HWSE = pd.DataFrame(columns=forecasts_HWSE.columns)
    first_wk_mape_HWSE = pd.DataFrame(columns=forecasts_HWSE.columns)
    for col in forecasts_HWSE.columns:
        revised_forecasts_HWSE[col] = revised_forecasts_HWSE[col].apply(funct)
        mape_HWSE.loc[0, col], rmse_HWSE.loc[0, col], mae_HWSE.loc[0, col], first_wk_mape_HWSE.loc[
            0, col] = time_series_scores(revised_forecasts_HWSE[col], temp_test[col])

    # with Fb Prophet Model 
    print("starting Propeht Model")
    PROP_model = pd.DataFrame(columns=sum_mat_labels)
    forecasts_PROP = pd.DataFrame(columns=sum_mat_labels)
    forecasts_PROP_1 = pd.DataFrame(columns=sum_mat_labels)
    for col in sum_mat_labels:
        temp_prop_train = pd.DataFrame()
        temp_prop_train['ds'] = temp_train.index
        temp_prop_train['y'] = temp_train[col].values
        temp_prop_test = pd.DataFrame()
        temp_prop_test['ds'] = temp_test.index
        for exogenous_col in exogenus_variables :
            temp_prop_train[exogenous_col] = temp_train[exogenous_col].values
            temp_prop_test[exogenous_col] = temp_test[exogenous_col].values
        try:
            PROP_model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False, interval_width=0.80)
            PROP_model = PROP_model.add_seasonality(name='weekly', period=7, fourier_order=1, prior_scale=0.02).fit(temp_prop_train)
            for exogenous_col in exogenus_variables :
                PROP_model = PROP_model.add_regressor(exogenous_col)
            forecasts_PROP[col] = PROP_model.predict(temp_prop_test)['yhat'].values
            print("prop successful")
        except:
            PROP_model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False,
                                 interval_width=0.80)
            PROP_model = PROP_model.add_seasonality(name='weekly', period=7, fourier_order=1, prior_scale=0.02).fit(temp_prop_train)
            forecasts_PROP[col] = PROP_model.predict(temp_prop_test)['yhat'].values
    forecasts_PROP.index = temp_test.index
    pred_dict_PROP = collections.OrderedDict()
    for label in forecasts_PROP.columns:
        if np.all(np.array(forecasts_PROP[label].values) ) :
            forecasts_PROP_1[label] = np.array(forecasts_PROP[label])
        else :
            forecasts_PROP_1[label]  =  np.random.randint(low = 0,high=2,size=len(temp_test))
        pred_dict_PROP[label] = pd.DataFrame(data=abs(forecasts_PROP_1[label].values), columns=['yhat'])
    revised_PROP = hts.functions.optimal_combination(pred_dict_PROP, sum_mat, method='OLS', mse={})
    revised_forecasts_PROP = pd.DataFrame(data=revised_PROP[0:, 0:], index=forecasts_PROP.index,
                                          columns=forecasts_PROP.columns)
    mape_PROP = pd.DataFrame(columns=forecasts_PROP.columns)
    rmse_PROP = pd.DataFrame(columns=forecasts_PROP.columns)
    mae_PROP = pd.DataFrame(columns=forecasts_PROP.columns)
    first_wk_mape_PROP = pd.DataFrame(columns=forecasts_PROP.columns)
    for col in forecasts_PROP.columns:
        revised_forecasts_PROP[col] = revised_forecasts_PROP[col].apply(funct)
        mape_PROP.loc[0, col], rmse_PROP.loc[0, col], mae_PROP.loc[0, col], first_wk_mape_PROP.loc[
            0, col] = time_series_scores(revised_forecasts_PROP[col], temp_test[col])
    # print(revised_forecasts_PROP.head())

    # With ARCH Model 

    forecasts_ARCH = pd.DataFrame(columns=sum_mat_labels)
    forecasts_ARCH_1 = pd.DataFrame(columns=sum_mat_labels)
    for col in sum_mat_labels:
        try:
            try:
                stepwise_model = arch_model(temp_train[[col]],  vol='ARCH',  p=1, lags=12).fit()
                fcst = stepwise_model.forecast(horizon=len(temp_test)).mean.values[-1, :]
                # print("ARCH_fcst:", fcst)
            except:
                try:
                    stepwise_model = arch_model(temp_train[[col]], vol='ARCH', p=1, lags=6).fit()
                    fcst = stepwise_model.forecast(horizon=len(temp_test)).mean.values[-1, :]
                    # print("ARCH_fcst:", fcst)
                except:
                    try:
                        stepwise_model = arch_model(temp_train[[col]], vol='ARCH',  p=1, lags=3).fit()
                        fcst = stepwise_model.forecast(horizon=len(temp_test)).mean.values[-1, :]
                        # print("ARCH_fcst:", fcst)
                    except:
                        stepwise_model = arch_model(temp_train[[col]], vol='ARCH',  p=1, lags=1).fit()
                        fcst = stepwise_model.forecast(horizon=len(temp_test)).mean.values[-1, :]
                        # print("ARCH_fcst:", fcst)
        except:
            print("ARCH_fcst: failed")
            fcst = np.random.randint(low = 0,high=2,size=len(temp_test))
        forecasts_ARCH[col] = fcst
        # print("ending ARCH:", col)
    forecasts_ARCH.index = temp_test.index
    # print("forecasts_ARCH:", forecasts_ARCH)
    pred_dict_ARCH = collections.OrderedDict()
    for label in forecasts_ARCH.columns:
        if np.all(np.array(forecasts_ARCH[label].values)) :
            forecasts_ARCH_1[label] = np.array(forecasts_ARCH[label])
        else :
            forecasts_ARCH_1[label]  =  np.random.randint(low = 0,high=2,size=len(temp_test))
        pred_dict_ARCH[label] = pd.DataFrame(data=abs(forecasts_ARCH_1[label].values), columns=['yhat'])
    revised_ARCH = hts.functions.optimal_combination(pred_dict_ARCH, sum_mat, method='OLS', mse={})
    revised_forecasts_ARCH = pd.DataFrame(data=revised_ARCH[0:, 0:], index=forecasts_ARCH.index,
                                          columns=forecasts_ARCH.columns)
    mape_ARCH = pd.DataFrame(columns=forecasts_ARCH.columns)
    rmse_ARCH = pd.DataFrame(columns=temp_train.columns)
    mae_ARCH = pd.DataFrame(columns=temp_train.columns)
    first_wk_mape_ARCH = pd.DataFrame(columns=temp_train.columns)
    for col in revised_forecasts_ARCH.columns:
        revised_forecasts_ARCH[col] = revised_forecasts_ARCH[col].apply(funct)
        mape_ARCH.loc[0, col], rmse_ARCH.loc[0, col], mae_ARCH.loc[0, col], first_wk_mape_ARCH.loc[
            0, col] = time_series_scores(revised_forecasts_ARCH[col], temp_test[col])
    print("ending the forecast models function")
    return  revised_forecasts_ARIM, revised_forecasts_HWSE, revised_forecasts_PROP, revised_forecasts_ARCH,  mape_ARIM, mape_HWSE, mape_PROP, mape_ARCH


def ensemble_fcst(temp_test, revised_forecasts_ARIM, revised_forecasts_HWSE, revised_forecasts_PROP, revised_forecasts_ARCH,  mape_ARIM,
                  mape_HWSE, mape_PROP, mape_ARCH, sum_mat_labels):
    print("starting ensemble forecasting ")              
    select_model = pd.DataFrame(columns=sum_mat_labels)
    forecasts_ENSE = pd.DataFrame(columns=sum_mat_labels)
    models_list = ["ARIM", "HWSE", "PROP", "ARCH"]
    revised_forecasts_ARIM['approach'] = "ARIM"
    revised_forecasts_HWSE['approach'] = "HWSE"
    revised_forecasts_PROP['approach'] = "PROP"
    revised_forecasts_ARCH['approach'] = "ARCH"
    mape_ARIM['approach'] = "ARIM"
    mape_HWSE['approach'] = "HWSE"
    mape_PROP['approach'] = "PROP"
    mape_ARCH['approach'] = "ARCH"
    # print("mape_ARIM", mape_ARIM.shape)
    # print("mape_HWSE", mape_HWSE.shape)
    # print("mape_PROP", mape_PROP.shape)
    # print("mape_ARCH", mape_ARCH.shape)
    mape_data = ((mape_ARIM.append(mape_HWSE)).append(mape_PROP)).append(mape_ARCH)
    # print("mape_data", mape_data.transpose())
    for col in sum_mat_labels:
        select_model.loc[0, col] = mape_data[mape_data[col] == mape_data[col].min()]['approach'].to_numpy()[0]
        # select_model[col] = select_model[col].to_string()[6:10]
    select_model = select_model.transpose().reset_index()
    select_model.columns = ['area/dealer', 'APPROACH']
    list_ARIMA = select_model[select_model['APPROACH'] == "ARIM"]['area/dealer'].to_list()
    list_HWSE = select_model[select_model['APPROACH'] == "HWSE"]['area/dealer'].to_list()
    list_PROP = select_model[select_model['APPROACH'] == "PROP"]['area/dealer'].to_list()
    list_ARCH = select_model[select_model['APPROACH'] == "ARCH"]['area/dealer'].to_list()
    print("list of ARIMA", list_ARIMA)
    print("list of HWSE", list_HWSE)
    print("list of PROP", list_PROP)
    print("list of ARCH", list_ARCH)
    print("ending the ensemble forecasting function")
    return list_ARIMA, list_HWSE, list_PROP, list_ARCH


def forecast_refit(ts_data_2, list_ARIMA,  list_HWSE,  list_PROP, list_ARCH, tree, sum_mat, sum_mat_labels, fcst_input_data,exogenus_variables):
#def forecast_refit(ts_data_2, list_ARIMA, list_HWSE,  tree, sum_mat, sum_mat_labels,fcst_input_data):
    print("starting forecast refitting ") 
    forecasts_ARIM = pd.DataFrame(columns=list_ARIMA)
    fcst_start_date= fcst_input_data.index[0]
    print("fcst_start_date", fcst_start_date)
    fcst_end_date=  fcst_input_data.index[-1]
    print("fcst_end_date",fcst_end_date)
    if len(list_ARIMA) > 0:
        print("starting the refit for arima ")
        stepwise_model = pd.DataFrame(columns=list_ARIMA)
        forecasts_ARIM = pd.DataFrame()
        for col in list_ARIMA:
                try:
                    try:
                        stepwise_model = auto_arima(ts_data_2[col].values, exogenous=ts_data_2[exogenus_variables], stepwise=True,error_action='ignore', seasonal=True,  start_P=1, D=None, start_Q=1,  max_p= 2, max_q= 2, max_d= 2, max_P=2, max_D=2, max_Q=2).fit(ts_data_2[col].values)
                        print("with exo exogeneous-m-12")
                        fcst = stepwise_model.predict(n_periods= len(fcst_input_data), start=fcst_start_date, end=fcst_end_date, exog=fcst_input_data[exogenus_variables])
                    except:
                        stepwise_model = auto_arima(ts_data_2[col].values, stepwise=True, error_action='ignore', seasonal=True, m=52,  start_P=1, D=None, start_Q=1,   max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2).fit(ts_data_2[col].values)
                        print(col, "without exogeneous-m-12")
                        fcst = stepwise_model.predict(n_periods=len(fcst_input_data))
                except:
                    try:
                        try:
                            stepwise_model = auto_arima(ts_data_2[col].values, exogenous=ts_data_2[exogenus_variables], stepwise=True, error_action='ignore', seasonal=True, m=12,  start_P=1, D=None, start_Q=1,   max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2).fit(ts_data_2[col].values)
                            print("with exogeneous-m-6")
                            fcst = stepwise_model.predict(n_periods= len(fcst_input_data), start=fcst_start_date, end=fcst_end_date,exog=fcst_input_data[exogenus_variables])
                        except:
                            stepwise_model = auto_arima(ts_data_2[col].values, stepwise=True, error_action='ignore', seasonal=True, m=12, start_P=1, D=None, start_Q=1,   max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2).fit(ts_data_2[col].values)
                            print("without exogeneous-m-6")
                            fcst = stepwise_model.predict(n_periods=len(fcst_input_data))
                    except:
                        try:
                            try:
                                stepwise_model = auto_arima(ts_data_2[col].values, exogenous=ts_data_2[exogenus_variables], stepwise=True, error_action='ignore', seasonal=True, m=4, start_P=1, D=None, start_Q=1,   max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2).fit(ts_data_2[col].values)
                                print("with exogeneous-m-1")
                                fcst = stepwise_model.predict(n_periods= len(fcst_input_data), start=fcst_start_date, end=fcst_end_date, exog=fcst_input_data[exogenus_variables])
                            except:
                                stepwise_model = auto_arima(ts_data_2[col].values, stepwise=True, m=4, error_action='ignore', seasonal=True,   start_P=1, D=None, start_Q=1,   max_p= 3, max_q= 3, max_d= 3, max_P=2, max_D=1, max_Q=2).fit(ts_data_2[col].values)
                                print("without exogeneous-m-1")
                                fcst = stepwise_model.predict(n_periods=len(fcst_input_data))
                        except:
                            pass
                            fcst= [0]*len(fcst_input_data)
                forecasts_ARIM[col] = fcst
                forecasts_ARIM.index = np.array(fcst_input_data.index)
        else:
            pass

          
          
          
    HWSE_model = pd.DataFrame(columns=list_HWSE)
    forecasts_HWSE = pd.DataFrame(columns=list_HWSE)
    if len(list_HWSE) > 0:
        print("starting the refit for HWSE")
        for col in list_HWSE:
            try:
                HWSE_model = HWES(ts_data_2[col], seasonal_periods=52, trend='add', seasonal='add').fit()
                forecasts_HWSE[col] = HWSE_model.forecast(steps=len(fcst_input_data))
            except:
                try:
                    HWSE_model = HWES(ts_data_2[col], seasonal_periods=12, trend='add', seasonal='add').fit()
                    forecasts_HWSE[col] = HWSE_model.forecast(steps=len(fcst_input_data))
                except:
                    try:
                        HWSE_model = HWES(ts_data_2[col], seasonal_periods=4, trend='add', seasonal='add').fit()
                        forecasts_HWSE[col] = HWSE_model.forecast(steps=len(fcst_input_data))
                    except:
                        pass
                        forecasts_HWSE[col] =[0]* len(fcst_input_data)
            forecasts_HWSE[col] = forecasts_HWSE[col].fillna(0)
        forecasts_HWSE.index = np.array(fcst_input_data.index)
    else:
        pass

    PROP_model = pd.DataFrame(columns=list_PROP)
    forecasts_PROP = pd.DataFrame(index=np.array(fcst_input_data.index),  columns=list_PROP)
    if len(list_PROP) > 0:
        print("starting the refit for PROP ")
        for col in list_PROP:
            temp_prop_train = pd.DataFrame()
            temp_prop_train['ds'] = np.array(ts_data_2.index)
            temp_prop_train['y'] = ts_data_2[col].values
            temp_prop_test = pd.DataFrame()
            temp_prop_test['ds'] = np.array(fcst_input_data.index)
            for exogenous_col in exogenus_variables :
                temp_prop_train[exogenous_col] = ts_data_2[exogenous_col].values
                temp_prop_test[exogenous_col] = fcst_input_data[exogenous_col].values
            
            # print("temp_prop_train", temp_prop_train)
            # print("temp_prop_test", temp_prop_test)
            # print("temp_prop_train", temp_prop_train.dtypes)
            # print("temp_prop_test", temp_prop_test.dtypes)
            try:
                PROP_model=Prophet(daily_seasonality=False,yearly_seasonality=True,weekly_seasonality=False,interval_width=0.80)
                PROP_model = PROP_model.add_seasonality(name='yearly', period=12, fourier_order=5, prior_scale=0.02).fit(temp_prop_train)
                for exogenous_col in exogenus_variables :
                    PROP_model = PROP_model.add_regressor(exogenous_col)
                    
                forecasts_PROP[col] = PROP_model.predict(temp_prop_test)['yhat'].values
                print("prop successful")
            except:
                PROP_model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False,interval_width=0.80)
                PROP_model = PROP_model.add_seasonality(name='yearly', period=12, fourier_order=5, prior_scale=0.02).fit(temp_prop_train)
                forecasts_PROP[col] = PROP_model.predict(temp_prop_test)['yhat'].values

    else:
        pass
    forecasts_PROP.index = np.array(fcst_input_data.index)
    # print(forecasts_PROP)

    forecasts_ARCH = pd.DataFrame(columns=list_ARCH)
    fcst_start_date = fcst_input_data.index[0]
    print("fcst_start_date", fcst_start_date)
    fcst_end_date = fcst_input_data.index[-1]
    if len(list_ARCH) > 0:
        print("starting the refit for ARCH ")
        stepwise_model = pd.DataFrame(columns=list_ARCH)
        forecasts_ARCH = pd.DataFrame()
        for col in list_ARCH:
            try:
                try:
                    stepwise_model = arch_model(ts_data_2[col].values,  mean="AR", vol= "ARCH", p=1, lags=12).fit()
                    fcst = stepwise_model.forecast(horizon=len(fcst_input_data)).mean.values[-1, :]
                except:
                    try:
                        stepwise_model = arch_model(ts_data_2[col].values,   mean="AR", vol= "ARCH", p=1,lags=6).fit()
                        fcst = stepwise_model.forecast(horizon=len(fcst_input_data)).mean.values[-1, :]
                        print("ARCH_fcst_6", fcst)
                    except:
                        try:
                            stepwise_model = arch_model(ts_data_2[col].values,  mean="AR", vol= "ARCH",p=1, lags=3).fit()
                            fcst = stepwise_model.forecast(horizon=len(fcst_input_data)).mean.values[-1, :]
                        except:
                            stepwise_model = arch_model(ts_data_2[col].values,  mean='AR', vol= "ARCH", p=1).fit()
                            fcst = stepwise_model.forecast(horizon=len(fcst_input_data)).mean.values[-1, :]
            except:
                print("ARCH failed")
                fcst = [0] * len(fcst_input_data)
            forecasts_ARCH[col] = fcst
            forecasts_ARCH.index = np.array(fcst_input_data.index)
        else:
            pass
    print("ending the forecasting refitting function")
    # print(forecasts_ARCH.shape, forecasts_ARIM.shape ,  forecasts_HWSE.shape, forecasts_PROP.shape)
    return forecasts_ARIM, forecasts_HWSE, forecasts_PROP, forecasts_ARCH

def forecast_output(final_forecasts_ARIM, final_forecasts_HWSE, final_forecasts_PROP, final_forecasts_ARCH, sum_mat, sum_mat_labels, fcst_input_data):
    print(" starting the last function")
    forecast_output = pd.concat([final_forecasts_ARIM, final_forecasts_HWSE, final_forecasts_PROP, final_forecasts_ARCH], axis=1)
    forecast_output_1 = forecast_output.copy()
    print("forecast_output:", forecast_output.columns)
    pred_dict = collections.OrderedDict()
    for label in sum_mat_labels:
        if np.all( np.array(forecast_output[label]) ):
            pass
        else :
            forecast_output_1[label] = np.random.randint(low = 0,high=2,size=len(forecast_output))
        pred_dict[label] = pd.DataFrame(data=abs(forecast_output_1[label].values), columns=['yhat'])
    revised_val = hts.functions.optimal_combination(pred_dict, sum_mat, method='OLS', mse={})
    revised_forecasts = pd.DataFrame(data=revised_val[0:,0:], index=forecast_output.index, columns=sum_mat_labels)
    revised_forecasts= revised_forecasts.fillna(0)
    for col in sum_mat_labels:
        revised_forecasts[col] = revised_forecasts[col].apply(funct)
    revised_forecasts.index= np.array(fcst_input_data.index)
    print(" ending the last function")
    return revised_forecasts

def hts_forecast_function(df : pd.DataFrame , hier : dict , exogenus_variables : list , predictable_variables : list , fcst_input_data : pd.DataFrame,m:int):
    # Splitting the data into train and test samples

    # create an error statement if the sum of exogenus and predcitable not equla to df variables

    train_data,test_data =    train_test_data(df)
    tree, sum_mat, sum_mat_labels =  define_tree(train_data[predictable_variables],hier) 
    revised_forecasts_ARIM, revised_forecasts_HWSE, revised_forecasts_PROP, revised_forecasts_ARCH,  mape_ARIM, mape_HWSE, mape_PROP, mape_ARCH  = \
    forecast_models(train_data, test_data, tree, sum_mat, sum_mat_labels, exogenus_variables, m)
    list_ARIMA, list_HWSE, list_PROP, list_ARCH =  ensemble_fcst(test_data, revised_forecasts_ARIM, revised_forecasts_HWSE, revised_forecasts_PROP, revised_forecasts_ARCH,  mape_ARIM,
                  mape_HWSE, mape_PROP, mape_ARCH, sum_mat_labels)
    final_forecasts_ARIM, final_forecasts_HWSE, final_forecasts_PROP, final_forecasts_ARCH = forecast_refit(df, list_ARIMA,list_HWSE,list_PROP,list_ARCH,tree,sum_mat, sum_mat_labels,fcst_input_data,exogenus_variables )

    forecasted_output = forecast_output(final_forecasts_ARIM, final_forecasts_HWSE, final_forecasts_PROP, final_forecasts_ARCH, sum_mat, sum_mat_labels, fcst_input_data)       
    return forecasted_output
    
def top_down_approach(df : pd.DataFrame, hier : dict, td_df : pd.DataFrame) :
    keys_list = set(hier.keys())
    values_list = []
    for i in hier.values():
        for j in i :
            values_list.append(j)
    leaf_list = list(set(values_list) - keys_list)
    leaf_list_with_total = leaf_list.copy()
    leaf_list_with_total.append('total')

    ratio_df = pd.DataFrame(np.zeros(len(leaf_list_with_total))).T
    ratio_df.columns = leaf_list_with_total

    for col in leaf_list_with_total :
        ratio_df[col] = np.sum(df[col])
    for col in leaf_list:
        ratio_df[col] = np.round(ratio_df[col]/ratio_df["total"],2)

# the td_df should have only one column i.e total 
    for col in leaf_list :
        td_df[col] = ratio_df.loc[0,col]*td_df['total']
        td_df[col]  = td_df[col].apply(funct)
    for hier_key in reversed(list(hier.keys())):
        td_df[hier_key] = np.sum(td_df[hier[hier_key]],axis =1)
    return td_df 

def bottom_up_approach(df : pd.DataFrame, hier : dict, bu_df : pd.DataFrame) :
    keys_list = set(hier.keys())
    values_list = []
    for i in hier.values():
        for j in i :
            values_list.append(j)
    leaf_list = list(set(values_list) - keys_list)   

    ## the number of column in bu_df should be equal to number of leaf nodes  
    for col in reversed(hier.keys()):
        bu_df.loc[col] = np.sum(bu_df[hier[col]],axis =1)       


# create a function to validate the hierarichial structure    