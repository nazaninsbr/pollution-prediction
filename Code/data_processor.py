import numpy as np 
import datetime
from keras.preprocessing.sequence import TimeseriesGenerator

def read_data(file_path):
    ###################################
    # the fields are: pollution, dew, temp, pressure, wind_dir, wind_spd, snow, rain
    ###################################
    return np.load(file_path)

def prep_data_for_model_method_1(data, window_size):
    X, y = [], []
    for i in range(data.shape[0]):
        if i<=window_size-1:
            continue 
        else:
            X.append(data[i-(window_size):i])
            y.append([data[i][0]])
    X = np.array(X)
    y = np.array(y)
    return X, y

def prep_data_for_model_weekly(data):
    date_time_added_data = {}
    this_time = datetime.datetime(2014, 1, 1, 0, 0)
    for data_id in range(data.shape[0]):
        if not data_id == 0:
            this_time = this_time + datetime.timedelta(hours=1)
        date_time_added_data[this_time] = data[data_id]
    
    X, y = [], []
    for time_key in date_time_added_data.keys():
        try:
            all_train_data_vals = [
                date_time_added_data[time_key - datetime.timedelta(days=7)],
                date_time_added_data[time_key - datetime.timedelta(days=6)],
                date_time_added_data[time_key - datetime.timedelta(days=5)],
                date_time_added_data[time_key - datetime.timedelta(days=4)],
                date_time_added_data[time_key - datetime.timedelta(days=3)],
                date_time_added_data[time_key - datetime.timedelta(days=2)],
                date_time_added_data[time_key - datetime.timedelta(days=1)]
            ]
            X.append(all_train_data_vals)
            y.append([date_time_added_data[time_key][0]])
        except Exception:
            pass

    X = np.array(X)
    y = np.array(y)
    return X, y 

def prep_data_for_model_monthly(data):
    date_time_added_data = {}
    this_time = datetime.datetime(2014, 1, 1, 0, 0)
    for data_id in range(data.shape[0]):
        if not data_id == 0:
            this_time = this_time + datetime.timedelta(hours=1)
        date_time_added_data[this_time] = data[data_id]
    
    X, y = [], []
    for time_key in date_time_added_data.keys():
        try:
            all_train_data_vals = [
                date_time_added_data[time_key - datetime.timedelta(days=21)],
                date_time_added_data[time_key - datetime.timedelta(days=14)],
                date_time_added_data[time_key - datetime.timedelta(days=7)]
            ]
            X.append(all_train_data_vals)
            y.append([date_time_added_data[time_key][0]])
        except Exception:
            pass

    X = np.array(X)
    y = np.array(y)
    return X, y 

def prep_data_for_model_fusion(data):
    date_time_added_data = {}
    this_time = datetime.datetime(2014, 1, 1, 0, 0)
    for data_id in range(data.shape[0]):
        if not data_id == 0:
            this_time = this_time + datetime.timedelta(hours=1)
        date_time_added_data[this_time] = data[data_id]
    
    X, y = [[], [], []], []
    for time_key in date_time_added_data.keys():
        try:
            all_train_data_vals_monthly = [
                date_time_added_data[time_key - datetime.timedelta(days=21)],
                date_time_added_data[time_key - datetime.timedelta(days=14)],
                date_time_added_data[time_key - datetime.timedelta(days=7)]
            ]

            all_train_data_vals_weekly = [
                date_time_added_data[time_key - datetime.timedelta(days=7)],
                date_time_added_data[time_key - datetime.timedelta(days=6)],
                date_time_added_data[time_key - datetime.timedelta(days=5)],
                date_time_added_data[time_key - datetime.timedelta(days=4)],
                date_time_added_data[time_key - datetime.timedelta(days=3)],
                date_time_added_data[time_key - datetime.timedelta(days=2)],
                date_time_added_data[time_key - datetime.timedelta(days=1)]
            ]

            all_train_data_vals_hourly = [
                date_time_added_data[time_key - datetime.timedelta(hours=11)],
                date_time_added_data[time_key - datetime.timedelta(hours=10)],
                date_time_added_data[time_key - datetime.timedelta(hours=9)],
                date_time_added_data[time_key - datetime.timedelta(hours=8)],
                date_time_added_data[time_key - datetime.timedelta(hours=7)],
                date_time_added_data[time_key - datetime.timedelta(hours=6)],
                date_time_added_data[time_key - datetime.timedelta(hours=5)],
                date_time_added_data[time_key - datetime.timedelta(hours=4)],
                date_time_added_data[time_key - datetime.timedelta(hours=3)],
                date_time_added_data[time_key - datetime.timedelta(hours=2)],
                date_time_added_data[time_key - datetime.timedelta(hours=1)]
            ]

            X[0].append(all_train_data_vals_monthly)
            X[1].append(all_train_data_vals_weekly)
            X[2].append(all_train_data_vals_hourly)

            y.append([date_time_added_data[time_key][0]])
        except Exception:
            pass

    X = [np.array(X[0]), np.array(X[1]), np.array(X[2])]
    y = np.array(y)
    return X, y 

def prep_data_for_model_method_using_generetor(data, window_size):
    inputs, outputs = [], []
    for val in data:
        outputs.append(val[0])
        inputs.append(val)
    inputs, outputs = np.array(inputs), np.array(outputs)
    data_gen_train = TimeseriesGenerator(
        inputs,
        outputs,
        length=11,
        stride=1,
        start_index=0,
        end_index=12000,
        shuffle=False,
        reverse=False,
        batch_size = 1
    )
    
    X_train, y_train = [], []
    for this_val_ind in range(len(data_gen_train)):
        X_train.append(data_gen_train[this_val_ind][0][0])
        y_train.append(data_gen_train[this_val_ind][1])
    X_train, y_train = np.array(X_train), np.array(y_train) 

    data_gen_test = TimeseriesGenerator(
        inputs,
        outputs,
        length=11,
        stride=12,
        start_index=12000,
        end_index=15000,
        shuffle=False,
        reverse=False,
        batch_size = 1
    )

    X_test, y_test = [], []
    for this_val_ind in range(len(data_gen_test)):
        X_test.append(data_gen_test[this_val_ind][0][0])
        y_test.append(data_gen_test[this_val_ind][1])
    X_test, y_test = np.array(X_test), np.array(y_test) 

    return X_train, y_train, X_test, y_test

def split_and_prep_data(data, split_point, window_size, method_name):
    if method_name == 'normal':
        X, y = prep_data_for_model_method_1(data, window_size)
    if method_name == 'generator':
        return prep_data_for_model_method_using_generetor(data, window_size)
    if method_name == 'weekly':
        X, y = prep_data_for_model_weekly(data)
    if method_name == 'monthly':
        X, y = prep_data_for_model_monthly(data)
    if method_name == 'fusion':
        X, y = prep_data_for_model_fusion(data)


    if method_name == 'fusion':
        X_train = [X[0][:split_point[0]], X[1][:split_point[0]], X[2][:split_point[0]]]
        X_test = [
            X[0][split_point[0]+1:split_point[1]],  
            X[1][split_point[0]+1:split_point[1]],
            X[2][split_point[0]+1:split_point[1]]
        ]

        y_train = y[:split_point[0]]
        y_test = y[split_point[0]+1:split_point[1]]
    else:
        X_train, y_train = X[:split_point[0]], y[:split_point[0]]
        X_test, y_test = X[split_point[0]+1:split_point[1]], y[split_point[0]+1:split_point[1]]
    return X_train, y_train, X_test, y_test