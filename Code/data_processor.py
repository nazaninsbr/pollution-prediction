import numpy as np 
from sklearn.model_selection import train_test_split

def read_data(file_path):
    ###################################
    # each line is one hour
    # the fields are: pollution, dew, temp, pressure, wind_dir, wind_spd, snow, rain
    ###################################
    return np.load(file_path)

def prep_data_for_model(data, window_size):
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


def split_and_prep_data(data, split_point, window_size):
    X, y = prep_data_for_model(data, window_size)
    X_train, y_train = X[:split_point[0]], y[:split_point[0]]
    X_test, y_test = X[split_point[0]+1:split_point[1]], y[split_point[0]+1:split_point[1]]
    return X_train, y_train, X_test, y_test