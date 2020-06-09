from datetime import datetime as dt
import numpy as np
from functools import reduce
import pandas as pd
from sklearn.impute import KNNImputer
import copy
import matplotlib.pyplot as plt

def erase_data(data, nan_rate):
    output_data = copy.deepcopy(data)
    nr, nc = output_data.shape
    for ind in range(nc):
        observed_index = np.random.random(nr) > nan_rate
        output_data[observed_index == False, ind] = np.nan
    return output_data

def triple_dot(D1, D2, D3):
    return np.dot(np.dot(D1, D2), D3)

def predict_method1(X, max_iter = 3000, eps = 1e-08):
    nr, nc = X.shape
    C = np.isnan(X) == False

    M = np.arange(1, nc+1) * (C == False) - 1
    O = np.arange(1, nc+1) * C - 1
    Mu = np.nanmean(X, axis = 0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows, :].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis = 0))

    Mu_p, S_p = {}, {}
    X_p = copy.deepcopy(X)
    no_conv = True
    for iteration in range(0, max_iter):
        if not no_conv:
            break
        for i in range(nr):
            S_p[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i, ]) != set(np.arange(nc)):
                M_i, O_i = M[i, :][M[i, :] != -1], O[i, :][O[i, :] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                temp = (X_p[i, O_i] - Mu[np.ix_(O_i)])
                triple = triple_dot(S_MO, np.linalg.inv(S_OO), temp)
                Mu_p[i] = Mu[np.ix_(M_i)] + triple
                X_p[i, M_i] = Mu_p[i]
                S_MM_O = S_MM - triple_dot(S_MO, np.linalg.inv(S_OO), S_OM)
                S_p[i][np.ix_(M_i, M_i)] = S_MM_O
        Mu_new = np.mean(X_p, axis = 0)
        cov = np.cov(X_p.T, bias = 1)
        S_new = cov + reduce(np.add, S_p.values()) / nr
        no_conv1 = np.linalg.norm(Mu - Mu_new) >= eps
        no_conv2 = np.linalg.norm(S - S_new, ord = 2) >= eps
        no_conv = no_conv1 or no_conv2
        Mu = Mu_new
        S = S_new
    return X_p

def predict_method2(X):
    df = pd.DataFrame(missed_data)
    imputer = KNNImputer(n_neighbors=5)
    return imputer.fit_transform(df)

def predict_method3(X):
    nr, nc = X.shape
    c = np.isnan(X)
    pred = copy.deepcopy(X)
    for i in range(nc):
        true_ind = np.where((c==False)[:,i])[0]
        ind = np.where(c[:,i])[0]
        m = np.mean(pred[true_ind, i])
        pred[ind, i] = m
    return pred

def evaluate(X, pred, missed_data, for_all_data=True):
    from sklearn.metrics import mean_squared_error
    for i in range(0,8):
        c = np.isnan(missed_data)
        ind = np.where(c[:,i])[0]
        if for_all_data:
            mse = mean_squared_error(X[:, i], pred[:, i])
        else:
            mse = mean_squared_error(X[ind, i], pred[ind, i])
        print(i, mse)

def plot_kde(X):
    import seaborn as sns
    c = np.isnan(X)
    col_name = ["pollution", "dew", "temp", "pressure",
                "wind_dir", "wind_spd", "snow", "rain"]
    fig, arr_ax = plt.subplots(2, 4, figsize=(14,8))
    a = arr_ax.reshape(8)
    for i in range(8):
        true_ind = np.where((c==False)[:,i])[0]
        sns.kdeplot(missed_data[true_ind, i], shade=True, bw=.02, color="olive", ax=a[i])
        a[i].set_title(col_name[i])
    plt.show()

import data_processor
import constants
data = data_processor.read_data(constants.path_to_data)
missed_data = erase_data(data, 0.2)
# plot_kde(missed_data)

our_pred = predict_method1(missed_data)

knn_pred = predict_method2(missed_data)

mean_pred = predict_method3(missed_data)


print("our predictor ::: ")
print("> for all data:")
evaluate(data, our_pred, missed_data)
print("> for missing data:")
evaluate(data, our_pred, missed_data, False)

print("##################")

print("knn predictor ::: ")
print("> for all data:")
evaluate(data, knn_pred, missed_data)
print("> for missing data:")
evaluate(data, knn_pred, missed_data, False)

print("##################")

print("mean predictor ::: ")
print("> for all data:")
evaluate(data, mean_pred, missed_data)
print("> for missing data:")
evaluate(data, mean_pred, missed_data, False)
