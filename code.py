import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Preprocess data
# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Feature binning: split people into 5 groups (0, 1, 2, 3, 4) corresponding to the certain feature
age_binner = KBinsDiscretizer(n_bins=5, encode='ordinal')
income_binner = KBinsDiscretizer(n_bins=5, encode='ordinal')
score_binner = KBinsDiscretizer(n_bins=5, encode='ordinal')

data['Age_Group'] = age_binner.fit_transform(data['Age'].values.reshape(-1,1).astype('int64'))
data['Income_Group'] = age_binner.fit_transform(data['Annual Income (k$)'].values.reshape(-1,1).astype('int64'))
data['Score_Group'] = age_binner.fit_transform(data['Spending Score (1-100)'].values.reshape(-1,1).astype('int64'))

# Process gender data from string to int (Male -> 1 & Female -> 2)
le = LabelEncoder()
data['Genre'] = le.fit_transform(data['Genre'])

# Scale the data
X, t = data.drop('Spending Score (1-100)', axis=1), data['Spending Score (1-100)']
r = RobustScaler()
m = MinMaxScaler()

for col in X.columns:
    X[col] = r.fit_transform(X[col].values.reshape(-1, 1))
    X[col] = m.fit_transform(X[col].values.reshape(-1, 1))
    
t = r.fit_transform(t.values.reshape(-1, 1))
t = m.fit_transform(t.reshape(-1, 1))
t = t.ravel()

# Split data for training, validation and testing
# Train:Val:Test = 8:1:1 (due to limited num of data samples)
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=500) # 0.2
X_train, X_val, t_train, t_val = train_test_split(X_train, t_train, test_size=0.25, random_state=500) # 0.25

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

def SGD_train(X_train, t_train, X_val, t_val, param=None, test=False):
    if param:
        model = SGDRegressor(penalty=param[0], alpha=param[1])
    else:
        model = SGDRegressor()
        
    model.fit(X_train, t_train)
    val_score = model.score(X_val, t_val) * 100
    
    if test:
        pred = model.predict(X_val)
        test_score = r2_score(t_val, pred) * 100
        error = np.sqrt(mean_squared_error(t_val, pred))
        return test_score, error
    
    return val_score

def KNN_train(X_train, t_train, X_val, t_val, param=None, test=False):
    if param:
        model = KNeighborsRegressor(leaf_size=param[0], n_neighbors=param[1], p=param[2])
    else:
        model = KNeighborsRegressor()
        
    model.fit(X_train, t_train)
    val_score = model.score(X_val, t_val) * 100
    
    if test:
        pred = model.predict(X_val)
        test_score = r2_score(t_val, pred) * 100
        error = np.sqrt(mean_squared_error(t_val, pred))
        return test_score, error
    
    return val_score

def SVR_train(X_train, t_train, X_val, t_val, param=None, test=False):
    if param:
        model = SVR(C=param[0], gamma=param[1], kernel=param[2])
    else:
        model = SVR()
        
    model.fit(X_train, t_train)
    val_score = model.score(X_val, t_val) * 100
    
    if test:
        pred = model.predict(X_val)
        test_score = r2_score(t_val, pred) * 100
        error = np.sqrt(mean_squared_error(t_val, pred))
        return test_score, error
    
    return val_score

def RF_train(X_train, t_train, X_val, t_val, param=None, test=False):
    if param:
        model = RandomForestRegressor(max_depth=param[0], max_features=param[1])
    else:
        model = RandomForestRegressor()
        
    model.fit(X_train, t_train)
    val_score = model.score(X_val, t_val) * 100
    
    if test:
        pred = model.predict(X_val)
        test_score = r2_score(t_val, pred) * 100
        error = np.sqrt(mean_squared_error(t_val, pred))
        return test_score, error
    
    return val_score

def train():  
    # Config
    best_model = None 
    global_acc = -1
    global_param = None
    
    # SGD hyper-parameters
    regularizations = ['l1', 'l2', 'elasticnet'] 
    alphas = [0.001, 0.0001, 0.00001]
    
    # KNN hyper-parameters
    leaf_sizes = list(range(1, 50))
    n_neighbors = list(range(1, 10))
    ps = [1, 2]
    
    # SVR hyper-parameters
    Cs = [0.1, 1, 10, 100, 1000]
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    kernels = ['rbf', 'poly', 'sigmoid']
            
    # RF hyper-parameters
    max_depths = [80, 90, 100, 110, 120]
    max_features = [2, 3, 4, 5]    
    
    models = [SGDRegressor, KNeighborsRegressor, SVR, RandomForestRegressor]
    for model in models:
        if model == SGDRegressor:
            # Configs settings
            best_acc = -1
            best_param = None
            # Training
            for regularization in regularizations:
                for alpha in alphas:
                    hyper_param = [regularization, alpha]
                    acc = SGD_train(X_train, t_train, X_val, t_val, hyper_param)
                    if acc > best_acc:
                        best_acc = acc
                        best_param = hyper_param
                        
            # With hyper-parameter
            test_acc_param, test_error_param = SGD_train(X_train, t_train, X_test, t_test, best_param, True)
            # Without hyper-parameter
            test_acc, test_error = SGD_train(X_train, t_train, X_test, t_test, None, True)
            
            if test_acc > global_acc:
                global_acc = test_acc
                best_model = 'SGD without hyper-parameters'
                global_param = None
            if test_acc_param > global_acc:
                global_acc = test_acc
                best_model = 'SGD with hyper-parameters' 
                global_param = best_param
            
            print("SGD best param:", best_param, "validation accuracy: %.2f" % (best_acc))            
            print("SGD with hyper-parameters. Test accuracy: %.2f Error: %.4f" % (test_acc_param, test_error_param))
            print("SGD without hyper-parameters. Test accuracy: %.2f Error: %.4f" % (test_acc, test_error))

        elif model == KNeighborsRegressor:
            # Configs settings
            best_acc = -1
            best_param = None
            # Training
            for leaf_size in leaf_sizes:
                for n_neighbor in n_neighbors:
                    for p in ps:
                        hyper_param = [leaf_size, n_neighbor, p]
                        acc = KNN_train(X_train, t_train, X_val, t_val, hyper_param)
                        if acc > best_acc:
                            best_acc = acc
                            best_param = hyper_param
                            
            # With hyper-parameter
            test_acc_param, test_error_param = KNN_train(X_train, t_train, X_test, t_test, best_param, True)
            # Without hyper-parameter
            
            if test_acc > global_acc:
                global_acc = test_acc
                best_model = 'KNN without hyper-parameters'
                global_param = None
            if test_acc_param > global_acc:
                global_acc = test_acc
                best_model = 'KNN with hyper-parameters' 
                global_param = best_param
                
            test_acc, test_error = KNN_train(X_train, t_train, X_test, t_test, None, True)
            print("\nKNN best param:", best_param, "validation accuracy: %.2f" % (best_acc))            
            print("KNN with hyper-parameters. Test accuracy: %.2f Error: %.4f" % (test_acc_param, test_error_param))
            print("KNN without hyper-parameters. Test accuracy: %.2f Error: %.4f" % (test_acc, test_error))
        
        elif model == SVR:
            # Configs settings
            best_acc = -1
            best_param = None
            # Training
            for c in Cs:
                for gamma in gammas:
                    for kernel in kernels:
                        hyper_param = [c, gamma, kernel]
                        acc = SVR_train(X_train, t_train, X_val, t_val, hyper_param)
                        if acc > best_acc:
                            best_acc = acc
                            best_param = hyper_param
                            
            # With hyper-parameter
            test_acc_param, test_error_param = SVR_train(X_train, t_train, X_test, t_test, best_param, True)
            # Without hyper-parameter
            test_acc, test_error = SVR_train(X_train, t_train, X_test, t_test, None, True)
            
            if test_acc > global_acc:
                global_acc = test_acc
                best_model = 'SVR without hyper-parameters'
                global_param = None
            if test_acc_param > global_acc:
                global_acc = test_acc
                best_model = 'SVR with hyper-parameters' 
                global_param = best_param
                
            print("\nSVR best param:", best_param, "validation accuracy: %.2f" % (best_acc))            
            print("SVR with hyper-parameters. Test accuracy: %.2f Error: %.4f" % (test_acc_param, test_error_param))
            print("SVR without hyper-parameters. Test accuracy: %.2f Error: %.4f" % (test_acc, test_error))
            
        elif model == RandomForestRegressor:
            # Configs settings
            best_acc = -1
            best_param = None
            # Training
            for max_depth in max_depths:
                for max_feature in max_features:
                    hyper_param = [max_depth, max_feature]
                    acc = RF_train(X_train, t_train, X_val, t_val, hyper_param)
                    if acc > best_acc:
                        best_acc = acc
                        best_param = hyper_param
            
            # With hyper-parameter
            test_acc_param, test_error_param = RF_train(X_train, t_train, X_test, t_test, best_param, True)
            # Without hyper-parameter
            test_acc, test_error = RF_train(X_train, t_train, X_test, t_test, None, True)
            
            if test_acc > global_acc:
                global_acc = test_acc
                best_model = 'RF without hyper-parameters'
                global_param = None
            if test_acc_param > global_acc:
                global_acc = test_acc
                best_model = 'RF with hyper-parameters'
                global_param = best_param
                
            print("\nRF best param:", best_param, "validation accuracy: %.2f" % (best_acc))            
            print("RF with hyper-parameters. Test accuracy: %.2f Error: %.4f" % (test_acc_param, test_error_param))
            print("RF without hyper-parameters. Test accuracy: %.2f Error: %.4f" % (test_acc, test_error))
    
    if global_param:
        print("\nThe best model:", best_model, global_param)
    else:
        print("\nThe best model:", best_model)
    
train()