import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
import sub.optimizer as opt

def preprocessing():
    #load dataset
    x=pd.read_csv("data/parkinsons_updrs.csv")
    # dataset inspection  
    x.describe().T
    x.info()

    
    # Covariance matrix
    xnorm=(x-x.mean())/x.std()
    c=xnorm.cov()
    plt.matshow(np.abs(c))
    plt.colorbar()
    plt.title('Covariance matrix of the features')
    plt.show()
   
    
    #remove features
  
    #trash feature 
    trash_feature=['subject#', 'test_time']
                   
    x_filtered=x.copy(deep=True) 
    x_filtered=x_filtered.drop(trash_feature, axis=1)
    
    
    #Some interesting scatter plots
    
# =============================================================================
#     x_filtered.plot.scatter('motor_UPDRS', 'total_UPDRS')
#     plt.grid()
#     plt.plot([1,50],[1,50], 'r')
#     plt.title('Scatter plot: motor_UPDRS vs total_UPDRS')
#     plt.show()
#     x_filtered.plot.scatter('Shimmer:APQ3', 'Shimmer:DDA')
#     plt.title('Scatter plot: Shimmer:APQ3 vs Shimmer:DDA')
#     plt.grid()
#     plt.show()
#     x_filtered.plot.scatter('Jitter:RAP', 'Jitter:DDP')
#     plt.show()
#     
# =============================================================================
    
    
    #shuffle dataset
    x_shufld=x_filtered.sample(frac=1).reset_index(drop=True)
    
    
    #dataset partition
    Ntr=math.floor(x_filtered.shape[0]*0.50)
    Nva=math.floor((x_filtered.shape[0]-Ntr)/2)
    
    x_tr=x_shufld[0:Ntr]
    x_va=x_shufld[Ntr:Ntr+Nva]
    x_te=x_shufld[Ntr+Nva:]
    
    #normalization
    mean=x_tr.mean()
    std=x_tr.std()
    x_tr_norm=(x_tr-mean)/std
    x_va_norm=(x_va-mean)/std
    x_te_norm=(x_te-mean)/std
     

    #regression data
    y_tr_norm=x_tr_norm['total_UPDRS'].values.reshape(Ntr, 1)
    X_tr_norm=x_tr_norm.drop('total_UPDRS', axis=1).values
    
    y_te_norm=x_te_norm['total_UPDRS'].values.reshape(Nva,1)
    X_te_norm=x_te_norm.drop('total_UPDRS', axis=1).values
    
    y_va_norm=x_va_norm['total_UPDRS'].values.reshape(Nva,1)
    X_va_norm=x_va_norm.drop('total_UPDRS', axis=1).values
    return X_tr_norm, y_tr_norm, X_te_norm, y_te_norm, X_va_norm, y_va_norm, mean, std 

def test_regression(method,w,X_tr_norm, y_tr_norm, X_te_norm, y_te_norm, X_va_norm, y_va_norm, mean, std):
    y_tr_hat=np.dot(X_tr_norm, w)
    y_te_hat = np.dot(X_te_norm, w)
    y_va_hat=np.dot(X_va_norm, w)
    
    #unnormalize results for plots
    y_te_hat_u=unnormalize(y_te_hat, mean, std)
    y_tr_hat_u=unnormalize(y_tr_hat, mean, std)
    y_va_hat_u=unnormalize(y_va_hat, mean, std)
    y_tr_u=unnormalize(y_tr_norm, mean, std)
    y_te_u=unnormalize(y_te_norm, mean, std)
    y_va_u=unnormalize(y_va_norm, mean, std)
    
    #compute regression error
    err_train=y_tr_hat_u-y_tr_u
    err_val=y_va_hat_u-y_va_u
    err_test=y_te_hat_u-y_te_u
    
    #statistics
    y_hat_vs_y(method, y_te_u, y_te_hat_u)
    bins=np.linspace(-12, 12, 50)
    hist_estim_err('Distribution of estimation error using '+method,err_train, err_val, err_test, bins)
    print("Mean regression error for training:", err_train.mean())
    print("Standard deviation of regression error for training:", err_train.std())
    print("MSE of regression error for training:", opt.MSE(y_tr_hat_u, y_tr_u))
    print("Mean regression error for validation:", err_val.mean())
    print("Standard deviation of regression error for validation:", err_val.std())
    print("MSE of regression error for validation:", opt.MSE(y_va_hat_u, y_va_u))
    print("Mean regression error for testing:", err_test.mean())
    print("Standard deviation of regression error for testing:", err_test.std())
    print("MSE of regression error for testing:", opt.MSE(y_te_hat_u, y_te_u))
    R2(y_te_hat_u, y_te_u)

def y_hat_vs_y(method, y, y_hat):
    plt.figure()
    plt.plot(y_hat, y, 'ro')
    plt.plot([1,60],[1,60], 'b')
    plt.xlabel('y_hat: Estimated total_UPDRS')
    plt.ylabel('y: true total_UPDRS')
    plt.title('Estimated vs true UPDRS for '+method)
    plt.grid()
    plt.show()
    
def unnormalize(y, mean, std):
    y_un=y*std['total_UPDRS']+mean['total_UPDRS']
    return y_un

    
def R2 (y_hat, y):
    y_mean=y.mean()
    var_e=np.sum((y_hat-y)**2)
    var_d=np.sum((y-y_mean)**2)
    R2=1-(var_e/var_d)
    print('R2 value:', R2)

def hist_estim_err(title,err_train, err_val, err_test, bins):
    plt.figure()
    plt.hist(err_train, bins, alpha=0.5, label='err_train')
    plt.hist(err_val, bins, alpha=0.5, label='err_val')
    plt.hist(err_test, bins, alpha=0.5, label='err_test')
    plt.legend(loc='upper right')
    plt.xlabel('estimation_error: y_hat-y')
    plt.ylabel('absolute frequency of estimation error')
    plt.title(title)
    plt.grid()
    plt.show()
    return 


