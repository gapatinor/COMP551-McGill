import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
import time

def load_data1():
     digits = load_digits()
     n_samples = len(digits.images)
     X = digits.images.reshape((n_samples, -1))
     y = digits.target

    

     #define train, validation and test sets
     np.random.seed(1234)
     data = np.hstack((X,y[:,None]))
     np.random.shuffle(data)

     size_train = int(0.6*X.shape[0])
     size_val = int(0.25*X.shape[0])

     X_train = data[0:size_train, 0:-1]
     y_train = data[0:size_train, -1].astype(int)
     X_val = data[size_train:size_train+size_val, 0:-1]
     y_val = data[size_train:size_train+size_val, -1].astype(int)
     X_test = data[size_train+size_val:, 0:-1]
     y_test = data[size_train+size_val:, -1].astype(int)

     return X_train, y_train, X_val, y_val, X_test, y_test

def load_data2():
     mice = fetch_openml(name='miceprotein', version=4)
     X2 = mice.data
     y2 = mice.target
     col_mean = np.nanmean(X2, axis=0)
     X2 = np.where(np.isnan(X2),col_mean, X2) 
     
     
     labels = np.unique(y2)
     i = 0
     for label in labels:
         pos = y2 == label
         y2[pos] = i
         i += 1

     #define train, validation and test sets
     data2 = np.hstack((X2,y2[:,None]))
     np.random.shuffle(data2)

     size_train = int(0.6*X2.shape[0])
     size_val = int(0.25*X2.shape[0])

     X_train2 = data2[0:size_train, 0:-1]
     y_train2 = data2[0:size_train, -1].astype(int)
     X_val2 = data2[size_train:size_train+size_val, 0:-1]
     y_val2 = data2[size_train:size_train+size_val, -1].astype(int)
     X_test2 = data2[size_train+size_val:, 0:-1]
     y_test2 = data2[size_train+size_val:, -1].astype(int)
     
     return X_train2, y_train2, X_val2, y_val2, X_test2, y_test2

def one_hot(y, C):
    '''one hot representation of labels. Labels must be
       between [0, 1,...,C-1]
       Args: labels y [N,]
       return matrix [N,C]
    '''
    
    oh = np.zeros((len(y),C))
    oh[np.arange(len(y)), y] = 1
    return oh             
    
def build_mini_batches(x, y, batch_size): 
    '''
    Build mini batches.
    Args: [x] = [N,D], [y] = [N,], batch size
    Returns: array of minibatches, each component tuple (x_mini, y_mini)
    '''
    data = np.hstack((x,y[:,None]))
    mini_batches = [] 
    n_mini_batch = data.shape[0] // batch_size 
    i = 0
    for i in range(n_mini_batch): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_min_batch = mini_batch[:, :-1] 
        y_min_batch = mini_batch[:, -1]
        mini_batches.append((X_min_batch, y_min_batch)) 
    
    
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_min_batch = mini_batch[:, :-1] 
        y_min_batch = mini_batch[:, -1]
        mini_batches.append((X_min_batch, y_min_batch))    
    return mini_batches

def cross_validate(n, n_folds=5):
    #get the number of data samples in each split
    n_val = n // n_folds
    inds = np.random.permutation(n)
    inds = []
    for f in range(n_folds):
        tr_inds = []
        #get the validation indexes
        val_inds = list(range(f * n_val, (f+1)*n_val))
        #get the train indexes
        if f > 0:
            tr_inds = list(range(f*n_val))
        if f < n_folds - 1:
            tr_inds = tr_inds + list(range((f+1)*n_val, n))
        #The yield statement suspends function’s execution and sends a value back to the caller
        #but retains enough state information to enable function to resume where it is left off
        yield tr_inds, val_inds


def accuracy(y_pred, y):
    #accuracy, compares prediction and ground truth labels
    acc = (1.0/len(y))*(np.sum(np.array(y_pred) == np.array(y)))
    return acc

def build_grid(learning_rate_arr,  batch_size_arr, beta_arr):
    '''
    build grid to tune hyper-parameters.
    Args: array of learning rates, bacth sizes and betas
    Returns: array of dicts. Each component is a dict of parameters
    '''
    arr_dics = []
    
    for lr in learning_rate_arr:
        for bs in batch_size_arr:
            for b in beta_arr:
                params = {}
                params['learning_rate'] = lr
                params['batch_size'] = bs
                params['beta'] = b
                arr_dics.append(params)
    
   
    return arr_dics


     