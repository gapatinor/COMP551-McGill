import numpy as np
from utilities import *
import matplotlib.pyplot as plt

def plot_batches(batch_size_arr, run_times,  acc_val_arr):
     plt.plot(batch_size_arr, run_times)
     plt.xlabel("batch size")
     plt.ylabel("Convergence speed")
     plt.show()
     
     plt.plot(batch_size_arr, acc_val_arr)
     plt.xlabel("batch size")
     plt.ylabel("Accuracy")
     plt.show()
     
def plot_folds(folds_arr, title, ylabel, hyper_arr):
    '''
    plot information in folds
    Args: folds_arr array of representative folds, each component has the info of five folds
    e.g each component could have the train acc at each fold
    The array contain info of representative folds (where accuracy has an improve, see
    representative condition in find_best_hyper) 
    '''
    #select arbirraty a member of the array
    folds = folds_arr[-1]
    hyper = hyper_arr[-1]
    print(str(hyper))
    N = len(folds[0])
    iters = np.arange(N)
    for i,fold in enumerate(folds):
         plt.plot(iters, fold, label = "fold "+str(i+1))
    
    #plt.ylim(0.6,0.7)
    plt.title(title)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.show() 
    plt.close()
    
def plot_loss(iters,loss_train_hist, loss_val_hist):
    plt.plot(iters, loss_train_hist, label="train")
    plt.plot(iters, loss_val_hist, label="validation")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    
def plot_accuracy(iters, train_accs, val_accs):
    plt.plot(iters, train_accs, label = 'train acc')
    plt.plot(iters, val_accs, label = 'val acc')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()         
     
    