import numpy as np
from utilities import *

class SoftmaxRegression:
    def __init__(self, reg = 0.0, add_bias=True):
        self.add_bias = add_bias
        self.reg = reg
        pass
    
    
    def fit(self, x, y, x_val, y_val, optimizer):
        if self.add_bias:
            N = x.shape[0]
            #add bias to x
            x = np.column_stack([x,np.ones(N)])
            #add bias to x val
            x_val = np.column_stack([x_val,np.ones(x_val.shape[0])])
        #number of classes    
        self.C = len(np.unique(y))
        
        N,D = x.shape
        
        #gradient softmax with regularization
        def gradient(x, y, w):
            z = x.dot(w)
            exp_m = np.exp(z.astype(float))
            norm = np.sum(exp_m, axis = 1)
            loss = -(np.sum(one_hot(y, self.C)*z) - np.sum(np.log(norm)))
            loss /= N
            loss += self.reg*np.sum(w*w)
            
            norm = norm[:, np.newaxis]
            prob = exp_m/norm
            grad = x.T.dot(prob - one_hot(y, self.C))
            grad /= N
            grad += 2.0*self.reg*w
            
            return loss, grad
        w0 = np.zeros((D,self.C)) 
        
        results = optimizer.run(gradient, x, y, x_val, y_val, w0)      
        self.w = results['w']
        self.train_accs =  results['train_accs']
        self.val_accs =  results['valid_accs']
        self.loss_train = results['loss_train_hist']
        self.loss_val = results['loss_val_hist']
              
        return self
    
    def predict(self, x):
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x.dot(self.w)
       
        return yh