import numpy as np
from utilities import *

class GradientDescent:
    
    def __init__(self, learning_rate=.001, max_iters=100, epsilon=1e-8, record_history=False, \
                 batch_size=2048, beta=0.7, print_hist = False):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.record_history = record_history
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.beta = beta
        self.print_hist = print_hist
        if record_history:
            self.w_history = []                 
    
    
    
    def run(self, gradient_fn, x, y, x_val, y_val, w):
       
        grad = np.inf
        t = 1
        train_accs = []
        valid_accs = []
        loss_train_hist = []
        loss_val_hist = []
        mini_batches = build_mini_batches(x, y, self.batch_size) 
        delta_w = 0
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            loss_train = 0
            loss_val = 0
            
            #gradient descent loop
            for mini_batch in mini_batches: 
                x_mini_batch, y_mini_batch = mini_batch[0], mini_batch[1].astype(int) 
                
                # gradient and loss of training
                loss_t, grad = gradient_fn(x_mini_batch, y_mini_batch, w) 
                #loss of validation
                loss_v, _ = gradient_fn(x_val, y_val, w) 
                loss_train += loss_t
                loss_val += loss_v
                # update using BGD
                #w = w - self.learning_rate * grad 
                #update using momentum
                delta_w = self.beta*delta_w + (1.0-self.beta)*grad
                w = w - self.learning_rate * delta_w 
            if self.record_history:
                 self.w_history.append(w)
            
            #save loss train
            loss_train_hist.append(loss_train)
            #save loss val
            loss_val_hist.append(loss_val)
            #prediction train
            y_pred = x.dot(w)
            #prediction val
            y_val_pred = x_val.dot(w)
            #most probable class training
            y_pred = np.argmax(y_pred, axis =1)
            #most probable class val
            y_val_pred = np.argmax(y_val_pred, axis =1)
            #accuracy train
            acc_train = accuracy(y_pred, y)
            #accuracy val
            acc_val = accuracy(y_val_pred, y_val)
            if(self.print_hist == True):
                if(t%10 == 0): print(t, acc_val)
            #save acc train
            train_accs.append(acc_train)
            #save acc val
            valid_accs.append(acc_val)
            t += 1
        
        results = {}
        results['w'] = w
        results['train_accs'] = train_accs
        results['valid_accs'] = valid_accs 
        results['loss_train_hist'] = loss_train_hist
        results['loss_val_hist'] = loss_val_hist
        return results
        
        
        
        