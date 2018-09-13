from keras.callbacks import Callback
import numpy as np

class LossStopping(Callback):
    # Stop training when loss keeps increase for 30 epoches
    def __init__(self, patience=20):
        super(LossStopping, self).__init__()
        self.monitor = 'loss'
        self.patience = patience
        self.count = 0
        self.last_loss = np.Inf

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.count = 0
        self.last_loss = -np.Inf
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            print('\n')
            print(logs.keys())
            print(self.monitor)
            return
            
        if current < self.last_loss:
            self.count = 0
            self.last_loss = current
            
        else:
            self.count += 1
            self.last_loss = current
            if self.count > self.patience:
                self.model.stop_training = True
                print('loss stopping, hahahaha\n')
                
                  
class AccStopping(Callback):
    # Stop training when accuracy is lower than 1% after 200 batch
    def __init__(self):
        super(AccStopping, self).__init__()
        self.monitor = 'acc'
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            print('\n')
            print(logs.keys())
            print(self.monitor)
            return
            
        if current < 0.1 and batch>300:
           self.model.stop_training = True
           print('acc stopping, hahahaha\n')
        
   