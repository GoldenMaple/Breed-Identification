from keras import optimizers, regularizers
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.callbacks import TerminateOnNaN# , EarlyStopping
import numpy as np
import pandas as pd
import tabulate as tab
from dataset import GetDataLoaders, GetDataGenerator
import cfg
from end_condition import LossStopping, AccStopping
pd.set_option('display.multi_sparse', False)

def MyPrintDF(df):
    table = tab.tabulate(df, headers='keys', tablefmt='psql')
    print(table)
    return
    
def GetBaseModel():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False
    return base_model
    
def GetModel(base_model, reg=0.0001, rate=0.2, feature=512):
    x = base_model.output
    # x = Dense(feature, activation='relu', kernel_regularizer=regularizers.l2(reg))(x)
    # x = Dropout(rate)(x)
    x = Dense(cfg.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(reg))(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def Save(model):
    model.save_weights(cfg.ModelFileName)
def Load(model):
    model.load_weights(cfg.ModelFileName)
   
def ShowHistory(h):
    keys = h.history.keys()
    for key in keys:
        info = 'info of {} is: {}'
        print(info.format(key, h.history[key]))
        
def HistoryToDict(h):
    keys = h.history.keys()
    return {key: h.history[key][-1] for key in keys}
        
def Run(lr, reg, base_model, gen_tr, gen_val, epoch):
    model = GetModel(base_model=base_model, reg=reg)
    model.compile(loss='categorical_crossentropy',
                    # optimizer=optimizers.RMSprop(lr=lr),
                    optimizer=optimizers.SGD(lr=lr),
                    metrics=['accuracy'])
    
    # early_stopping = EarlyStopping(monitor='acc', min_delta=0.05, patience=1, verbose=1)
    
    callbacks = [
                    LossStopping(),
                    AccStopping(),
                    TerminateOnNaN(), 
                    # early_stopping,
                    ]
                 
    history = model.fit_generator(gen_tr,
                        epochs=epoch,
                        validation_data=gen_val,
                        callbacks=callbacks,
                        steps_per_epoch=len(loader_tr),
                        validation_steps=len(loader_val),
                        )
    return HistoryToDict(history)
    
if __name__=='__main__':
    iteration = 10
    epoch=3
    
    lr_min = -10
    lr_max = 0
    
    re_min = -10
    re_max = 0
    
    base_model = GetBaseModel()
    loader_tr, loader_val = GetDataLoaders()
    gen_tr = GetDataGenerator(loader_tr)
    gen_val = GetDataGenerator(loader_val)
    
    results = []
    for i in range(iteration):
        print('--------------------')
        print('current iteration is {}'.format(i+1))
        print('--------------------')
        lr = np.exp(np.random.uniform(low=lr_min, high=lr_max))
        re = np.exp(np.random.uniform(low=re_min, high=re_max))
        result_dict = Run(lr, re, base_model, gen_tr, gen_val, epoch)
        result_dict['lr'] = np.log(lr)
        result_dict['reg'] = np.log(re)
        if 'val_loss' not in result_dict.keys():
            result_dict['val_loss'] = np.NaN
        if 'val_acc' not in result_dict.keys():
            result_dict['val_acc'] = np.NaN
        results.append(result_dict)
    
    d = {key:[dictionary[key] for dictionary in results] for key in results[0].keys()}
    df = pd.DataFrame(d)
    MyPrintDF(df)
        


   
    