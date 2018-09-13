from keras import optimizers, regularizers
from keras.layers import Dense
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.callbacks import TerminateOnNaN# , EarlyStopping
import numpy as np
import pandas as pd
import tabulate as tab
from dataset import GetDataLoaders, GetDataGenerator
import cfg
from end_condition import LossStopping

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
        
if __name__=='__main__':

    base_model = GetBaseModel()
    loader_tr, loader_val = GetDataLoaders()
    gen_tr = GetDataGenerator(loader_tr)
    gen_val = GetDataGenerator(loader_val)
    
    reg = np.exp(-7.3)
    
    model = GetModel(base_model=base_model, reg=reg)
    model.compile(loss='categorical_crossentropy',
                    # optimizer=optimizers.RMSprop(lr=lr),
                    optimizer=optimizers.SGD(lr=0.33),
                    metrics=['accuracy'])
    
    callbacks = [
                    LossStopping(),
                    #AccStopping(),
                    TerminateOnNaN(), 
                    # early_stopping,
                    ]
                 
    history = model.fit_generator(gen_tr,
                        epochs=3,
                        validation_data=gen_val,
                        callbacks=callbacks,
                        steps_per_epoch=len(loader_tr),
                        validation_steps=len(loader_val),
                        )
    
    Save(model)

    for i, layer in enumerate(base_model.layers):
        print(i,': ', layer.name)
        
    for layer in base_model.layers[762:]:
        layer.trainable = True
        
    lr = 0.003
    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=lr),
                    metrics=['accuracy'])
    
    history = model.fit_generator(gen_tr,
                        epochs=3,
                        validation_data=gen_val,
                        callbacks=callbacks,
                        steps_per_epoch=len(loader_tr),
                        validation_steps=len(loader_val),
                        )
    
    
    

    
    
    
    
    
