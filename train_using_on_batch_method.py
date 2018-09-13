import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
import cfg
import numpy as np
from dataset import GetDataLoaders

def Validate(dataloader, model):
    losses = []
    accs = []
    for images, labels in dataloader:
        labels = keras.utils.to_categorical(labels, cfg.num_classes)
        images = preprocess_input(images)
        [loss, acc] = model.test_on_batch(images, labels)
        losses.append(loss)
        accs.append(acc)
    return [np.average(losses), np.average(accs)]
    
def TrainOneEpoch(dataloader, model):
    for b, (images, labels) in enumerate(dataloader):
        labels = keras.utils.to_categorical(labels, cfg.num_classes)
        images = preprocess_input(images)
        [loss, acc] = model.train_on_batch(images, labels)
        if b % 30 == 0:
            info = 'acc:{0:6.3f}  loss:{1:6.3f}'
            print(info.format(acc, loss))


def TrainProcess(loader_tr, loader_val, model, epoch=1):
    for e in range(epoch):
        print('epoch {}'.format(e))
        TrainOneEpoch(loader_tr, model)
        loss, acc = Validate(loader_val, model)
        info = 'validation result: acc:{0:6.3f}  loss:{1:6.3f}'
        print(info.format(acc, loss))
    

def GetModel():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
    
    x = base_model.output
    x = Dense(cfg.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model, base_model
   
def Save(model):
    model.save_weights(cfg.ModelFileName)
def Load(model):
    model.load_weights(cfg.ModelFileName)
  

if __name__=='__main__':
    
    model, base_model = GetModel()
        
    for layer in base_model.layers:
        layer.trainable = False
    
    # opt = optimizers.rmsprop(lr=0.001, decay=0)
    opt = optimizers.SGD(lr=0.03)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    loader_tr, loader_val = GetDataLoaders()
    TrainProcess(loader_tr, loader_val, model, epoch=1)
    #Save(model)
    
    '''
    for layer in model.layers[:141]:
       layer.trainable = False
    for layer in model.layers[141:]:
       layer.trainable = True
       
    opt = optimizers.rmsprop(lr=0.001, decay=0.00004)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    TrainProcess(dl_tr, dl_val, model, epoch=5)
    '''

    
