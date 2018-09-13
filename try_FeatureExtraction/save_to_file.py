import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image
from os.path import join, splitext
from os import listdir
from tqdm import tqdm
import cfg
 
def ReadImage(path, size):
    img = image.load_img(path, target_size=size)
    img = image.img_to_array(img)
    return img
    
    
def GetModel():
    return InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    

def TrainingImgToArray(model):
    img_names = listdir(cfg.PathTrainImage)
    for img_name in tqdm(img_names):
        img_path = join(cfg.PathTrainImage, img_name)
        img = ReadImage(img_path, (cfg.image_size, cfg.image_size))
        img = preprocess_input(np.expand_dims(img.copy(), axis=0))
        deep_feature = model.predict(img)
        array_path = join(cfg.PathTrainDeepFeature, '%s.npy' % splitext(img_name)[0])
        np.save(array_path, deep_feature)
        
def TestingImgToArray(model):
    img_names = listdir(cfg.PathTestImage)
    for img_name in tqdm(img_names):
        img_path = join(cfg.PathTestImage, img_name)
        img = ReadImage(img_path, (cfg.image_size, cfg.image_size))
        img = preprocess_input(np.expand_dims(img.copy(), axis=0))
        deep_feature = model.predict(img)
        array_path = join(cfg.PathTestDeepFeature, '%s.npy' % splitext(img_name)[0])
        np.save(array_path, deep_feature)
    
if __name__=='__main__':
    net = GetModel()
    # TrainingImgToArray(net)
    TestingImgToArray(net)
    





