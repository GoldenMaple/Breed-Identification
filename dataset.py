import tensorflow as tf
from PIL import Image
import numpy as np
import keras
import pandas as pd
slim = tf.contrib.slim
from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt
from UtilsData import DataLoader
from UtilsData import Dataset
import cfg
from preprocessing import preprocess_image
from sklearn.cross_validation import train_test_split
from keras.applications.inception_resnet_v2 import preprocess_input

def _ShowImg(img):
    plt.figure()
    plt.imshow(img.astype('uint8'))
    plt.show()
    plt.close()
    
def _ShowBatch(imgs):
    num = imgs.shape[0]
    for i in range(num):
        _ShowImg(imgs[i])
        

def GetFn_NameToLabel(class_names):
    # class_name is a list of class names
    # return a function
    mapping = {}
    for label, name in enumerate(class_names):
        mapping[name] = label
    
    NameToLabel = lambda name: mapping[name]
    return NameToLabel
    
def GetFn_FilenameToAbsolutePath(root):
    # root is the root path of all images
    # return a function
    return lambda name: join(root, '%s.jpg' % name)


class Transfer():
    def __init__(self, image_size=299, is_training=True, is_show=False):
        self.sess = tf.Session()
        with self.sess.as_default():
            self.pl = tf.placeholder(dtype=tf.uint8)
            self.result = preprocess_image(self.pl, image_size, image_size, is_training=is_training)
            # here result.dtype should be tf.float32
            if is_show and self.result.dtype != tf.uint8:
                self.result = tf.cast(self.result, dtype=tf.uint8)
                
    def T(self, image):
        feed_dict = {self.pl: image}
        result = self.sess.run(self.result, feed_dict=feed_dict)
        return result
        
class MyDataSet(Dataset):
    def __init__(self, df, image_size=299, is_training=True, is_show=False):
        '''
        df is a dataframe with two field:
        df['path'] contains the absolute path for each image
        df['label'] contains the class label (integer) for each image
        '''
        self.df = df
        self.Fn_GetImage = lambda path: np.array(Image.open(path))
        self.trans = Transfer(image_size, is_training, is_show)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        path = self.df.loc[idx, 'path']
        image = self.Fn_GetImage(path)
        image = self.trans.T(image)
        label = self.df.loc[idx, 'label']
        return image, label
  

# get training and validation data loader
def GetDataLoaders(is_show=False):
    df = pd.read_csv(cfg.PathImageLabels)
    selected_class = list(df.groupby('breed').count().sort_values(by='id',ascending=False).head(cfg.num_classes).index)
    selected_df = df[df['breed'].isin(selected_class)]
    selected_df.reset_index(drop=True,inplace=True)
    selected_df.loc[:, 'label'] = selected_df['breed'].apply(GetFn_NameToLabel(selected_class))
    selected_df.loc[:, 'path'] = selected_df['id'].apply(GetFn_FilenameToAbsolutePath(cfg.PathTrainImage))
    
    df = pd.DataFrame()
    df.loc[:, 'path'] = selected_df['path'].copy()
    df.loc[:, 'label'] = selected_df['label'].copy()
    
    data_tr, data_val = train_test_split(df.values, train_size=0.9)
    df_tr = pd.DataFrame(data_tr, columns=df.columns)
    df_val = pd.DataFrame(data_val, columns=df.columns)
    
    dataset_tr = MyDataSet(df=df_tr, image_size=cfg.network_input_size, is_training=True, is_show=is_show)
    dataset_val = MyDataSet(df=df_val, image_size=cfg.network_input_size, is_training=False, is_show=is_show)
    
    loader_tr = DataLoader(dataset_tr, batch_size=cfg.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False)
    
    return loader_tr, loader_val
    
def GetDataGenerator(dataloader):
    while True:
        for images, labels in dataloader:
            images = preprocess_input(images)
            labels = keras.utils.to_categorical(labels, cfg.num_classes)
            yield (images, labels)
    
if __name__=='__main__':
    loader_tr, loader_val = GetDataLoaders(is_show=True)
    images, labels = next(iter(loader_tr))
    print(images.shape)
    print(labels)
    _ShowBatch(images)
            
    

    

