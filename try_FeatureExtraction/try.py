import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.preprocessing import image
from os.path import join, splitext
from os import listdir
from tqdm import tqdm
import cfg


def GetMappingFns():
    with open(cfg.PathAllClasses,'r') as file:
        lines = file.readlines()
    class_names = [line.strip() for line in lines]

    mapping_name2label = {}
    mapping_label2name = {}
    for label, name in enumerate(class_names):
        mapping_name2label[name] = label
        mapping_label2name[label] = name

    NameToLabel = lambda name: mapping_name2label[name]
    LabelToName = lambda label: mapping_label2name[label]
    return NameToLabel, LabelToName
    
def GetFn_FilenameToAbsolutePath(root):
    # root is the root path of all images
    # return a function
    return lambda name: join(root, '%s.npy' % name)
    
def ReadImage(path, size):
    img = image.load_img(path, target_size=size)
    img = image.img_to_array(img)
    return img
    
def GetTrainingData(train_size = 0.9):
    df = pd.read_csv(cfg.PathImageLabels)
    NameToLabel, LabelToName = GetMappingFns()
    df.loc[:, 'label'] = df['breed'].apply(NameToLabel)
    df.loc[:, 'path'] = df['id'].apply(GetFn_FilenameToAbsolutePath(cfg.PathTrainDeepFeature))
    

    X = np.zeros([len(df), 1536], dtype='float32')
    for i in tqdm(range(len(df))):
        path = df.loc[i, 'path']
        feature = np.load(path)
        X[i] = feature[0]
    
    y = df['label'].values

    if train_size == 1:
        X_tr = X
        X_val = None
        y_tr = y
        y_val = None
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, train_size=train_size, random_state=0)
    return X_tr, X_val, y_tr, y_val, LabelToName
    
def GetTestData():
    array_names = listdir(cfg.PathTestDeepFeature)
    X = np.zeros([len(array_names), 1536], dtype='float32')
    for i, name in tqdm(enumerate(array_names)):
        path = join(cfg.PathTestDeepFeature, name)
        feature = np.load(path)
        X[i] = feature[0]

    array_names = [splitext(name)[0] for name in array_names]
    array_names = np.array(array_names).reshape(-1,1)
    return X, array_names
'''
def GetTrainingData():
    df = pd.read_csv(cfg.PathImageLabels)
    classes = list(df.groupby('breed').count().sort_values(by='id',ascending=False).head(cfg.num_classes).index)
    NameToLabel, LabelToName = GetMappingFns(classes)
    
    df = df[df['breed'].isin(classes)]
    df.reset_index(drop=True,inplace=True)
    df.loc[:, 'label'] = df['breed'].apply(NameToLabel(classes))
    df.loc[:, 'path'] = df['id'].apply(GetFn_FilenameToAbsolutePath(cfg.PathTrainDeepFeature))
    

    X = np.zeros([len(df), 1536], dtype='float32')
    for i in tqdm(range(len(df))):
        path = df.loc[i, 'path']
        feature = np.load(path)
        X[i] = feature[0]
    
    y = df['label'].values
    # y = keras.utils.to_categorical(y, cfg.num_classes)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
    return X_tr, X_te, y_tr, y_te, LabelToName
'''    

if __name__=='__main__':
    X_tr, X_val, y_tr, y_val, LabelToName = GetTrainingData(train_size=1)
    clf = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', multi_class='multinomial')
    print('training')
    clf.fit(X_tr, y_tr)
    print('done')
    
    if X_val is not None and y_val is not None:
        acc = clf.score(X_val, y_val)
        print('prediction score is {}'.format(acc))
        
    X_te, array_names = GetTestData()
    prob = clf.predict_proba(X_te)
    result = np.hstack([array_names, prob])
    columns = [LabelToName(c) for c in range(cfg.num_classes)]
    columns = ['id'] + columns

    final_result = pd.DataFrame(result, columns=columns)
    final_result.to_csv(cfg.PathSubmission, sep=',', index=False)
    




