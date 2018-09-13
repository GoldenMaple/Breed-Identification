from os.path import join
ImgFormat = 'png'
Channel = 3
num_classes = 120
batch_size = 20
network_input_size = 299
# input_size for inception_res_v2 = 299
# input_size for vgg = 150

_PathData = 'E:\\DM\\Dog Breed Identification\\Data'
_PathTrain = join(_PathData, 'train')
_PathTrain = join(_PathData, 'train')
_PathVal = join(_PathData, 'val')
_PathTest = join(_PathData, 'test')
_PathLabel = 'label'


PathTrainImage = join(_PathTrain, 'img')
PathTrainRecord = join(_PathTrain, 'tf.record')

PathTestImage = join(_PathTest, 'img')
PathTestRecord = join(_PathTest, 'tf.record')

PathValImage = join(_PathVal, 'img')
PathValRecord = join(_PathVal, 'tf.record')

PathAllClasses = join(_PathLabel, 'classes.txt')
PathLabelMapping = join(_PathLabel, 'mapping.txt')
PathImageLabels = join(_PathLabel, 'labels.csv')

_PathModel = join('..', '..', 'SavedModel')
ModelFileName = join(_PathModel, 'mymodel.h5')


