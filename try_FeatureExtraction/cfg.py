from os.path import join
ImgFormat = 'png'
Channel = 3
num_classes = 120
# num_classes = 16
batch_size = 20
image_size = 299

_PathData = join('..', '..', 'Data')
_PathTrain = join(_PathData, 'train')
_PathTest = join(_PathData, 'test')
_PathLabel = join(_PathData, 'label')

PathTrainImage = join(_PathTrain, 'img')
PathTrainDeepFeature = join(_PathTrain, 'feature')
PathTestImage = join(_PathTest, 'img')
PathTestDeepFeature = join(_PathTest, 'feature')

PathAllClasses = join(_PathLabel, 'classes.txt')
PathLabelMapping = join(_PathLabel, 'mapping.txt')
PathImageLabels = join(_PathLabel, 'labels.csv')
PathSubmission = 'submission.csv'




