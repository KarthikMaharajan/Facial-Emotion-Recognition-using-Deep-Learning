import argparse
import cPickle as pickle
import warnings
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib
import warnings
import numpy as np

from lib import deepbelief as db
from lib import restrictedBoltzmannMachine as rbm

from lib.activationfunctions import *
from lib.common import *
from read.readfacedatabases import *

import matplotlib
import os
havedisplay = "DISPLAY" in os.environ
if not havedisplay:
  exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
  havedisplay = (exitval == 0)
if havedisplay:
  import matplotlib.pyplot as plt
else:
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='emotion recongnition')
parser.add_argument('--rbmnesterov', dest='rbmnesterov',action='store_true', default=False,
                    help=("if true, rbms are trained using nesterov momentum"))
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',dest='train',action='store_true', default=False,
                    help=("if true, the network is trained from scratch from the"
                          "traning data"))
parser.add_argument('--rbm', dest='rbm',action='store_true', default=False,
                    help=("if true, the code for traning an rbm on the data is run"))
parser.add_argument('--sparsity', dest='sparsity',action='store_true', default=False,
                    help=("if true, the the networks are trained with sparsity constraints"))
parser.add_argument('--dbKanade', dest='dbKanade',action='store_true', default=False,
                    help=("if true, the code for training a deepbelief net on the"
                          "data is run, where the supervised data is the Kanade DB"))
parser.add_argument('--dbPIE', dest='dbPIE',action='store_true', default=False,
                    help=("if true, the code for training a deepbelief net on the"
                          "data is run, where the supervised data is the PIE DB"))
parser.add_argument('--trainSize', type=int, default=10000,
                    help='the number of tranining cases to be considered')
parser.add_argument('--testSize', type=int, default=1000,
                    help='the number of testing cases to be considered')
parser.add_argument('netFile', help="file where the serialized network should be saved")
parser.add_argument('--nesterov', dest='nesterov',action='store_true', default=False,
                    help=("if true, the deep belief net is trained using nesterov momentum"))
parser.add_argument('--rmsprop', dest='rmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the deep belief net."))
parser.add_argument('--rbmrmsprop', dest='rbmrmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the rbms."))
parser.add_argument('--save_best_weights', dest='save_best_weights',action='store_true', default=False,
                    help=("if true, the best weights are used and saved during training."))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, do cross validation"))
parser.add_argument('--cvPIE', dest='cvPIE',action='store_true', default=False,
                    help=("if true, do cross validation"))
parser.add_argument('--illumination',dest='illumination',action='store_true', default=False,
                    help="if true, trains and tests the images with different illuminations")
parser.add_argument('--pose',dest='pose',action='store_true', default=False,
                    help="if true, trains and tests the images with different poses")
parser.add_argument('--subjects',dest='subjects',action='store_true', default=False,
                    help="if true, trains and tests the images with different subjects")
parser.add_argument('--missing', dest='missing',action='store_true', default=False,
                    help=("tests the network with missing data."))
parser.add_argument('--crossdb', dest='crossdb',action='store_true', default=False,
                    help=("if true, trains the DBN with multi pie and tests with Kanade."))
parser.add_argument('--crossdbCV', dest='crossdbCV',action='store_true', default=False,
                    help=("if true, trains the DBN with multi pie and tests with Kanade."))
parser.add_argument('--facedetection', dest='facedetection',action='store_true', default=False,
                    help=("if true, do face detection"))
parser.add_argument('--maxEpochs', type=int, default=50,
                    help='the maximum number of supervised epochs')
parser.add_argument('--miniBatchSize', type=int, default=10,
                    help='the number of training points in a mini batch')
parser.add_argument('--validation',dest='validation',action='store_true', default=False,
                    help="if true, the network is trained using a validation set")
parser.add_argument('--equalize',dest='equalize',action='store_true', default=False,
                    help="if true, the input images are equalized before being fed into the net")
parser.add_argument('--crop',dest='crop',action='store_true', default=False,
                    help="crops images from databases before training the net")
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))
parser.add_argument('--preTrainEpochs', type=int, default=1,
                    help='the number of pretraining epochs')
parser.add_argument('--machine', type=int, default=0,
                    help='the host number of the machine running the experiment')
parser.add_argument('--kaggle',dest='kaggle',action='store_true', default=False,
                      help='if true, trains a net on the kaggle data')
parser.add_argument('--kagglecv',dest='kagglecv',action='store_true', default=False,
                      help='if true, cv for kaggle data')
parser.add_argument('--kagglesmall',dest='kagglesmall',action='store_true', default=False,
                      help='if true, cv for kaggle data')


# DEBUG mode?
parser.add_argument('--debug', dest='debug',action='store_false', default=False,
                    help=("if true, the deep belief net is ran in DEBUG mode"))

# Get the arguments of the program
args = parser.parse_args()

# Set the debug mode in the deep belief net
db.DEBUG = args.debug

SMALL_SIZE = ((40, 30))


def deepbeliefKaggleCompetitionSmallDataset(big=False):
  print "you are using the net file" , args.netFile
  print "after nefile"
  trainData, trainLabels = readKaggleCompetitionSmallDataset(args.equalize,True)
  print trainData.shape
  print trainLabels.shape

  print "after train"
  testData, testLabels = readKaggleCompetitionSmallDataset(args.equalize,False)
  print testData.shape
  print testLabels.shape

  if args.relu:
    activationFunction = Rectified()
    unsupervisedLearningRate = 0.5
    supervisedLearningRate = 0.01
    momentumMax = 0.95
    trainData = scale(trainData)
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()
  else:
    print "in else"
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

    unsupervisedLearningRate = 0.5
    supervisedLearningRate = 0.1
    momentumMax = 0.9

  if args.train:
    print "In training"

    net = db.DBN(5, [2304, 1500, 1500, 1500, 7],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               save_best_weights=args.save_best_weights,
               firstRBMheuristic=False,
               hiddenDropout=0.5,
               visibleDropout=0.8,
               rbmVisibleDropout=1.0,
               rbmHiddenDropout=1.0,
               initialInputShape=(48, 48),
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = readKaggleCompetitionUnlabelled()
    #unsupervisedData = None

    print "net print",net
    '''

    change april 12

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)
              '''


    net.train(trainData, trainLabels, args.maxEpochs,args.validation,unsupervisedData)
  else:
    # Take the saved network and use that for reconstructions
    with open(args.netFile, "rb") as f:
      net = pickle.load(f)

  print "nr layers: ", net.layerSizes

  probs, predicted = net.classify(testData)


  correct = 0
  errorCases = []

  for i in xrange(len(testData)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(testData)

  confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)
  print "confusion matrix"
  print confMatrix

  if args.save:
    with open(args.netFile, "wb") as f:
      print "you are saving in file", args.netFile
      pickle.dump(net, f)



deepbeliefKaggleCompetitionSmallDataset()