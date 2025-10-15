"""
Example of a simple feedforward neural network using TensorFlow/Keras for classification
on a tabular dataset loaded from a CSV file.
The dataset is assumed to be numerical and comma-delimited, with the output column(s)
at the end of each row.
The script includes data loading, optional shuffling, train/validation splitting,
model definition, compilation, training, evaluation, and reporting of results.
It also includes options to silence TensorFlow messages during model compilation and training.

"""
import os
from numpy import loadtxt
import numpy as np
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix
import config as cfg # configuration file with hyper-parameters

# ###############################
# Begin silencing operations. This has nothing to do with ML and the
# topic of Neural Networks. It's just that TF is very noisy and hard
# to see what's happening through all the noise. This section of code
# eliminates the noise for demonstration purposes.
# ###############################

if cfg.SILENCE:
    from silenceStdError import SilenceStdErr as silence

# Silence TensorFlow messages by importing silence_tensorflow before TensorFlow
# this is for demo purposes only, not recommended for debugging during development or production
# must be imported before tensorflow
if cfg.SILENCE:
    import silence_tensorflow.auto

# more silencing of TensorFlow messages by setting environment variable and absl logging level
if cfg.SILENCE:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = cfg.LOGLEVEL # or any {'0', '1', '2', '3'}
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)

# import TensorFlow here because it must be after silence_tensorflow
# awkward, but it is what it is
import tensorflow as tf

# Further silence TensorFlow messages by setting environment variable and absl logging level
if cfg.SILENCE:
    tf.get_logger().setLevel('ERROR')

# ###############################
# end silencing operations
# ###############################


# load the dataset
# it is prepared, cleaned and ready
# normally you would have a lot of data prep work
dataset = loadtxt(cfg.DATAFILE, delimiter=',')

# first shuffle the dataset to eliminate any ordering bias
# do not do this after splitting into train and test sets!
# do not shiffle here for time series or sequential data!
# time-series and sequential data require special handling
# e.g. use sklearn.model_selection.TimeSeriesSplit for time series data
if cfg.SHUFFLE:
    # note this is an in-place shuffle
    np.random.shuffle(dataset)

# split into train and validation sets
n_train = int(cfg.TRAINSPLIT * len(dataset))
train, val = dataset[:n_train], dataset[n_train:]
print("Shapes:", train.shape, val.shape)

ninputs = dataset.shape[1]-cfg.OUTPUTS
# split into input (X) and output (y) variables
X_trn, y_trn = train[:,0:ninputs], train[:,ninputs]
X_val, y_val = val[:,0:ninputs], val[:,ninputs]
print("I/O Train Shapes:", X_trn.shape, y_trn.shape)
print("I/O Valid Shapes:", X_val.shape, y_val.shape)

# define the keras NN model
model = tf.keras.models.Sequential()
# the following are ROUGH guidelines for number of neurons in each layer (NOT ALWAYS TRUE!)
# number of neurons in first layer should be between number of inputs and number of outputs
# 2nd etc. layer neurons should be between number of neurons in the previous layer and number of outputs
# all hidden layers usually use relu activation (NOT ALWAYS TRUE!)
# output layer activation
#       sigmoid for binary classification
#       softmax for multi-class classification
#       linear for regression
# the number inputs can be confusing, it is sometimes the number of features, but if that number is different,
# then there is an implicit input layer which is the number of features
model.add(tf.keras.layers.Dense(cfg.HIDDEN1, activation='relu'))
if cfg.HIDDEN2 > 0:
    model.add(tf.keras.layers.Dense(cfg.HIDDEN2, activation='relu'))
model.add(tf.keras.layers.Dense(cfg.OUTPUTS, activation='sigmoid'))

# compile the keras model using silence_stderr to suppress TensorFlow messages
with silence():
    history = model.compile(loss=cfg.LOSS, optimizer=cfg.OPTIMIZER, metrics=cfg.METRICS)
model.summary()

# train (fit) the keras model on the dataset using silence_stderr to suppress TensorFlow messages
with silence() if cfg.SILENCE else None:
    # this both trains and returns a history object which contains training history
    history = model.fit(X_trn, y_trn, validation_data=(X_val, y_val), epochs=cfg.EPOCHS, batch_size=cfg.BATCHSIZE, verbose=cfg.VERBOSE)
model.summary()

# show the training history if there is any
# history.history is a dictionary with keys 'loss', 'val_loss', and any metrics specified
# e.g. 'accuracy', 'val_accuracy', etc.
# each key contains a list of values for each epoch
if len(history.history['loss']) and len(history.history['val_loss']):
    # plot history
    matplotlib.pyplot.plot(history.history['loss'], label='train')
    matplotlib.pyplot.plot(history.history['val_loss'], label='test')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

# evaluate the keras model
# print all stats and confusion matrix
_, train_acc = model.evaluate(X_trn, y_trn, verbose=0)
_, test_acc = model.evaluate(X_val, y_val, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
