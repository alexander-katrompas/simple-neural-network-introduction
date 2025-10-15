"""
Configuration file for neural network training parameters
"""

LOGLEVEL = '3' # or any {'0', '1', '2', '3'}
SILENCE = True # whether to silence TensorFlow messages during model compilation and training
DATAFILE = 'pima.csv'
OUTPUTS = 1 # number of output columns in the dataset, assumes at the end of each row
TRAINSPLIT = 0.8 # 80% for training, 20% for validation
EPOCHS = 15
BATCHSIZE = 10
HIDDEN1 = 6
HIDDEN2 = 4 # set to 0 for no second hidden layer
VERBOSE = 1 # 0, 1, or 2
SEED = 7 # random seed for reproducibility
SHUFFLE = True # whether to shuffle the dataset before splitting into train and validation sets
OPTIMIZER = 'adam' # 'adam', 'sgd', 'rmsprop', etc.
LOSS = 'binary_crossentropy' # 'binary_crossentropy', 'mean_squared_error', etc
METRICS = ['accuracy'] # list of metrics to evaluate during training

