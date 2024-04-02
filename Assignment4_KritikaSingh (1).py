#!/usr/bin/env python
# coding: utf-8

# ### Import the libraries and load the training and testing data and training and testing labels


from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd

data_dir = 'Assignment4'


#subject_data = np.loadtxt(f'{data_dir}/train/subject_train.txt')
# Samples
train_data = np.zeros((2000, 48, 4))
train_data[:,:,0] = pd.read_csv('C:/Users/Mohith/Downloads/Assignment4/train/Heart.csv', header=None).values
train_data[:,:,1]  = pd.read_csv('C:/Users/Mohith/Downloads/Assignment4/train/Temperature.csv', header=None).values
train_data[:,:,2]  = pd.read_csv('C:/Users/Mohith/Downloads/Assignment4/train/Respiration.csv', header=None).values
train_data[:,:,3]  = pd.read_csv('C:/Users/Mohith/Downloads/Assignment4/train/Glucose.csv', header=None).values

test_data = np.zeros((200, 48, 4))
test_data[:,:,0] = pd.read_csv('C:/Users/Mohith/Downloads/Assignment4/test_2/Heart.csv', header=None).values
test_data[:,:,1] = pd.read_csv('C:/Users/Mohith/Downloads/Assignment4/test_2/Temperature.csv', header=None).values
test_data[:,:,2] = pd.read_csv('C:/Users/Mohith/Downloads/Assignment4/test_2/Respiration.csv', header=None).values
test_data[:,:,3] = pd.read_csv('C:/Users/Mohith/Downloads/Assignment4/test_2/Glucose.csv', header=None).values


# labels
train_labels = np.loadtxt('Assignment4/y_train.csv', delimiter=',')
train_labels -= np.min(train_labels)
test_labels = np.loadtxt('Assignment4/y_test.csv', delimiter=',')
test_labels -= np.min(test_labels )


train_data.shape


test_data.shape


train_labels


# ### Convert to categorical class labels of multiclass classification


from keras.utils import np_utils
NB_CLASSES = 4 #number of classes
print('shape of y_train and y_test before categorical conversion')
print(train_labels.shape)
print(test_labels.shape)
y_train= np_utils.to_categorical(train_labels, NB_CLASSES)
y_test= np_utils.to_categorical(test_labels, NB_CLASSES)
print('shape of y_train and y_test after categorical conversion')
print(y_train.shape)
print(y_test.shape)


# ### Defining Hyperparameter

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from sklearn.metrics import roc_auc_score
from functools import partial


cnn_space = {
            'num_kernel': scope.int(hp.quniform('num_kernel',6, 100, 1)),
            'kernel_size': scope.int(hp.quniform('kernel_size',3,5,1)),
            'size_pooling': scope.int(hp.quniform('size_pooling',2,4,1)),
            'dropout': hp.uniform('dropout',.20,.50),
            'batch_size' : scope.int(hp.quniform('batch_size',20,30,50)),
            'optimizer': hp.choice('optimizer',['Adam', 'SGD', 'RMSprop']),
            'strides': scope.int(hp.quniform('strides',1,2,1))
        }


# The function then initializes a Sequential model, adds a Conv1D layer with the number of kernels and kernel size specified in the params dictionary, followed by a Dropout layer, a MaxPooling1D layer with window size 4 and stride 1, another Dropout layer, and a Flatten layer. Finally, it adds a Dense output layer with one node and sigmoid activation. The loss function is binary cross-entropy and the optimizer is specified in the params dictionary. The model is compiled, and then trained on the input data x_train and y_train for 100 epochs, with early stopping criteria of patience = 3.
# 
# After training, the function finds the validation loss of the trained model and returns a dictionary containing the validation loss, a status code indicating the status of the optimization (STATUS_OK indicates successful optimization), and the trained model itself. This function is intended to be used with hyperparameter optimization algorithms such as Hyperopt to search for the best combination of hyperparameters that minimizes the validation loss of the CNN model.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
    
def cnn2_model(params,x_train,y_train):

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 3)
    INPUT_SHAPE = (x_train.shape[1],x_train.shape[2])
    model = Sequential()
    model.add(Conv1D(params['num_kernel'], params['kernel_size'], input_shape = INPUT_SHAPE, activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=4, strides=1))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(4, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    '''
    suggest to make verbose = 0 so that it doesn't print the progress of model training for every iteration of hyper parameter optimization
    '''
    history = model.fit(x_train,y_train,batch_size=params['batch_size'], epochs = 100, verbose = 0,
                                validation_split = 0.2,callbacks=[early_stopping_monitor])
    
    keys = history.history.keys()
    res = [i for i in keys if ('val' in i and 'loss' in i)]
    val_loss = min(history.history[res[0]])
    return {'loss': val_loss, 'status': STATUS_OK, 'Trained_Model': model}


# This code performs hyperparameter optimization using the Tree-structured Parzen Estimator (TPE) algorithm from the hyperopt library. The cnn2_model function is used as the objective function to minimize, and cnn_space is a dictionary that specifies the hyperparameter search space. The search is performed using 10 iterations (max_evals=10), and the results of each iteration are stored in the trials object.
# 
# The best set of hyperparameters found by the search is returned as a dictionary, which is printed out by the print statement.


trials =Trials()
best_model = fmin(partial(cnn2_model,x_train=train_data,y_train=y_train),cnn_space,algo=tpe.suggest,max_evals=10,trials=trials)
print(best_model)


# This code is accessing the first trial in the list of trials generated by the hyperparameter optimization process.
# 
# trials is an object that stores the information about each trial during the optimization process, including the hyperparameters used and the loss value of the model trained with those hyperparameters. The for trial in trials statement loops through each trial, and [trial for trial in trials] creates a list of all the trials.
# 
# The indexing [0] at the end selects the first trial in the list, which is returned as an object that can be further analyzed or used to extract information such as the best hyperparameters found during the optimization.


[trial for trial in trials][0]


# This function(getBestModelfromTrials) is used to extract the best model obtained from the hyperparameter optimization trials and save it for future use


def getBestModelfromTrials(trials,modelname):
    
    # extracts all valid iterations of the hyperoptimization 
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    
    # extracts obj. function value in all valid iterations of the hyperoptimization 
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    
    # find the one with lowest obj. function
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    
    # extracts the model corresponding to the lowest obj. function
    Model = best_trial_obj['result']['Trained_Model']
    model_json=Model.to_json()
    with open(modelname+'.json','w') as json_file:
         json_file.write(model_json)
    Model.save_weights(modelname+'.h5')
    return best_trial_obj['result']['Trained_Model']


model = getBestModelfromTrials(trials,'hyperopt_model')


# This function takes in a trained model, test data, and test class labels as input. It then uses the trained model to make predictions on the test data and calculates the accuracy of the predictions by comparing the predicted class labels with the actual class labels. The accuracy is calculated using the accuracy_score function from the scikit-learn library. Finally, the function prints the testing accuracy.

from sklearn.metrics import accuracy_score
def GetAccuracy(model,test_data,test_class):
    pred_p = model.predict(test_data) # predict probabilities
    pred_classes = np.argmax(pred_p , axis=1) # predicted class labels
    y_classes = np.argmax(test_class, axis=1) # convert categorial class labels to actual class labels
    print('testing accuracy',accuracy_score(y_classes,pred_classes))    


GetAccuracy(model,test_data,y_test)


# This code is loading a trained model from a JSON file and its corresponding weight from an H5 file, creating a Keras model object from the JSON file and loading the weights into it. Finally, it is calling the GetAccuracy function to evaluate the accuracy of the loaded model on the test dataset.

from tensorflow.keras.models import model_from_json
f = open('hyperopt_model.json','r')
model_ = f.read()
f.close()
model_ = model_from_json(model_)
model_.load_weights("hyperopt_model.h5")
GetAccuracy(model_,test_data,y_test)


trials1 =Trials()
best_model = fmin(partial(cnn2_model,x_train=train_data,y_train=y_train),cnn_space,algo=tpe.suggest,max_evals=50,trials=trials1)
print(best_model)

model = getBestModelfromTrials(trials1,'hyperopt_model_new')
GetAccuracy(model,test_data,y_test)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from functools import partial
import numpy as np
import random
import matplotlib.pyplot as plt
import math


# ### Ploting Graphs

# The first subplot shows the histogram of the random variables generated from the uniform distribution. The second to fourth subplots show the histograms of the random variables generated from the quniform distribution with different quantization levels (q=2, q=4, q=6). As the quantization level increases, the histogram becomes more discrete and the distribution becomes more spiky.

# In[23]:


fig, ax = plt.subplots(1, 4,figsize=(15, 4))


low=30
high=120
from scipy.stats import uniform
rv = uniform.rvs(low, high,size=1000)
ax[0].hist(rv, density=True, histtype='stepfilled', alpha=0.2,bins=100)
ax[0].set_title('low=6,high,q=10,uniform')

q=2
rv1=np.round(rv/q)*q
ax[1].hist(rv1, density=True, histtype='stepfilled', alpha=0.2,bins=100)
ax[1].set_title('low=6,high=100,q=2,quniform')

q=4
rv2=np.round(rv/q)*q
ax[2].hist(rv2, density=True, histtype='stepfilled', alpha=0.2,bins=100)
ax[2].set_title('low=6,high=100,q=4,quniform')


q=6
rv3=np.round(rv/q)*q
ax[3].hist(rv3, density=True, histtype='stepfilled', alpha=0.2,bins=100)
ax[3].set_title('low=6,high=10,q=6,quniform')


plt.show()


# Each subplot shows a histogram of the random numbers with a varying number of bins (100 for the uniform distribution and 200 for the log-uniform distribution with quantization). The density=True parameter indicates that the histogram is normalized, and histtype='stepfilled' and alpha=0.2 set the style of the histogram bars. 


fig, ax = plt.subplots(1, 4,figsize=(15, 5))

low=3
high=6
from scipy.stats import loguniform
rv = loguniform.rvs(math.exp(low), math.exp(high),size=1000)
ax[0].hist(rv, density=True, histtype='stepfilled', alpha=0.2,bins=100)
ax[0].set_title("low=3,high=6,loguniform")


q = 2
rv1 = np.round(rv/q)*q
ax[1].hist(rv1, density=True, histtype='stepfilled', alpha=0.2,bins=200)
ax[1].set_title("low=3,high=6,q=2,qloguniform")

q = 6
rv2 = np.round(rv/q)*q
ax[2].hist(rv2, density=True, histtype='stepfilled', alpha=0.2,bins=200)
ax[2].set_title("low=3,high=6,q=6,qloguniform")


q = 8
rv3 = np.round(rv/q)*q
ax[3].hist(rv3, density=True, histtype='stepfilled', alpha=0.2,bins=200)
ax[3].set_title("low=3,high=6,q=8,qloguniform")

plt.show()




