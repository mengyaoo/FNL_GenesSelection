# TC1 data preparation ; S.Ravichandran 04/2020 
# (saka.ravi@gmail.com; ravichandrans@mail.nih.gov)
# Github: https://github.com/ravichas/ML-TC1
#---------------------------------------------------

# submitters_id_to_project_id and "Merged_FPKM-UQ.tsv" 
# files were extracted and modified from GDC 

# Load the libraries
from __future__ import print_function

import os, sys, gzip, glob, json, time, argparse
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

from pandas.io.json import json_normalize
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau


#-----------------------------------------------------
# 1. Data Preparation                                *
#-----------------------------------------------------
# read the merged FPKM file 

df_FPKM_UQ = pd.read_csv("Merged_FPKM-UQ.tsv", low_memory=False, sep="\t")

# explore the file 
df_FPKM_UQ.iloc[0:6, 0:10]
cols = df_FPKM_UQ.columns[2:].values.tolist()
# len(cols)
# cols
# type(cols)

# submitters_id_list.txt
# is the list of submitters_id and their corresponding project_id 
# downloaded from GDC site.

# read the file 
submitters_id_to_project_id = pd.read_csv("submitters_id_to_project_id.tsv", 
                                          low_memory=False, sep="\t")

# explore the contents
submitters_id_to_project_id.columns
submitters_id_to_project_id

# There are 15 cancer types. 
submitters_id_to_project_id.mappedProject.value_counts()

# Time elapsed to transpose: ~100 seconds on a modern laptop
dft_FPKM_UQ = df_FPKM_UQ.T

print('Pre df', df_FPKM_UQ.shape)
print('After df Transpose', dft_FPKM_UQ.shape)

dft_FPKM_UQ[0:2]
dft_FPKM_UQ.index[0:2]

# We are removing the first two rows (ESG*** and gene_name rows) 
# and saving them in dftm_FPM_UQ

print('Dimension of dft_FPKM_UQ :', dft_FPKM_UQ.shape)
dftm_FPKM_UQ = dft_FPKM_UQ.drop(dft_FPKM_UQ.index[0:2], axis=0)
print('Dimension of dftm_FPKM_UQ :', dftm_FPKM_UQ.shape)

# view few entries.
dftm_FPKM_UQ.iloc[0:3, 0:15]
dft_FPKM_UQ.iloc[0:5,0:15]
dftm_FPKM_UQ.index

# Extract the Submitter ID from the index and attach 
# it as a column called Submitter ID

dftm_FPKM_UQ['submitter_id'] = dftm_FPKM_UQ.index
dftm_FPKM_UQ
dftm_FPKM_UQ = dftm_FPKM_UQ.reset_index(drop=True)

dftm_FPKM_UQ.iloc[0:3, 0:12]
dft_FPKM_UQ.iloc[2:5,0:12]

dftm_FPKM_UQ['submitter_id']
dftm_FPKM_UQ.iloc[0:5,60483]

print(submitters_id_to_project_id.iloc[0:3,0:3])
sid_list = submitters_id_to_project_id['submittedAliquot ID'].values.tolist()
type(sid_list)
sid_list.index('TCGA-04-1338-01A-01R-1564-13')

# explore the file 
dftm_FPKM_UQ.shape

dftm_FPKM_UQ['Project_id'] = ' '
for idx, val in dftm_FPKM_UQ['submitter_id'].items():
    temp_sid = sid_list.index(val)
    dftm_FPKM_UQ['Project_id'][idx] = submitters_id_to_project_id['mappedProject'][temp_sid]

# checking few random project_id and submitters_ids
dftm_FPKM_UQ[['Project_id','submitter_id']].iloc[[1,100,500,1000,2000,3000,4000],]

print(dftm_FPKM_UQ.shape)
dftm_FPKM_UQ.drop(['submitter_id'], axis = 1,  inplace=True)
print(dftm_FPKM_UQ.shape)
print(len(sid_list))

# ## Final check before moving on
dftm_FPKM_UQ.Project_id.value_counts()

print( 'before', dftm_FPKM_UQ['Project_id'][0:10] )
le = preprocessing.LabelEncoder()

# Create a label (category) encoder object
dftm_FPKM_UQ['Project_id'] = le.fit_transform(dftm_FPKM_UQ.Project_id.values)

print( 'end', dftm_FPKM_UQ['Project_id'][0:10] )

print(len(le.classes_))
print(le.classes_)

dftm_FPKM_UQ['Project_id'].value_counts()
dftm_FPKM_UQ.shape
dftm_FPKM_UQ.columns

# Use to_categorical on your labels
features = dftm_FPKM_UQ.drop(['Project_id'], axis=1)
# pandas.core.series.Series
outcome = dftm_FPKM_UQ.Project_id
print(outcome)

#-----------------------------------------------------
# 2. Scaline                                         *
#-----------------------------------------------------

sfeatures = features.div(features.sum(axis=1), axis=0)
sfeatures = sfeatures * 1000000

# log scaling
sfeatures1 = sfeatures.astype(np.float64).apply(np.log10)

# since we have negative numbers
sfeatures1[sfeatures1 < 0] = 0

# just making sure
sfeatures1.isnull().sum().sum()

# not needed
#sfeatures1.replace(np.nan,0)

# SCALING ENDS ******

# just checking
print('features few entries')
features.iloc[0:5,0:5]

print('sfeatures few entries')
sfeatures.iloc[0:5,0:5]

print('sfeatures1 few entries')
sfeatures1.iloc[0:5,0:5]

# if needed, save the files
# sfeatures1.to_csv('TC1-S1-data15.tsv', sep='\t', index=False)
# outcome.to_csv('TC1-outcome-data15.tsv', sep='\t', index=False, header = False)

# Read features and output files 
#TC1data15 = pd.read_csv("TC1-S2-data15.tsv", sep="\t", low_memory = False)

TC1data15 = sfeatures1

####### just checking
# outcome[0].value_counts()
outcome = outcome.values
outcome

def encode(data): 
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

from keras.utils import to_categorical
outcome = encode(outcome)
# outcome = np.expand_dims(outcome, axis=2)
# outcome[0].value_counts()

#-----------------------------------------------------
# 3. Train/Test split                                *
#-----------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(TC1data15, outcome, 
                                                    train_size=0.75, 
                                                    test_size=0.25, 
                                                    random_state=123, 
                                                    stratify = outcome)

#-----------------------------------------------------
# 4. CONV1D                                          *
#-----------------------------------------------------
# parameters  
activation='relu'
batch_size=20
# Number of sites
classes=15
drop = 0.1
feature_subsample = 0
loss='categorical_crossentropy'
# metrics='accuracy'
out_act='softmax'
pool=[1, 10]
# optimizer='sgd'
shuffle = False 
epochs=400

optimizer = optimizers.SGD(lr=0.1)
metrics = ['acc']

# X_train shape: (3375, 60483)
# X_test shape:  (1125, 60483)
# Y_train shape: (3375,1)
# Y_test shape:  (1125,1)

# 60483
x_train_len = X_train.shape[1]   

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# X_train shape: (3375, 60483, 1)
# X_test shape:  (1125, 60483, 1)


filters = 128 
filter_len = 20 
stride = 1 

# inside pool_list loop
pool_list = [1,10]

K.clear_session()

model = Sequential()

# model.add  CONV1D
model.add(Conv1D(filters = filters, 
                 kernel_size = filter_len, 
                 strides = stride, 
                 padding='valid', 
                 input_shape=(x_train_len, 1)))

# x_train_len = 60,483
# Activation
model.add(Activation('relu'))

# MaxPooling
model.add(MaxPooling1D(pool_size = 1))

filters = 128
filter_len = 10 
stride = 1 
# Conv1D
model.add(Conv1D(filters=filters, 
                 kernel_size=filter_len, 
                 strides=stride, 
                 padding='valid'))
# Activation
model.add(Activation('relu'))

# MaxPooling
model.add(MaxPooling1D(pool_size = 10))
model.add(Flatten())

model.add(Dense(200))

# activation 
# model.add(Activation('relu')) # SR
model.add(Activation(activation))
#dropout
model.add(Dropout(0.1))

model.add(Dense(20))
# activation
# model.add(Activation('relu')) # SR
model.add(Activation(activation))

#dropout
model.add(Dropout(0.1))

model.add(Dense(15))
model.add(Activation(out_act))

model.compile( loss= loss, 
              optimizer = optimizer, 
              metrics = metrics )
model.summary()

# save
save = '.'
output_dir = "/data/ravichandrans/TC1/Modeling"
          
output_dir = save 
if not os.path.exists(output_dir): 
	os.makedirs(output_dir)

model_name = 'tc1'
path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
checkpointer = ModelCheckpoint(filepath=path, 
                               verbose=1, 
                               save_weights_only=False, 
                               save_best_only=True)
          
csv_logger = CSVLogger('{}/training.log'.format(output_dir))

# SR: change epsilon to min_delta
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=10, 
                              verbose=1, mode='auto', 
                              min_delta=0.0001, 
                              cooldown=0, 
                              min_lr=0)
# batch_size = 20 
history = model.fit(X_train, Y_train, batch_size=batch_size, 
                    epochs=epochs, verbose=1, validation_data=(X_test, Y_test), 
                    callbacks = [checkpointer, csv_logger, reduce_lr])

score = model.evaluate(X_test, Y_test, verbose=0)

#-----------------------------------------------------
# 5. Finish up save model weights                    *
#-----------------------------------------------------

print('Test score:', score[0])
print('Test accuracy:', score[1])

# serialize weights to HDF5
model.save_weights("{}/{}.model.h5".format(output_dir, model_name))
print("Saved model to disk")

# load weights into new model
loaded_model_yaml.load_weights('{}/{}.model.h5'.format(output_dir, model_name))
print("Loaded yaml model from disk")

# evaluate loaded model on test data
loaded_model_yaml.compile(loss=loss,optimizer=optimizer, metrics=metrics) 
score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

print('yaml Test score:', score_yaml[0])
print('yaml Test accuracy:', score_yaml[1])

print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1]*100))

