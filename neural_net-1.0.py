#!/usr/bin/env python3.7
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_docs.modeling as tfmod
import tensorflow.keras as keras
import tensorflow.keras.backend as kback
import tensorflow.keras.layers as klays
import tensorflow.keras.utils as kutils
import tensorflow.keras.initializers as kinit
import sklearn.preprocessing as procs
import pickle
import scipy.stats as stats

def swish(x,beta=1):
  return (x*kback.sigmoid(beta*x))
kutils.get_custom_objects().update({'swish':klays.Activation(swish)})

def build_model():
  model=keras.Sequential([
    klays.Dense(5,activation='relu',input_shape=[len(raw_dataset.keys())]),
    klays.Dense(256,activation='relu'),
    klays.Dense(256,activation='relu'),
    klays.Dense(128,activation='sigmoid'),
    klays.Dense(1)])
  learn_decay=keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
  optimizer=tf.keras.optimizers.RMSprop(learning_rate=learn_decay,centered=True)
  model.compile(loss='mse',optimizer=optimizer,metrics=['mse','mae'])
 return model


# DATOS DE ENTRADA
# para el aprendizaje
names=['npxls','density','edep_max','edepT','zenith','ein']
finp0='train/features_neut.dat'
data=np.loadtxt(finp0)
M=np.shape(data)[0]
data_x=data[:,0:5]
y=data[:,5]
test=y<5000
data_y=data_y[test]
data_x=data_x[test,:]

# SCALLING OR NORMALIZE
x_scaler=procs.StandardScaler()
x_scaler.fit(data_x)
x_norm=x_scaler.transform(data_x)
raw_dataset=pd.DataFrame(x_norm,columns=names)
target_scaler=procs.MaxAbsScaler()
target_scaler.fit(data_y)
label_dy=target_scaler.transform(data_y)
label_dataset=pd.DataFrame(label_dy,columns=name)

# DATOS DE ENTRADA
# para la prueba
finp1='test/features_neut.dat'
dtest=np.loadtxt(finp1)
K=np.shape(dtest)[0]
z_test=dtest[:,0:5]
z_scaler=procs.StandardScaler()
z_scaler.fit(z_test)
z_test_norm=z_scaler.transform(z_test)
raw_dataset_test=pd.DataFrame(z_test_norm,columns=names)

w_test=dtest[:,5]
test_low=w_test<5000
w_test=w_test[test_low]

model=build_model()
EPOCHS=1000
for j in range(0,3):
  early_stop=keras.callbacks.EarlyStopping(monitor='loss',patience=20)
  history=model.fit(raw_dataset,label_dataset,epochs=EPOCHS,validation_split=0.2,
                    verbose=0,callbacks=[early_stop,tfmod.EpochDots()])
  loss,mae,mse=model.evaluate(raw_dataset,label_dataset,verbose=2)
  model.summary()

  test_predict=model.predict(raw_dataset_test).flatten()
  M5=np.size(test_predict)
  test_predict=test_predict.reshape(K,1)
  test_predict=target_scaler.inverse_transform(test_predict)
  test_predict=test_predict.T[0]
  print('Datos predecidos (M): {0}'.format(np.shape(test_predict)[0]))

  # para la Regression, solo bajas energias
  df=pd.DataFrame()
  test_out=test_predict[test_low]

  df['RealData']=w_test
  df['PredictedData']=test_out
  m,b,rcor,p_value,std_err=stats.linregress(df['RealData'],df['PredictedData'])
  print('R correlacion : {0:1.2f}'.format(rcor))
  if rcor>=0.8:
    model.save('ann_model-{0:1.2f}'.format(rcor))
    pickle.dump(target_scaler,open('scaler_{0:1.2f}.pkl'.format(rcor),'wb'))
