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
import sklearn.preprocessing as procs
import pickle
import scipy.stats as stats

def swish(x,beta=1):
  return (x*kback.sigmoid(beta*x))
kutils.get_custom_objects().update({'swish':klays.Activation(swish)})

def build_model():
  model=keras.Sequential([
    klays.Dense(256,activation='swish',input_shape=[len(raw_dataset.keys())]),
    klays.Dense(256,activation='swish'),
    klays.Dense(256,activation='swish'),
    klays.Dense(256,activation='sigmoid'),
    klays.Dense(1)])
  optimizer=tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',optimizer=optimizer,metrics=['mse','mae'])
  return model

# DATOS DE ENTRADA
# para el aprendizaje
names=['Npxls','Density','Edep_max','EdepT']
fin0='train/features_neut.dat'
x=np.loadtxt(fin0)
M0=int(np.size(x)/4.0)
data_x=x.reshape((M0,4))

name=['Energy']
fin1='train/energy_neut.dat'
y=np.loadtxt(fin1)
M1=np.size(y)
data_y=y.reshape(M1,1)
test=y<5000
data_y=data_y[test]
data_x=data_x[test,:]

# SCALLING OR NORMALIZE
x_norm=procs.normalize(data_x,norm='l1')
raw_dataset=pd.DataFrame(x_norm,columns=names)
target_scaler=procs.QuantileTransformer()
target_scaler.fit(data_y)
label_dy=target_scaler.transform(data_y)
label_dataset=pd.DataFrame(label_dy,columns=name)

# DATOS DE ENTRADA
# para la prueba
fin2='test/features_neut.dat'
z=np.loadtxt(fin2)
M2=int(np.size(z)/4)
z_test=z.reshape((M2,4))
z_test_norm=procs.normalize(z_test,norm='l1')
raw_dataset_test=pd.DataFrame(z_test_norm,columns=names)

fin3='test/energy_neut.dat'
w_test=np.loadtxt(fin3)
M3=np.size(w_test)
test_low=w_test<1000
w_test=w_test[test_low]

model=build_model()
EPOCHS=100
for j in range(0,5):
  early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
  history=model.fit(raw_dataset,label_dataset,epochs=EPOCHS,validation_split=0.2,
                    verbose=0,callbacks=[early_stop,tfmod.EpochDots()])
  loss,mae,mse=model.evaluate(raw_dataset,label_dataset,verbose=2)
  model.summary()

  test_predict=model.predict(raw_dataset_test).flatten()
  M5=np.size(test_predict)
  test_predict=test_predict.reshape(M5,1)
  test_predict=target_scaler.inverse_transform(test_predict)
  test_predict=test_predict.T[0]
  print('Datos predecidos (M): {0}'.format(np.shape(test_predict)[0]))

  # para la Regression, solo bajas energias
  df=pd.DataFrame()
  test_out=test_predict[test_low]

  df['RealData']=w_test
  df['PredictedData']=test_out
  m,b,r_value,p_value,std_err=stats.linregress(df['RealData'],df['PredictedData'])
  cdet=r_value**2.0
  print('R cuadrada : {0:1.2f}'.format(cdet))
  if cdet>=0.8:
    model.save('ann_model-{0:1.2f}'.format(cdet))
    pickle.dump(target_scaler,open('scaler_{0:1.2f}.pkl'.format(cdet),'wb'))
